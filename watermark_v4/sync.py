"""V4 的 FFT 同步导频：几何变换的检测与校正。

**要解决的问题。** DCT 水印是按 128×128 网格对齐嵌入的。图片一旦被缩放或
旋转，网格就错位了，即便水印信号本身还在，也无法按原格点读出。
本模块负责在解码前先回答："这张图相对原图被缩放/旋转/平移了多少？"

**做法。** 嵌入时在亮度通道叠加四个已知频率、已知相位的二维正弦波（导频）。
它们在频域中是四个尖锐的峰，构成一张固定的"星座图"。图片被缩放时
频谱峰随之向内/外移动，被旋转时整体转过相同角度——因此只要在频谱里
找到这四个峰的实际位置，就能反解出缩放比与旋转角。

**三级搜索。** 穷举所有 (旋转, 缩放) 组合代价太高，故分级进行：

1. **粗搜**：13 个旋转角 × 14 个缩放比 = 182 个假设，大步长扫全域；
2. **精搜**：在粗搜的最优点附近以 0.5° / 0.02 的细步长复扫；
3. **平移估计**：几何摆正后，再用导频的相位求出 128 网格的起始偏移。

前两级只能定出"缩放和旋转"，定不出平移——傅里叶幅度谱天生对平移不变
（平移只改变相位）。所以第 3 级专门去看**相位**，这正是导频要固定初相
（见 :func:`_pilot_phases`）的原因。

**保守优先。** 宁可返回 ``None`` 让上层走别的检测路径，也不输出一个错误
的几何估计——错误的校正会把后续 DCT 解码彻底带偏，比不校正更糟。
因此设了多道否决：支撑峰不足 3 个、最优与次优过于接近（歧义）、
置信度低于下限，任一条命中都直接放弃。
"""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from time import monotonic

import cv2
import numpy as np
from PIL import Image

from .config import V4Config


# 导频分量个数，与 config.pilot_frequency_vectors 长度一致
PILOT_COMPONENT_COUNT = 4
# 平移估计的规模上限：超过就跳过这一步。该步骤要对每个候选偏移做相位比对，
# 代价随像素数增长，对超大图必须设闸，否则单次检测会拖到分钟级。
MAX_OFFSET_ANALYSIS_PIXELS = 4_000_000
MAX_OFFSET_ANALYSIS_SIDE = 4096
# 生成/嵌入导频时的分块行数，控制中间数组的内存峰值
PILOT_ROW_CHUNK = 256
TWO_PI = 2.0 * np.pi
# 判定"这是一个真实峰"的阈值：峰值必须达到周围背景中位数的 2.5 倍。
# 用相对比值而非绝对幅度，是为了不受图片整体对比度影响。
PEAK_SUPPORT_RATIO = 2.5

# ---- 粗搜网格 ----
# 旋转：-12°~+12°，步长 2°。范围有限是因为超出这个幅度的旋转，
# DCT 分块本身也已严重形变，即使同步成功也解不出码字。
COARSE_ROTATIONS = tuple(float(value) for value in range(-12, 13, 2))
# 缩放：0.5~2.0 倍。取值不均匀分布，在 1.0 附近（最常见）更密，
# 并特意包含 0.5/0.75/1.25/1.5/2.0 这些社交平台常用的缩放档位。
COARSE_SCALES = (
    0.5,
    0.65,
    0.75,
    0.8,
    0.95,
    1.0,
    1.1,
    1.25,
    1.4,
    1.5,
    1.55,
    1.7,
    1.85,
    2.0,
)
# ---- 精搜步长 ----
REFINE_ROTATION_STEP = 0.5
REFINE_SCALE_STEP = 0.02
# 歧义判定：次优假设得分若在最优的 95% 以内，说明无法可靠区分，放弃
AMBIGUOUS_SCORE_MARGIN = 0.05
# 置信度下限，低于此值宁可返回 None
MIN_SYNC_CONFIDENCE = 0.02


@dataclass(frozen=True, slots=True)
class PilotPeakEvidence:
    """单个导频分量的峰检测证据。

    实数信号的频谱是**共轭对称**的，所以每个正弦分量必然产生关于原点对称的
    一对峰。这里同时记录正负两侧，并要求**两侧都达标**才算 ``supported``——
    单侧强而另一侧弱，通常说明那是图像自身纹理造成的偶然峰，而非导频。
    """

    component_index: int
    # 峰值与周围背景中位数之比（正频侧 / 负频侧）
    positive_ratio: float
    negative_ratio: float
    # 两侧比值都达到 PEAK_SUPPORT_RATIO 时为真
    supported: bool
    # 实际找到峰的频谱格点坐标（行, 列），用于评估定位误差
    positive_bin: tuple[int, int]
    negative_bin: tuple[int, int]

    def __post_init__(self) -> None:
        """校验分量下标、两侧比值与格点坐标的类型和取值范围。

        比值要求**有限且非负**：它是"峰值 ÷ 背景中位数"，负值或 NaN 说明
        上游计算出了问题，放行会让后续的阈值比较静默失效。
        """
        if type(self.component_index) is not int or self.component_index not in range(4):
            raise ValueError("component index must be an integer from 0 through 3")
        for ratio in (self.positive_ratio, self.negative_ratio):
            if type(ratio) is not float or not np.isfinite(ratio) or ratio < 0.0:
                raise ValueError("ratio values must be finite nonnegative floats")
        if type(self.supported) is not bool:
            raise TypeError("supported must be a boolean")
        for bin_value in (self.positive_bin, self.negative_bin):
            if (
                type(bin_value) is not tuple
                or len(bin_value) != 2
                or any(type(value) is not int for value in bin_value)
            ):
                raise ValueError("bin values must be integer pairs")


@dataclass(frozen=True, slots=True)
class SyncEstimate:
    """同步估计结果：把图片还原到原始几何所需的变换参数。

    .. important::
        ``rotation_degrees`` 是**校正角**（需要把图转回多少度），
        与内部使用的"频谱旋转角"符号相反，见 :func:`detect_pilot` 末尾的取负。

    ``offset_x`` / ``offset_y`` 只在图片几乎未旋转时才会给出（见
    :func:`detect_pilot`），因此可能为 ``None``——旋转状态下网格是斜的，
    "水平/垂直偏移"这个概念本身就不成立。
    """

    rotation_degrees: float
    scale: float
    # 综合置信度，由峰强度与最优/次优差距共同决定
    confidence: float
    supported_peaks: int
    # 本次共评估了多少个假设，用于性能观测
    evaluated_hypotheses: int
    elapsed_seconds: float
    # 128 网格的起始偏移，取值 0~127
    offset_x: int | None = None
    offset_y: int | None = None

    def __post_init__(self) -> None:
        """校验各字段落在算法约定的取值范围内。

        区间与 ``COARSE_ROTATIONS`` / ``COARSE_SCALES`` 的覆盖范围一致；
        偏移限定在 ``[0, 128)`` 是因为它表达的是网格内的相对位置，
        超出一个网格周期的部分没有意义。
        """
        real_fields = (
            ("rotation", self.rotation_degrees),
            ("scale", self.scale),
            ("confidence", self.confidence),
            ("elapsed", self.elapsed_seconds),
        )
        for name, value in real_fields:
            if type(value) is not float or not np.isfinite(value):
                raise TypeError(f"{name} must be a finite float")
        if not -12.0 <= self.rotation_degrees <= 12.0:
            raise ValueError("rotation must be between -12 and 12 degrees")
        if not 0.5 <= self.scale <= 2.0:
            raise ValueError("scale must be between 0.5 and 2")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.elapsed_seconds < 0.0:
            raise ValueError("elapsed must be nonnegative")
        if type(self.supported_peaks) is not int or not 0 <= self.supported_peaks <= 4:
            raise ValueError("supported peaks must be an integer from 0 through 4")
        if type(self.evaluated_hypotheses) is not int or self.evaluated_hypotheses <= 0:
            raise ValueError("evaluated hypotheses must be a positive integer")
        for name, value in (("offset_x", self.offset_x), ("offset_y", self.offset_y)):
            if value is not None and (type(value) is not int or not 0 <= value < 128):
                raise ValueError(f"{name} must be None or an integer from 0 through 127")


@lru_cache(maxsize=1)
def _pilot_phases(codec: str) -> tuple[float, ...]:
    """由 codec 名派生出四个导频分量的初相，取值 ``[0, 2π)``。

    相位是**确定性**的（SHA-256 派生），嵌入端与提取端无需交换任何数据。
    这一点是平移估计的基础：知道理论初相，才能拿实测相位与之相减，
    从相位差反推出图片被平移了多少。

    取摘要前 8 字节当作 64 位无符号整数，再线性映射到 ``[0, 2π)``。

    用不同的初相而非全零，是为了避免四个分量在原点处同时取极值、
    叠加出一个视觉上可见的亮点。
    """
    if type(codec) is not str or not codec:
        raise TypeError("codec must be a nonempty string")
    return tuple(
        int.from_bytes(
            hashlib.sha256(
                f"{codec}:pilot-phase:{index}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        * (TWO_PI / 2**64)
        for index in range(PILOT_COMPONENT_COUNT)
    )


def pilot_signal(height: int, width: int, config: V4Config) -> np.ndarray:
    """生成指定尺寸的导频信号（四个正弦波之和），供叠加到亮度通道。

    :return: ``(height, width)`` 的浮点数组，取值在 ±4×振幅 之间。

    按行分块生成，避免为大图一次性分配整块中间数组。
    """
    _validate_dimensions(height, width)
    _validate_config(config)

    # x 坐标全图共用，提到循环外只算一次
    x = np.arange(width, dtype=np.float64)[None, :]
    signal = np.zeros((height, width), dtype=np.float64)
    phases = _pilot_phases(config.codec)
    for row_start in range(0, height, PILOT_ROW_CHUNK):
        row_stop = min(row_start + PILOT_ROW_CHUNK, height)
        signal[row_start:row_stop] = _pilot_signal_rows(
            row_start,
            row_stop,
            x,
            phases,
            config,
        )
    if not np.isfinite(signal).all():
        raise ValueError("pilot signal must contain only finite values")
    return signal


def embed_pilot(image: Image.Image, config: V4Config) -> Image.Image:
    """把同步导频叠加到图片的亮度通道，返回新图片。

    这是 V4 嵌入链路的**第一步**，必须在 DCT 码字之前完成：
    导频要打在干净的图上，否则码字调制会削弱它在频域中的峰值。

    导频振幅默认 0.75（亮度量纲，满量程 255），远低于人眼在自然图像中的
    察觉阈值；但因为它在频域高度集中于四个点，信噪比反而很高，易于检出。
    这正是频域水印"能量分散于空域、集中于频域"的典型优势。

    按行分块处理，每块独立完成 RGB→YCrCb→叠加→RGB 的往返。
    """
    _validate_image(image)
    _validate_config(config)

    source = np.asarray(image)
    output = source.copy()
    x = np.arange(image.width, dtype=np.float64)[None, :]
    phases = _pilot_phases(config.codec)
    for row_start in range(0, image.height, PILOT_ROW_CHUNK):
        row_stop = min(row_start + PILOT_ROW_CHUNK, image.height)
        rgb = source[row_start:row_stop, :, :3]
        # 用 float32 转换：导频振幅不足 1 个灰阶，若在 uint8 上做色彩空间
        # 往返，量化误差会把这点微弱信号直接抹平。
        ycrcb = cv2.cvtColor(
            rgb.astype(np.float32),
            cv2.COLOR_RGB2YCrCb,
        )
        # 注意 y 坐标用的是**全图绝对行号**（row_start 起算），
        # 这样分块之间的正弦相位才能连续，拼起来才是一个完整的平面波。
        signal = _pilot_signal_rows(
            row_start,
            row_stop,
            x,
            phases,
            config,
        )
        ycrcb[..., 0] = ycrcb[..., 0] + signal
        converted_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        output[row_start:row_stop, :, :3] = np.clip(
            np.rint(converted_rgb),
            0,
            255,
        ).astype(np.uint8)

    return Image.fromarray(output)


def _pilot_signal_rows(
    row_start: int,
    row_stop: int,
    x: np.ndarray,
    phases: tuple[float, ...],
    config: V4Config,
) -> np.ndarray:
    """生成指定行区间的导频信号：四个二维正弦平面波之和。

    每个分量形如 ``A · sin(2π(fx·x + fy·y) + φ)``，是一列沿
    ``(fx, fy)`` 方向传播的平面波。频域中它对应一对共轭峰。

    ``x`` 为行向量 ``(1, W)``、``y`` 为列向量 ``(H, 1)``，
    相加时 NumPy 自动广播成 ``(H, W)`` 的完整网格，无需显式构造二维坐标。
    """
    y = np.arange(row_start, row_stop, dtype=np.float64)[:, None]
    chunk = np.zeros((row_stop - row_start, x.shape[1]), dtype=np.float64)
    for (frequency_x, frequency_y), phase in zip(
        config.pilot_frequency_vectors,
        phases,
    ):
        chunk += config.pilot_amplitude * np.sin(
            TWO_PI * (frequency_x * x + frequency_y * y) + phase
        )
    return chunk


def _analysis_spectrum(
    image: Image.Image,
    config: V4Config,
    deadline: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """计算用于峰搜索的频谱与幅度谱。

    :return: ``(复数频谱, 幅度谱)``。幅度谱用于找峰，复数谱保留相位信息
        供 :func:`_phase_aligned_ratio` 做相位相干检测。

    四步预处理，每步都不可省：

    1. **转灰度**——导频只嵌在亮度上；
    2. **限制尺寸**——超大图先缩到 ``analysis_max_side``，控制 FFT 耗时。
       缩放会等比改变频谱峰的位置，后续计算已按 ``图宽/分析宽`` 折算回去；
    3. **去均值**——直流分量（图片平均亮度）在频谱中心是一个巨大的峰，
       不去掉会通过频谱泄漏抬高整个背景，淹没导频峰；
    4. **加汉宁窗**——FFT 隐含"图像左右/上下无缝循环"的假设，
       而真实图片的边界是突变的，这个虚假的不连续会在频域产生十字形拖尾。
       窗函数把边界平滑压到零，消除拖尾。

    ``deadline`` 在各耗时步骤之间反复检查，使超时能被及时中断，
    而不是等整个 FFT 算完才发现已经超时。
    """
    _validate_image(image)
    _validate_config(config)
    _validate_deadline(deadline)
    _check_deadline(deadline)

    rgb = np.asarray(image)[..., :3]
    luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _check_deadline(deadline)
    if max(image.size) > config.analysis_max_side:
        resize_scale = config.analysis_max_side / max(image.size)
        analysis_width = max(1, round(image.width * resize_scale))
        analysis_height = max(1, round(image.height * resize_scale))
        # INTER_AREA 是缩小图像的最佳选择：它做的是区域平均，
        # 相当于先低通再采样，能抑制混叠——若用双线性/最近邻，
        # 高频混叠会在频谱中造出假峰。
        luminance = cv2.resize(
            luminance,
            (analysis_width, analysis_height),
            interpolation=cv2.INTER_AREA,
        )
    values = luminance.astype(np.float64)
    values -= np.mean(values)
    # 行、列方向各加一维汉宁窗，外积即二维窗
    window_y = np.hanning(values.shape[0])[:, None]
    window_x = np.hanning(values.shape[1])[None, :]
    windowed = values * window_y * window_x
    _check_deadline(deadline)
    # fftshift 把零频从数组角落搬到中心，这样峰的坐标就能用
    # "中心 + 频率×尺寸" 直观地算出来。
    spectrum = np.fft.fftshift(np.fft.fft2(windowed))
    magnitude = np.abs(spectrum).astype(np.float64, copy=False)
    if not np.isfinite(spectrum).all() or not np.isfinite(magnitude).all():
        raise ValueError("analysis spectrum must contain only finite values")
    _check_deadline(deadline)
    return spectrum, magnitude


def pilot_peak_evidence(
    image: Image.Image,
    config: V4Config,
    rotation_degrees: float = 0.0,
    scale: float = 1.0,
    deadline: float | None = None,
) -> tuple[PilotPeakEvidence, ...]:
    """在给定的旋转/缩放假设下，逐个检查四个导频峰是否存在。

    :param rotation_degrees: 待检验的**频谱旋转角**假设。
    :param scale: 待检验的缩放比假设。
    :return: 四条 :class:`PilotPeakEvidence`，每个导频分量一条。

    这是对外暴露的诊断接口（内部搜索走的是更轻量的
    :func:`_score_sync_hypothesis`），用于排查"某张图为什么同步失败"。

    对每个分量：先按假设预测峰应该出现在频谱的哪个格点，
    再去那个位置附近实测峰强。四个都对得上，假设才成立。
    """
    _validate_image(image)
    _validate_config(config)
    rotation = _validated_rotation(rotation_degrees)
    scale_value = _validated_scale(scale)
    _validate_deadline(deadline)
    _check_deadline(deadline)
    spectrum, magnitude = _analysis_spectrum(image, config, deadline)

    analysis_height, analysis_width = magnitude.shape
    center_y = analysis_height // 2
    center_x = analysis_width // 2
    radians = np.deg2rad(rotation)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    evidence = []
    for component_index, (frequency_x, frequency_y) in enumerate(
        config.pilot_frequency_vectors
    ):
        _check_deadline(deadline)
        # 预测峰位，两步变换：
        #   1. 旋转 —— 标准二维旋转矩阵作用于频率向量；
        #   2. 缩放 —— 图像放大 s 倍，其频谱**向内收缩** 1/s 倍，故此处是除法。
        rotated_x = (cosine * frequency_x - sine * frequency_y) / scale_value
        rotated_y = (sine * frequency_x + cosine * frequency_y) / scale_value
        # 再折算到分析尺寸的坐标系（图片可能已被缩小以控制 FFT 耗时）
        analysis_x = rotated_x * image.width / analysis_width
        analysis_y = rotated_y * image.height / analysis_height
        # 实信号频谱共轭对称，正负两侧各有一个峰，关于中心对称
        predicted_positive = (
            round(center_y + analysis_y * analysis_height),
            round(center_x + analysis_x * analysis_width),
        )
        predicted_negative = (
            round(center_y - analysis_y * analysis_height),
            round(center_x - analysis_x * analysis_width),
        )
        # 0.5 是奈奎斯特频率（归一化后的采样上限）。预测位置超出它意味着
        # 该分量在当前假设下已被混叠，测到的任何峰都不可信，直接判 0。
        if abs(analysis_x) >= 0.5 or abs(analysis_y) >= 0.5:
            positive_ratio = 0.0
            negative_ratio = 0.0
            positive_bin = predicted_positive
            negative_bin = predicted_negative
        else:
            positive_ratio, positive_bin = _peak_ratio(
                magnitude, *predicted_positive
            )
            negative_ratio, negative_bin = _peak_ratio(
                magnitude, *predicted_negative
            )
            # 只有在图片未被缩小分析时（分析谱尺寸 == 原图尺寸），
            # 理论相位才与实测相位可比，相位相干检测才有意义。
            # 它能在幅度峰被噪声压低时救回来，故与幅度比取较大者。
            if magnitude.shape == (image.height, image.width):
                phase_ratio = _phase_aligned_ratio(
                    spectrum,
                    predicted_positive[0],
                    predicted_positive[1],
                    frequency_x,
                    frequency_y,
                    phases=_pilot_phases(config.codec),
                    component_index=component_index,
                )
                positive_ratio = max(positive_ratio, phase_ratio)
                negative_ratio = max(negative_ratio, phase_ratio)
        evidence.append(
            PilotPeakEvidence(
                component_index=component_index,
                positive_ratio=float(positive_ratio),
                negative_ratio=float(negative_ratio),
                supported=(
                    positive_ratio >= PEAK_SUPPORT_RATIO
                    and negative_ratio >= PEAK_SUPPORT_RATIO
                ),
                positive_bin=positive_bin,
                negative_bin=negative_bin,
            )
        )
    return tuple(evidence)


@dataclass(frozen=True, slots=True)
class _SyncHypothesis:
    """一个待评估的 (旋转, 缩放) 假设及其评分。"""

    # 频谱域的旋转角（与图像校正角符号相反）
    spectral_rotation: float
    scale: float
    # 四个分量中有几个达到支撑阈值
    supported_peaks: int
    # 综合得分，见 _score_sync_hypothesis
    score: float
    # 四个分量各自的峰值比
    ratios: tuple[float, ...]
    # 实测峰位与预测峰位的平方距离之和，越小说明假设越准
    localization_error: float


def detect_pilot(
    image: Image.Image,
    config: V4Config,
    deadline: float | None = None,
) -> SyncEstimate | None:
    """检测同步导频，估计图片相对原图的几何变换。

    :return: 成功则返回 :class:`SyncEstimate`，任一否决条件命中则 ``None``。
    :raises TimeoutError: 超过 ``deadline``。

    流程：粗搜 182 个假设 → 取两个候选峰值模式 → 各自邻域精搜 →
    多重否决 → 几何近乎正立时再估平移。

    **为什么要保留两个粗搜候选？** 频谱中有时会出现一个由图像自身纹理
    造成的强假峰，得分甚至超过真峰。只精修得分第一名就会被它带偏。
    因此额外保留一个与第一名"足够远"（旋转差 ≥3° 或缩放差 ≥0.10）的候选，
    两个邻域都精修一遍，让真峰有机会在精修后反超。
    """
    _validate_image(image)
    _validate_config(config)
    _validate_deadline(deadline)
    started = monotonic()
    _check_deadline(deadline)
    spectrum, magnitude = _analysis_spectrum(image, config, deadline)

    # ---- 第一级：粗搜，13 × 14 = 182 个假设 ----
    # search_radius=8：粗搜步长大，预测位置本就偏差较多，
    # 需要在较宽的邻域里找峰，否则真峰会因为差几个格点而被漏掉。
    coarse_hypotheses: list[_SyncHypothesis] = []
    for rotation in COARSE_ROTATIONS:
        for scale in COARSE_SCALES:
            _check_deadline(deadline)
            coarse_hypotheses.append(
                _score_sync_hypothesis(
                    spectrum,
                    magnitude,
                    image.size,
                    config,
                    rotation,
                    scale,
                    search_radius=8,
                )
            )
    ranked_coarse = sorted(
        coarse_hypotheses,
        key=_hypothesis_sort_key,
        reverse=True,
    )
    # 取两个"模式"：第一名，以及排名最高的、与第一名相距足够远的那个。
    # 距离阈值确保第二个是**另一个独立的峰**，而不是第一名旁边的邻居
    # ——否则两次精修会扫同一片区域，白费一倍时间。
    coarse_modes = [ranked_coarse[0]]
    for item in ranked_coarse[1:]:
        if (
            abs(item.spectral_rotation - coarse_modes[0].spectral_rotation) >= 3.0
            or abs(item.scale - coarse_modes[0].scale) >= 0.10
        ):
            coarse_modes.append(item)
            break

    # ---- 第二级：在每个候选模式邻域内精搜 ----
    # search_radius=1：此时预测位置已经很准，收窄搜索半径可以避免
    # 误把邻近的噪声峰当成目标。
    refined_hypotheses: list[_SyncHypothesis] = []
    # 两个模式的邻域可能重叠，用 seen 去重避免重复计算
    seen: set[tuple[float, float]] = set()
    for coarse_mode in coarse_modes:
        # ±2° / ±0.1 的邻域，并夹在全局合法范围内。
        # 加 1e-9 是为了抵消浮点误差，保证 arange 能取到闭区间右端点。
        refine_rotations = np.arange(
            max(-12.0, coarse_mode.spectral_rotation - 2.0),
            min(12.0, coarse_mode.spectral_rotation + 2.0) + 1e-9,
            REFINE_ROTATION_STEP,
        )
        refine_scales = np.arange(
            max(0.5, coarse_mode.scale - 0.1),
            min(2.0, coarse_mode.scale + 0.1) + 1e-9,
            REFINE_SCALE_STEP,
        )
        for rotation in refine_rotations:
            for scale in refine_scales:
                key = (round(float(rotation), 6), round(float(scale), 6))
                if key in seen:
                    continue
                _check_deadline(deadline)
                seen.add(key)
                refined_hypotheses.append(
                    _score_sync_hypothesis(
                        spectrum,
                        magnitude,
                        image.size,
                        config,
                        float(rotation),
                        float(scale),
                        search_radius=1,
                    )
                )

    # ---- 多重否决：任一条不过就放弃，宁缺毋滥 ----
    best = max(refined_hypotheses, key=_hypothesis_sort_key)
    # 否决 1：四个峰里至少要有 3 个达标。允许缺 1 个是为了容忍局部裁剪
    # 或压缩把某个分量削掉；少于 3 个就无法可靠区分真信号与巧合。
    if best.supported_peaks < 3:
        return None
    # 否决 2：歧义检查。在**足够远**的其他假设里找最高分作为次优；
    # 邻近假设本就该得分相近，不能算竞争对手，所以要按距离过滤。
    alternatives = [
        item
        for item in refined_hypotheses
        if item.supported_peaks >= 3
        and (
            abs(item.spectral_rotation - best.spectral_rotation) >= 3.0
            or abs(item.scale - best.scale) >= 0.10
        )
    ]
    second_score = max((item.score for item in alternatives), default=0.0)
    # 领先幅度：次优为 0 时得 1.0，次优逼近最优时趋于 0
    score_margin = max(0.0, 1.0 - second_score / max(best.score, 1e-12))
    if second_score >= best.score * (1.0 - AMBIGUOUS_SCORE_MARGIN):
        return None
    # 否决 3：置信度下限。
    # 取第三强的峰（下标 2）而非最强峰来算——因为只要求 3 个峰达标，
    # 第三强的那个就是"最弱的合格者"，用它衡量才是这次检测的真实底线。
    third_ratio = sorted(best.ratios, reverse=True)[2]
    support_confidence = max(0.0, min(1.0, 1.0 - PEAK_SUPPORT_RATIO / third_ratio))
    # 最终置信度 = 峰强可信度 × 领先幅度，两者任一薄弱都会拉低结果
    confidence = float(max(1e-12, min(1.0, support_confidence * score_margin)))
    if confidence < MIN_SYNC_CONFIDENCE:
        return None

    # ---- 第三级：平移估计（仅在图像近乎正立时进行）----
    # 旋转超过 0.5° 就跳过：网格是斜的，"水平/垂直偏移"无从谈起。
    offset_x: int | None = None
    offset_y: int | None = None
    if abs(best.spectral_rotation) <= 0.5:
        offset_image: Image.Image | None = image
        # 缩放偏离 1.0 超过 2% 时，先按估计比例还原到原始尺寸再估偏移，
        # 否则 128 网格周期本身就是错的。
        if abs(best.scale - 1.0) > 0.02:
            normalized_size = (
                max(1, round(image.width / best.scale)),
                max(1, round(image.height / best.scale)),
            )
            normalized_pixels = normalized_size[0] * normalized_size[1]
            # 还原后过大就放弃平移估计（置 None），但**保留**已经得到的
            # 旋转与缩放结果——有部分结论也好过完全没有。
            if (
                max(normalized_size) <= MAX_OFFSET_ANALYSIS_SIDE
                and normalized_pixels <= MAX_OFFSET_ANALYSIS_PIXELS
            ):
                offset_image = image.resize(
                    normalized_size,
                    Image.Resampling.BICUBIC,
                )
            else:
                offset_image = None
        offset = (
            _estimate_tile_offset(offset_image, config, deadline)
            if offset_image is not None
            else None
        )
        if offset is not None:
            offset_x, offset_y = offset
    return SyncEstimate(
        # 取负：内部算的是"频谱转了多少"，对外要给的是"需要把图转回多少"。
        rotation_degrees=float(-best.spectral_rotation),
        scale=float(best.scale),
        confidence=confidence,
        supported_peaks=best.supported_peaks,
        evaluated_hypotheses=len(coarse_hypotheses) + len(refined_hypotheses),
        elapsed_seconds=float(monotonic() - started),
        offset_x=offset_x,
        offset_y=offset_y,
    )


def _score_sync_hypothesis(
    spectrum: np.ndarray,
    magnitude: np.ndarray,
    image_size: tuple[int, int],
    config: V4Config,
    spectral_rotation: float,
    scale: float,
    *,
    search_radius: int,
) -> _SyncHypothesis:
    """给一个 (旋转, 缩放) 假设打分。

    :param search_radius: 峰搜索半径。粗搜用 8（预测粗糙、需宽域找峰），
        精搜用 1（预测已准、收窄以排除邻近噪声）。

    评分公式::

        score = Σ log1p(ratio_i) - 0.01 × 定位误差

    * ``log1p`` 压缩动态范围，防止某一个极强的峰独自决定成败——
      四个峰"都还不错"应当胜过"一个极强 + 三个很差"；
    * 减去定位误差项，是要求峰不仅强，还得**出现在预测的位置上**。
      随机噪声峰即使很强，位置也对不上，会被这一项扣分扣掉。

    每个分量取正负两侧比值的**较小者**：共轭对称要求两侧同时存在，
    以弱侧为准可以过滤掉只有单侧强的伪峰。
    """
    image_width, image_height = image_size
    analysis_height, analysis_width = magnitude.shape
    center_y = analysis_height // 2
    center_x = analysis_width // 2
    radians = np.deg2rad(spectral_rotation)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    ratios: list[float] = []
    localization_error = 0.0
    for component_index, (frequency_x, frequency_y) in enumerate(
        config.pilot_frequency_vectors
    ):
        rotated_x = (cosine * frequency_x - sine * frequency_y) / scale
        rotated_y = (sine * frequency_x + cosine * frequency_y) / scale
        analysis_x = rotated_x * image_width / analysis_width
        analysis_y = rotated_y * image_height / analysis_height
        if abs(analysis_x) >= 0.5 or abs(analysis_y) >= 0.5:
            ratios.append(0.0)
            continue
        positive = (
            round(center_y + analysis_y * analysis_height),
            round(center_x + analysis_x * analysis_width),
        )
        negative = (
            round(center_y - analysis_y * analysis_height),
            round(center_x - analysis_x * analysis_width),
        )
        positive_ratio, positive_bin = _peak_ratio(
            magnitude, *positive, search_radius=search_radius
        )
        negative_ratio, negative_bin = _peak_ratio(
            magnitude, *negative, search_radius=search_radius
        )
        # 累计实测峰位与预测峰位的平方距离，作为"假设有多贴合"的惩罚项
        localization_error += (
            (positive_bin[0] - positive[0]) ** 2
            + (positive_bin[1] - positive[1]) ** 2
            + (negative_bin[0] - negative[0]) ** 2
            + (negative_bin[1] - negative[1]) ** 2
        )
        ratio = min(positive_ratio, negative_ratio)
        ratios.append(float(ratio))
    ratio_tuple = tuple(ratios)
    return _SyncHypothesis(
        spectral_rotation=float(spectral_rotation),
        scale=float(scale),
        supported_peaks=sum(value >= PEAK_SUPPORT_RATIO for value in ratio_tuple),
        score=float(
            sum(np.log1p(value) for value in ratio_tuple)
            - 0.01 * localization_error
        ),
        ratios=ratio_tuple,
        localization_error=float(localization_error),
    )


def _hypothesis_sort_key(item: _SyncHypothesis) -> tuple[float, int]:
    """假设排序键：先比综合得分，得分相同再比达标峰数。"""
    return item.score, item.supported_peaks


def _estimate_tile_offset(
    image: Image.Image,
    config: V4Config,
    deadline: float | None,
) -> tuple[int, int] | None:
    """估计 128 网格的起始偏移，即图片被平移/裁剪了多少。

    :return: ``(offset_x, offset_y)``，取值 0~127；无法可靠估计时 ``None``。

    **原理。** 幅度谱对平移不变，但**相位**会随平移线性变化：
    信号平移 ``(dx, dy)``，频率 ``(fx, fy)`` 处的相位就偏移
    ``2π(fx·dx + fy·dy)``。导频的理论初相是已知的，
    于是拿实测相位与理论相位相减，就能反解出平移量。

    **三步实现**：

    1. 用半块步长滑窗切出许多 128×128 块，对每块做 FFT，
       在四个导频频点上取复数值；
    2. 用**中位数**（而非均值）合并所有块的观测，得到稳健的相位估计——
       图像内容会给个别块带来强烈干扰，中位数对这类离群值免疫；
    3. 对 128×128 种候选偏移逐一算相位残差，取残差最小者。

    最后有一道质量闸：残差大于 0.002 就返回 ``None``。
    """
    tile_size = config.tile_size
    if image.width < tile_size or image.height < tile_size:
        return None
    _check_deadline(deadline)
    grayscale = cv2.cvtColor(np.asarray(image)[..., :3], cv2.COLOR_RGB2GRAY)
    # 半块重叠滑窗：步长取半块而非整块，样本数翻倍，中位数更稳健。
    stride = tile_size // 2
    x_starts = tuple(range(0, image.width - tile_size + 1, stride))
    y_starts = tuple(range(0, image.height - tile_size + 1, stride))
    # 样本太少时中位数没有意义，直接放弃
    if len(x_starts) * len(y_starts) < 4:
        return None

    # 同样要加窗，抑制块边界不连续造成的频谱泄漏
    window = np.hanning(tile_size)
    window_2d = window[:, None] * window[None, :]
    component_values: list[list[complex]] = [
        [] for _ in range(PILOT_COMPONENT_COUNT)
    ]
    # 逐行批处理：一行内的所有块堆成一个批次，一次 FFT 算完
    for start_y in y_starts:
        _check_deadline(deadline)
        blocks = np.stack(
            [
                grayscale[
                    start_y : start_y + tile_size,
                    start_x : start_x + tile_size,
                ]
                for start_x in x_starts
            ]
        ).astype(np.float64)
        blocks -= np.mean(blocks, axis=(1, 2), keepdims=True)
        spectra = np.fft.fft2(blocks * window_2d, axes=(-2, -1))
        for component_index, (frequency_x, frequency_y) in enumerate(
            config.pilot_frequency_vectors
        ):
            # 导频频率乘块尺寸即该分量在块频谱中的格点下标
            frequency_column = round(frequency_x * tile_size)
            frequency_row = round(frequency_y * tile_size)
            values = spectra[:, frequency_row, frequency_column]
            # 相位对齐因子：每个块的起点不同，其观测相位天然带有
            # 2π(fx·x₀ + fy·y₀) 的位置偏移。乘以对应的共轭因子把这部分
            # 消掉，各块的观测才能统一到"全图起点"的同一参考系下合并。
            alignment = np.asarray(
                [
                    np.exp(
                        -1j
                        * TWO_PI
                        * (frequency_x * start_x + frequency_y * start_y)
                    )
                    for start_x in x_starts
                ]
            )
            component_values[component_index].extend(values * alignment)

    # 实部虚部分别取中位数再组成复数。不直接对相位角取中位数，
    # 是因为相位是环形量（-π 与 +π 相邻），在 ±π 附近取中位数会得出
    # 完全错误的结果；在复平面上做则没有这个问题。
    coefficient_phases = []
    for values in component_values:
        array = np.asarray(values)
        robust_coefficient = complex(
            float(np.median(array.real)),
            float(np.median(array.imag)),
        )
        coefficient_phases.append(float(np.angle(robust_coefficient)))

    # 一次性构造 128×128 的候选偏移网格，全部候选并行计算
    offsets_y, offsets_x = np.mgrid[0:tile_size, 0:tile_size]
    residuals = []
    phases = _pilot_phases(config.codec)
    for observed_phase, base_phase, (frequency_x, frequency_y) in zip(
        coefficient_phases,
        phases,
        config.pilot_frequency_vectors,
    ):
        # 残差 = 实测相位 - 理论初相 - 平移引起的相位偏移。
        # 减 π/2 是 sin 与 FFT 复指数基之间的固定相位差
        # （sin θ = cos(θ - π/2)），属于约定换算，不是经验值。
        residuals.append(
            observed_phase
            - (base_phase - np.pi / 2.0)
            - TWO_PI * (frequency_x * offsets_x + frequency_y * offsets_y)
        )
    # 用 1 - cos(残差) 作为误差度量：它对 2π 周期天然免疫，
    # 残差为 0 或 2π 的整数倍时都取 0，不会因相位缠绕产生假的高误差。
    # 四个分量取平均，要求所有分量同时吻合。
    phase_error = np.mean(1.0 - np.cos(np.stack(residuals)), axis=0)
    best_flat_index = int(np.argmin(phase_error))
    best_y, best_x = np.unravel_index(best_flat_index, phase_error.shape)
    # 质量闸：最优残差仍偏大，说明四个分量给不出一致的偏移，判定不可信。
    if float(phase_error[best_y, best_x]) > 0.002:
        return None
    _check_deadline(deadline)
    return int(best_x), int(best_y)


def _phase_aligned_ratio(
    spectrum: np.ndarray,
    predicted_row: int,
    predicted_column: int,
    frequency_x: float,
    frequency_y: float,
    *,
    phases: tuple[float, ...],
    component_index: int,
) -> float:
    """相位相干检测：利用已知理论相位，把峰从噪声里"对齐"出来。

    :return: 相干投影值与背景弥散度之比，语义与 :func:`_peak_ratio` 的
        返回值一致，可直接取较大者合并。

    与只看幅度的 :func:`_peak_ratio` 相比，这里多用了一条信息——
    导频的相位是我们自己定的、已知的。把复数谱旋转到理论相位方向再取实部，
    真信号会**相干叠加**成一个正的大值，而随机噪声的相位是均匀分布的，
    投影后正负相消。于是即便幅度上峰不突出，相位上仍可能清晰可辨。

    仅在分析谱与原图同尺寸时调用——一旦缩放过，理论相位就对不上了。
    """
    height, width = spectrum.shape
    center_y = height // 2
    center_x = width // 2
    # 真实频率通常落在两个格点之间，这里算出与最近格点的小数偏差，
    # 用于补偿由此产生的额外相位（下面两个 π·frac·(N-1)/N 项）。
    fractional_x = (predicted_column - center_x) - frequency_x * width
    fractional_y = (predicted_row - center_y) - frequency_y * height
    expected_phase = (
        phases[component_index]
        - np.pi / 2.0
        - np.pi * fractional_x * (width - 1) / width
        - np.pi * fractional_y * (height - 1) / height
    )

    row_start = max(0, predicted_row - 8)
    row_stop = min(height, predicted_row + 9)
    column_start = max(0, predicted_column - 8)
    column_stop = min(width, predicted_column + 9)
    local = spectrum[row_start:row_stop, column_start:column_stop]
    # 乘 e^(-jφ) 把复数值旋转到理论相位方向，取实部即相干投影。
    projected = np.real(local * np.exp(-1j * expected_phase))
    target_row = predicted_row - row_start
    target_column = predicted_column - column_start
    # 负值意味着相位反了，那必然不是我们的导频，按 0 处理。
    target = max(0.0, float(projected[target_row, target_column]))

    # 背景取 17×17 邻域中**排除中心 ±2 格**的部分：
    # 峰会因频谱泄漏在紧邻几格上留下拖尾，把它们算进背景会高估噪声水平。
    rows = np.arange(row_start, row_stop)[:, None]
    columns = np.arange(column_start, column_stop)[None, :]
    background = projected[
        (np.abs(rows - predicted_row) > 2)
        | (np.abs(columns - predicted_column) > 2)
    ]
    dispersion = float(np.std(np.abs(background))) if background.size else 0.0
    if target <= 0.0:
        return 0.0
    # 除以机器精度下限，避免背景全零（如纯色图）时除零。
    return target / max(dispersion, np.finfo(np.float64).eps)


def _peak_ratio(
    magnitude: np.ndarray,
    predicted_row: int,
    predicted_column: int,
    *,
    search_radius: int = 1,
) -> tuple[float, tuple[int, int]]:
    """在预测位置附近找峰，返回 ``(峰背景比, 实际峰位)``。

    :param search_radius: 搜索半径。粗搜阶段放宽到 8，精搜收紧到 1。
    :return: 比值越大越像真峰；峰位供调用方计算定位误差。

    两段式：先在 ``search_radius`` 邻域内定位实际峰，再以**实际峰位**
    （而非预测位）为中心估计周围背景。以实际峰为中心很关键——
    预测偏了一两格时，若仍按预测位取背景，会把真峰算进背景里，
    把比值稀释得毫无意义。

    背景用**中位数**而非均值：中位数不受峰本身及其拖尾影响，
    能真实反映噪声基线。
    """
    height, width = magnitude.shape
    search_row_start = max(0, predicted_row - search_radius)
    search_row_stop = min(height, predicted_row + search_radius + 1)
    search_column_start = max(0, predicted_column - search_radius)
    search_column_stop = min(width, predicted_column + search_radius + 1)
    search = magnitude[
        search_row_start:search_row_stop,
        search_column_start:search_column_stop,
    ]
    peak_offset = np.unravel_index(int(np.argmax(search)), search.shape)
    peak_bin = (
        search_row_start + int(peak_offset[0]),
        search_column_start + int(peak_offset[1]),
    )
    peak = float(magnitude[peak_bin])

    row_start = max(0, peak_bin[0] - 8)
    row_stop = min(height, peak_bin[0] + 9)
    column_start = max(0, peak_bin[1] - 8)
    column_stop = min(width, peak_bin[1] + 9)
    # 背景取以**实际峰位**为中心的 17×17 邻域，扣掉中心 ±2 格的拖尾区
    local = magnitude[row_start:row_stop, column_start:column_stop]
    rows = np.arange(row_start, row_stop)[:, None]
    columns = np.arange(column_start, column_stop)[None, :]
    background = local[
        (np.abs(rows - peak_bin[0]) > 2)
        | (np.abs(columns - peak_bin[1]) > 2)
    ]
    local_median = float(np.median(background)) if background.size else 0.0
    if peak <= 0.0:
        return 0.0, peak_bin
    return peak / max(local_median, np.finfo(np.float64).eps), peak_bin


def _validate_dimensions(height: int, width: int) -> None:
    """导频尺寸必须是正整数。"""
    if type(height) is not int or type(width) is not int:
        raise TypeError("pilot dimensions must be integers")
    if height <= 0 or width <= 0:
        raise ValueError("pilot dimensions must be positive")


def _validate_config(config: V4Config) -> None:
    """配置必须是 :class:`V4Config` 本身，不接受子类（防止格式契约被覆写）。"""
    if type(config) is not V4Config:
        raise TypeError("config must be an exact V4Config instance")


def _validate_image(image: Image.Image) -> None:
    """图片必须是 RGB 或 RGBA 模式的 PIL 图像。"""
    if type(image) is not Image.Image:
        raise TypeError("image must be an exact PIL Image")
    if image.mode not in ("RGB", "RGBA"):
        raise ValueError("image mode must be RGB or RGBA")


def _validated_rotation(value: float) -> float:
    """校验旋转角并转成 ``float``。

    这里允许 ±180°，比 :class:`SyncEstimate` 的 ±12° 宽得多——
    本函数服务于 :func:`pilot_peak_evidence` 这个诊断接口，
    调用方可以主动探查任意角度；而 ±12° 是自动搜索的实际工作范围。
    """
    if type(value) not in (int, float) or not np.isfinite(value):
        raise TypeError("rotation must be a finite number")
    rotation = float(value)
    if not -180.0 <= rotation <= 180.0:
        raise ValueError("rotation must be between -180 and 180 degrees")
    return rotation


def _validated_scale(value: float) -> float:
    """校验缩放比并转成 ``float``（同样比自动搜索范围更宽，供诊断用）。"""
    if type(value) not in (int, float) or not np.isfinite(value):
        raise TypeError("scale must be a finite number")
    scale = float(value)
    if not 0.25 <= scale <= 4.0:
        raise ValueError("scale must be between 0.25 and 4")
    return scale


def _validate_deadline(deadline: float | None) -> None:
    """截止时刻必须是有限数或 ``None``（表示不限时）。"""
    if deadline is None:
        return
    if type(deadline) not in (int, float) or not np.isfinite(deadline):
        raise TypeError("deadline must be a finite monotonic timestamp")


def _check_deadline(deadline: float | None) -> None:
    """超时即抛 :class:`TimeoutError`，在各耗时步骤之间密集调用。

    用 ``monotonic()`` 而非 ``time()``：单调时钟不受系统时间调整
    （NTP 校时、手动改钟、夏令时）影响，是计时的唯一正确选择。
    """
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("FFT synchronization deadline expired")


__all__ = ("SyncEstimate", "detect_pilot", "embed_pilot", "pilot_signal")
