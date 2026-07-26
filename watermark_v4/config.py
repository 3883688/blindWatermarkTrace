"""V4 水印算法的参数配置与校验。

本模块只定义 :class:`V4Config` 一个冻结数据类，是整套 V4 算法的**唯一参数来源**：
分块几何、DCT 系数对、FFT 导频频率、检测门限、各阶段时间预算全部集中在此。

设计上有三条原则：

1. **不可变**。``frozen=True`` 保证一次检测过程中参数不会被中途改写，
   嵌入端与提取端拿到的一定是同一组数值。
2. **大部分字段不可外部指定**。带 ``init=False`` 的字段属于**格式契约**——
   改了它们，旧图就再也解不出来，因此不开放构造参数，只能改代码并同步升版本号。
   可调的只有强度、尺寸、门限、时间预算这几项运行期参数。
3. **构造即校验**。:meth:`V4Config.__post_init__` 里做了穷尽的类型、形状、
   区间与交叉一致性检查，把配置错误挡在算法入口，
   而不是等到运行到一半才出现莫名其妙的数值结果。
"""

from dataclasses import dataclass, field
from math import isfinite


# 单元格内的 (行, 列) 坐标，用于定位 DCT 系数
Coordinate = tuple[int, int]
# 一对 DCT 系数：比较这两个位置的大小关系来编码 1 个比特
CoefficientPair = tuple[Coordinate, Coordinate]
# FFT 导频的归一化频率向量 (fx, fy)
FrequencyVector = tuple[float, float]


@dataclass(frozen=True, slots=True)
class V4Config:
    """V4 水印的全部算法参数。

    ``init=False`` 的字段是**格式契约**，改动会导致既有水印图无法解码；
    其余字段可在构造时按场景调整。
    """

    # ---- 版本标识（写入记录，用于提取端选择正确的解码路径）----
    version: int = field(default=4, init=False)
    # 编解码方案的完整描述：HMAC-32 位认证 + RS(8,4) 纠错 + 全图重复 + 同步导频
    codec: str = field(
        default="hmac32_rs_8_4_full_repeat_sync_v4",
        init=False,
    )

    # ---- 分块几何：128 = 8×16，三者必须严格满足此等式（见 __post_init__）----
    # 每个分块 128×128 像素，是嵌入与检测的最小完整单元
    tile_size: int = field(default=128, init=False)
    # 每个分块切成 8×8 = 64 个单元格
    grid_size: int = field(default=8, init=False)
    # 每个单元格 16×16 像素，正好是一次 DCT 变换的尺寸
    cell_size: int = field(default=16, init=False)

    # ---- DCT 系数调制强度 ----
    # 一对系数之间要拉开的差值。越大越抗压缩，但画面越容易出现块状伪影。
    dct_margin: float = 6.0
    dct_margin_range: tuple[float, float] = field(
        default=(2.0, 10.0), init=False
    )
    # 检测端的多档试解强度：按弱→中→强依次尝试，
    # 覆盖"嵌入时用了不同强度"以及"图片被压缩后信号衰减"两种情况。
    dct_margin_calibration: tuple[int, int, int] = field(
        default=(4, 6, 8), init=False
    )
    # 两对中频系数。选 (2,3)/(3,2) 与 (2,4)/(4,2) 这类**关于对角线对称**的位置，
    # 是因为它们在 JPEG 量化表中权重相近，压缩后仍能保持相对大小关系；
    # 而中频段既避开了低频（改动即可见），也避开了高频（压缩时先被丢弃）。
    coefficient_pairs: tuple[CoefficientPair, CoefficientPair] = field(
        default=(((2, 3), (3, 2)), ((2, 4), (4, 2))), init=False
    )

    # ---- FFT 同步导频 ----
    # 导频幅度：决定几何校正的可靠性与画质的平衡点
    pilot_amplitude: float = 0.75
    pilot_amplitude_range: tuple[float, float] = field(
        default=(0.25, 1.25), init=False
    )
    pilot_amplitude_calibration: tuple[float, float, float] = field(
        default=(0.5, 0.75, 1.0), init=False
    )
    # 四个导频频率向量，在频域中形成一个已知的"星座图"。
    # 提取端在频谱上找到这四个峰后，通过它们相对标准位置的缩放与旋转量，
    # 就能反推出图片被缩放/旋转了多少，从而把画面校正回原始几何。
    # 用四个而非一个，是为了在部分峰被裁剪或压缩破坏时仍能定位。
    # 取值都是 1/1024 的整数倍，保证在常见分辨率下落在整数频点上、不产生泄漏。
    pilot_frequency_vectors: tuple[
        FrequencyVector, FrequencyVector, FrequencyVector, FrequencyVector
    ] = field(
        default=(
            (0.0703125, 0.1093750),
            (0.1015625, 0.1562500),
            (0.1406250, 0.0859375),
            (0.1718750, 0.1250000),
        ),
        init=False,
    )

    # ---- 检测门限 ----
    # 分析前先把长边缩到这个尺寸，控制大图的计算量（不影响水印本身）
    analysis_max_side: int = 1024
    # 判定命中所需的最低比特覆盖率：解出的比特必须有 70% 以上一致
    minimum_coverage: float = 0.70
    # 至少要有 2 个分块给出一致结论，单块命中视为噪声
    minimum_tiles: int = 2
    # 至少要有 2 个相位对齐成功，防止随机对齐产生的偶然一致
    minimum_phases: int = 2
    # 每次检测最多深入验证的候选记录数，直接决定最坏耗时
    candidate_limit: int = 3

    # ---- 时间预算 ----
    # 在线检测的 P95 目标耗时（秒）
    online_p95_seconds: float = 10.0
    # 硬超时：超过即无条件中止，防止单次请求拖垮服务
    hard_timeout_seconds: float = 300.0
    # 检测流水线六个阶段各自的时间上限，总和不得超过 online_p95_seconds
    # （该约束在 __post_init__ 中强制校验）。前几个阶段短、后几个阶段长，
    # 对应"快速路径优先、逐级放宽"的检测策略。
    stage_budgets_seconds: tuple[float, float, float, float, float, float] = (
        0.3,
        0.6,
        1.0,
        1.6,
        5.0,
        1.0,
    )

    def __post_init__(self) -> None:
        """构造后的全量校验：类型 → 形状 → 区间 → 交叉一致性。

        分层进行，顺序不可打乱：后面的检查默认前面已经通过。例如区间检查
        直接做数值比较，前提是类型检查已确认它们确实是数字。

        类型判断一律用 ``type(x) is int`` 而非 ``isinstance``，是**刻意**为之：
        ``bool`` 是 ``int`` 的子类，``isinstance(True, int)`` 为真，
        会让 ``minimum_tiles=True`` 这种明显错误的配置蒙混过关。
        """
        # ---- 第一层：标量类型 ----
        integer_fields = (
            "tile_size",
            "grid_size",
            "cell_size",
            "analysis_max_side",
            "minimum_tiles",
            "minimum_phases",
            "candidate_limit",
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")

        # 实数字段额外要求有限：NaN / inf 会让后续所有比较静默失效
        # （NaN 参与的任何比较都是 False，区间校验会形同虚设）。
        real_fields = (
            "dct_margin",
            "pilot_amplitude",
            "minimum_coverage",
            "online_p95_seconds",
            "hard_timeout_seconds",
        )
        for name in real_fields:
            if not _is_finite_number(getattr(self, name)):
                raise TypeError(f"{name} must be a finite number")

        # ---- 第二层：元组的类型与长度 ----
        # 长度是硬约束：算法里对这些元组按固定下标取值，长度不对会越界或错位。
        tuple_shapes = (
            ("dct_margin_range", self.dct_margin_range, 2),
            ("dct_margin_calibration", self.dct_margin_calibration, 3),
            ("coefficient_pairs", self.coefficient_pairs, 2),
            ("pilot_amplitude_range", self.pilot_amplitude_range, 2),
            ("pilot_amplitude_calibration", self.pilot_amplitude_calibration, 3),
            ("pilot_frequency_vectors", self.pilot_frequency_vectors, 4),
            ("stage_budgets_seconds", self.stage_budgets_seconds, 6),
        )
        for name, value, length in tuple_shapes:
            _require_tuple(name, value, length)

        # 嵌套结构再深挖一层：系数对是"对的对"（两层嵌套），导频是"数值对"（一层）。
        _require_nested_pairs("coefficient_pairs", self.coefficient_pairs)
        _require_pair_items("pilot_frequency_vectors", self.pilot_frequency_vectors)

        # ---- 第三层：元组内每个元素的类型 ----
        # 系数坐标先摊平成一维再统一检查，省去三重嵌套循环。
        integer_tuples = (
            ("dct_margin_calibration", self.dct_margin_calibration),
            (
                "coefficient_pairs",
                tuple(
                    component
                    for pair in self.coefficient_pairs
                    for coordinate in pair
                    for component in coordinate
                ),
            ),
        )
        for name, values in integer_tuples:
            if any(type(value) is not int for value in values):
                raise TypeError(f"{name} contains an invalid number")

        real_tuples = (
            ("dct_margin_range", self.dct_margin_range),
            ("pilot_amplitude_range", self.pilot_amplitude_range),
            ("pilot_amplitude_calibration", self.pilot_amplitude_calibration),
            (
                "pilot_frequency_vectors",
                tuple(value for vector in self.pilot_frequency_vectors for value in vector),
            ),
            ("stage_budgets_seconds", self.stage_budgets_seconds),
        )
        for name, values in real_tuples:
            if any(not _is_finite_number(value) for value in values):
                raise TypeError(f"{name} contains an invalid number")

        # ---- 第四层：区间与交叉一致性 ----
        # 分块几何必须自洽：分块 = 网格数 × 单元格边长。
        # 不满足则切块时会剩下边角，嵌入与提取的分块对不齐，水印必然失效。
        if self.tile_size != self.grid_size * self.cell_size:
            raise ValueError("tile_size must equal grid_size * cell_size")

        # DCT 强度：区间本身有序，默认值在区间内，三档标定值也都在区间内。
        dct_low, dct_high = self.dct_margin_range
        if dct_low > dct_high:
            raise ValueError("dct_margin_range must be ordered")
        if not dct_low <= self.dct_margin <= dct_high:
            raise ValueError("dct_margin must be within dct_margin_range")
        if any(not dct_low <= value <= dct_high for value in self.dct_margin_calibration):
            raise ValueError("dct_margin_calibration must be within dct_margin_range")

        # 导频幅度：与 DCT 强度同构的三项检查。
        pilot_low, pilot_high = self.pilot_amplitude_range
        if pilot_low > pilot_high:
            raise ValueError("pilot_amplitude_range must be ordered")
        if not pilot_low <= self.pilot_amplitude <= pilot_high:
            raise ValueError("pilot_amplitude must be within pilot_amplitude_range")
        if any(
            not pilot_low <= value <= pilot_high
            for value in self.pilot_amplitude_calibration
        ):
            raise ValueError(
                "pilot_amplitude_calibration must be within pilot_amplitude_range"
            )

        # 每个 DCT 系数坐标都必须落在单元格内，否则取系数时会越界。
        for pair in self.coefficient_pairs:
            for row, column in pair:
                if not (0 <= row < self.cell_size and 0 <= column < self.cell_size):
                    raise ValueError("coefficient coordinate out of bounds")

        # 分析尺寸至少要容得下一个完整分块，否则缩图后一块都切不出来。
        if self.analysis_max_side < self.tile_size:
            raise ValueError("analysis_max_side must be at least tile_size")
        # 覆盖率下限不得低于 0.70：再低就接近随机猜测的水平，误报率不可接受。
        if not 0.70 <= self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be between 0.70 and one")
        if self.minimum_tiles < 2:
            raise ValueError("minimum_tiles must be at least 2")
        if not 2 <= self.minimum_phases <= 4:
            raise ValueError("minimum_phases must be between 2 and 4")
        if not 1 <= self.candidate_limit <= 3:
            raise ValueError("candidate_limit must be between 1 and 3")

        # ---- 时间预算：三者构成 阶段和 ≤ P95 ≤ 硬超时 的嵌套关系 ----
        if self.online_p95_seconds <= 0:
            raise ValueError("online_p95_seconds must be positive")
        if self.online_p95_seconds > 10.0:
            raise ValueError("online_p95_seconds must be at most 10.0")
        if self.hard_timeout_seconds <= 0:
            raise ValueError("hard_timeout_seconds must be positive")
        if self.hard_timeout_seconds > 300.0:
            raise ValueError("hard_timeout_seconds must be at most 300.0")
        if any(value <= 0 for value in self.stage_budgets_seconds):
            raise ValueError("stage_budgets_seconds values must be positive")
        if sum(self.stage_budgets_seconds) > self.online_p95_seconds:
            raise ValueError("stage budgets must not exceed online_p95_seconds")
        if self.hard_timeout_seconds < self.online_p95_seconds:
            raise ValueError("hard_timeout_seconds must be at least online_p95_seconds")


def _is_finite_number(value: object) -> bool:
    """判断是否为有限实数（``bool`` 不算数字）。

    用 ``type(...) is`` 精确匹配而非 ``isinstance``：``bool`` 是 ``int`` 的子类，
    放行它会让 ``dct_margin=True`` 这类配置被当成 ``1.0`` 静默接受。
    """
    if type(value) is int:
        return True
    if type(value) is float:
        return isfinite(value)
    return False


def _require_tuple(name: str, value: object, length: int) -> None:
    """校验必须是指定长度的元组。

    类型错用 ``TypeError``、长度错用 ``ValueError``，两类问题分开报，
    便于调用方（和测试）区分"传错了东西"和"传对了但形状不对"。
    """
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if len(value) != length:
        raise ValueError(f"{name} has invalid shape")


def _require_nested_pairs(name: str, values: tuple[object, ...]) -> None:
    """校验"对的对"结构，即 ``((a, b), (c, d))`` 形式。

    用于 ``coefficient_pairs``：外层是一对坐标，内层每个坐标又是 (行, 列)。
    """
    for pair in values:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError(f"{name} has invalid shape")
        for coordinate in pair:
            if type(coordinate) is not tuple or len(coordinate) != 2:
                raise ValueError(f"{name} has invalid shape")


def _require_pair_items(name: str, values: tuple[object, ...]) -> None:
    """校验每一项都是二元组，用于导频频率向量 ``(fx, fy)``。

    只查一层，是 :func:`_require_nested_pairs` 的浅版本。
    """
    for pair in values:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError(f"{name} has invalid shape")
