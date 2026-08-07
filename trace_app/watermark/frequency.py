"""频域水印三件套：DCT、DWT、FFT 三层叠加。

传统链路（非 V4）中的抗压缩层。与 V4 不同，这里嵌入的**不是可解码的数据**，
而是由溯源号派生的伪随机图案——检测时靠**相关性**判断"这张图是否含有
某个溯源号对应的图案"，属于需要先知道候选、再逐一验证的**非盲检测**。

**三层各占一个颜色通道，互不干扰**：

* :func:`apply_dct_layer` —— G 通道，分块 DCT 中频系数，抗 JPEG 压缩
* :func:`apply_dwt_layer` —— R 通道，Haar 小波水平细节子带，抗缩放
* :func:`apply_fft_layer` —— B 通道，频谱环形图案，抗旋转

分通道的好处是三层可以顺序叠加而不互相削弱。

**评分机制。** 每层都有配套的 ``*_layer_score``，返回**信噪比**性质的分数：
把图案与实际系数做相关，除以背景标准差。分数越高说明该层信号越明确。
分数一律用 ``max(0.0, ...)`` 截断——负相关意味着"完全不匹配"，
与"轻微不匹配"在结论上没有区别，统一归零便于上层设阈值。
"""

import hashlib
from typing import Callable

import cv2
import numpy as np
import pywt
from PIL import Image

from trace_app.config import DCT_BLOCK, DCT_DELTA, DWT_DELTA, FFT_DELTA, ROBUST_MAGIC


def robust_pattern(bit_index: int, size: int) -> np.ndarray:
    """为指定比特位生成 ±1 的伪随机块状图案。

    :param bit_index: 比特序号，不同序号得到彼此独立的图案。
    :param size: 输出方阵边长。

    先生成 4×4 的粗粒度图案，再用 Kron 积放大成块状。
    **块状而非逐像素随机**是关键：逐像素的高频噪声会被 JPEG 压缩
    和缩放直接抹平，而大色块属于低频成分，能在这些操作下存活。

    种子里的 7919 是个质数，用它做步长可以让相邻比特的种子充分分散，
    避免生成出相似的图案。
    """
    rng = np.random.default_rng(ROBUST_MAGIC + bit_index * 7919)
    coarse = rng.choice(np.array([-1, 1], dtype=np.int16), size=(4, 4))
    repeat = max(1, int(np.ceil(size / 4)))
    pattern = np.kron(coarse, np.ones((repeat, repeat), dtype=np.int16))
    # 向上取整放大后可能超出目标尺寸，裁掉多余部分
    return pattern[:size, :size]


def layer_seed(trace_id: str, layer: str) -> int:
    """由溯源号与层名派生随机种子。

    带上层名做**域分隔**：同一个溯源号在 DCT/DWT/FFT 三层得到完全不同的
    图案。否则三层图案相同，会在图上叠加成一个明显的规律性纹理，
    既影响画质，也容易被针对性擦除。

    ``& 0x7FFFFFFF`` 截成 31 位非负整数，适配 NumPy 对种子范围的要求。
    """
    digest = hashlib.blake2b(f"{trace_id}:{layer}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def pseudo_random_signs(trace_id: str, layer: str, count: int) -> np.ndarray:
    """生成该层专属的 ±1 符号序列，决定每个位置往哪个方向调制。

    序列由溯源号确定性派生：嵌入端与检测端各自算一遍即可得到同一序列，
    无需随图传递。这也是非盲检测的基础——必须先知道溯源号才能重建序列。
    """
    rng = np.random.default_rng(layer_seed(trace_id, layer))
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=count)


def apply_dct_layer(image: Image.Image, trace_id: str) -> Image.Image:
    """DCT 层：在 G 通道分块调制中频系数。

    对每个 ``DCT_BLOCK`` 方块做 DCT，按符号序列往 ``[3,4]`` 与 ``[4,3]``
    两个中频系数上加减固定量。

    选这两个位置的理由与 V4 相同：中频既避开了低频（改动肉眼可见），
    也避开了高频（JPEG 量化时最先被丢弃）；且两者关于对角线对称，
    在量化表中权重相近，压缩后能同步保留。

    选 G 通道是因为人眼对绿色最敏感，多数编码器会给它分配最多的比特，
    信号损失最小——代价是可见性风险最高，故调制量 ``DCT_DELTA`` 取得很小。
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    channel = arr[:, :, 1]
    height, width = channel.shape
    blocks_y = height // DCT_BLOCK
    blocks_x = width // DCT_BLOCK
    signs = pseudo_random_signs(trace_id, "dct", blocks_y * blocks_x)
    idx = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            y = by * DCT_BLOCK
            x = bx * DCT_BLOCK
            block = channel[y : y + DCT_BLOCK, x : x + DCT_BLOCK]
            coeff = cv2.dct(block)
            coeff[3, 4] += signs[idx] * DCT_DELTA
            coeff[4, 3] += signs[idx] * DCT_DELTA
            channel[y : y + DCT_BLOCK, x : x + DCT_BLOCK] = cv2.idct(coeff)
            idx += 1
    arr[:, :, 1] = channel
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def dct_layer_score(image: Image.Image, trace_id: str) -> float:
    """DCT 层的相关性评分。

    :return: ``均值 / 标准差`` 形式的信噪比，越高越可信；不含水印时接近 0。

    把每块的 ``coeff[3,4] + coeff[4,3]`` 乘上对应符号。若图中确实嵌入了
    该溯源号的图案，正负会被符号统一"翻正"，均值显著大于 0；
    若溯源号不对，符号序列与实际调制无关，正负相消、均值趋近 0。

    除以标准差是为了归一化：图像内容本身会给系数带来很大波动，
    只有相对于这个波动足够突出的均值才算真信号。
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    channel = arr[:, :, 1]
    height, width = channel.shape
    blocks_y = height // DCT_BLOCK
    blocks_x = width // DCT_BLOCK
    count = blocks_y * blocks_x
    # 样本少于 16 块时统计量没有意义，直接判 0 而不是给出一个不可信的分数
    if count < 16:
        return 0.0
    signs = pseudo_random_signs(trace_id, "dct", count)
    values = []
    idx = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            y = by * DCT_BLOCK
            x = bx * DCT_BLOCK
            coeff = cv2.dct(channel[y : y + DCT_BLOCK, x : x + DCT_BLOCK])
            values.append((coeff[3, 4] + coeff[4, 3]) * signs[idx])
            idx += 1
    values = np.array(values, dtype=np.float32)
    # +1e-6 防止零方差（纯色图）导致除零
    return float(max(0.0, values.mean() / (values.std() + 1e-6)))


def apply_dwt_layer(image: Image.Image, trace_id: str) -> Image.Image:
    """DWT 层：在 R 通道的 Haar 小波水平细节子带上叠加图案。

    一级 Haar 分解得到四个子带：``ll`` 低频概貌、``lh`` 水平细节、
    ``hl`` 垂直细节、``hh`` 对角细节。

    只改 ``lh``：``ll`` 承载图像主体内容，动它会明显改变画面；
    ``hh`` 是最高频成分，压缩时最先损失。``lh`` 处于中间地带，
    既不显眼又有一定韧性。

    选 Haar 小波是因为它的基函数是矩形阶跃，计算最快，
    且与"块状图案"的形态天然契合。
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    channel = arr[:, :, 0]
    coeffs = pywt.dwt2(channel, "haar")
    ll, (lh, hl, hh) = coeffs
    signs = pseudo_random_signs(trace_id, "dwt", lh.size).reshape(lh.shape)
    lh = lh + signs * DWT_DELTA
    rebuilt = pywt.idwt2((ll, (lh, hl, hh)), "haar")
    # 奇数尺寸的图像经小波往返后会多出一行/一列（Haar 需要偶数长度，
    # 内部做了padding）。两侧同时切片，取交集尺寸，避免形状不匹配。
    arr[: rebuilt.shape[0], : rebuilt.shape[1], 0] = rebuilt[: arr.shape[0], : arr.shape[1]]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def dwt_layer_score(image: Image.Image, trace_id: str) -> float:
    """DWT 层的相关性评分，原理与 :func:`dct_layer_score` 相同。"""
    channel = np.array(image.convert("RGB"), dtype=np.float32)[:, :, 0]
    _, (lh, _, _) = pywt.dwt2(channel, "haar")
    signs = pseudo_random_signs(trace_id, "dwt", lh.size).reshape(lh.shape)
    values = (lh * signs).ravel()
    return float(max(0.0, values.mean() / (values.std() + 1e-6)))


def fft_pattern(shape: tuple[int, int], trace_id: str) -> np.ndarray:
    """生成频域的环形点阵图案：96 对关于中心对称的亮点。

    :return: 与图像同尺寸的浮点掩码，亮点处接近 1，其余为 0。

    **为什么点要成对且中心对称。** 实数图像的频谱必然共轭对称，
    只在单侧加点会破坏这个性质，逆变换回去就得到复数（虚部非零），
    取实部会丢失信息、产生伪影。成对添加则天然满足对称性。

    **为什么排布成环。** 点到中心的距离即频率。把点限制在
    ``[radius_min, radius_max]`` 的圆环内，就是把能量约束在中频段——
    半径过小（低频）会造成可见的大面积色斑，过大（高频）则被压缩抹掉。
    半径按图像短边的 1/10 ~ 1/4 自适应，使不同尺寸的图落在相当的相对频段。

    **为什么最后要高斯模糊。** 单像素的频域尖峰对应空域中一个铺满全图的
    严格正弦波，一旦图片被轻微缩放，峰就偏离原格点、检测不到了。
    把峰"抹开"成小斑块相当于放宽了容差，代价是峰值强度略降。
    """
    height, width = shape
    rng = np.random.default_rng(layer_seed(trace_id, "fft"))
    pattern = np.zeros((height, width), dtype=np.float32)
    center_y, center_x = height // 2, width // 2
    radius_min = max(12, min(height, width) // 10)
    # 保证上界至少比下界大 4，避免小图上出现空区间
    radius_max = max(radius_min + 4, min(height, width) // 4)
    # 角度只在 [0, π) 取样：另外半圈由中心对称点自动覆盖
    for _ in range(96):
        angle = rng.uniform(0, np.pi)
        radius = rng.integers(radius_min, radius_max)
        y = int(round(center_y + np.sin(angle) * radius))
        x = int(round(center_x + np.cos(angle) * radius))
        y2 = int(round(center_y - np.sin(angle) * radius))
        x2 = int(round(center_x - np.cos(angle) * radius))
        if 0 <= y < height and 0 <= x < width:
            pattern[y, x] = 1.0
        if 0 <= y2 < height and 0 <= x2 < width:
            pattern[y2, x2] = 1.0
    return cv2.GaussianBlur(pattern, (0, 0), 1.2)


def apply_fft_layer(image: Image.Image, trace_id: str) -> Image.Image:
    """FFT 层：在 B 通道频谱的指定位置**按比例抬高幅度**。

    关键在于"乘"而不是"加"：``magnitude × (1 + pattern × FFT_DELTA)``。
    乘法使调制量与该位置原有的能量成正比——本就明亮的频点多加一些、
    暗淡的少加一些，从而自动贴合图像内容，避免在平坦区域凭空造出
    可见的纹理。这是一种简易的视觉掩蔽策略。

    **相位完全保留**，只动幅度。相位承载着图像的结构信息（边缘位置），
    改动它会立刻产生肉眼可见的失真。

    选 B 通道是因为人眼对蓝色最不敏感，可容纳相对更强的调制。
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    channel = arr[:, :, 2]
    spectrum = np.fft.fftshift(np.fft.fft2(channel))
    pattern = fft_pattern(channel.shape, trace_id)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    magnitude = magnitude * (1.0 + pattern * FFT_DELTA)
    # 用改动后的幅度与原相位重建复数谱，逆变换后取实部
    # （理论上虚部应为零，取实部是消除浮点误差残留）
    rebuilt = np.real(np.fft.ifft2(np.fft.ifftshift(magnitude * np.exp(1j * phase))))
    arr[:, :, 2] = rebuilt
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def fft_layer_score(image: Image.Image, trace_id: str) -> float:
    """FFT 层评分：比较图案位置与背景的频谱幅度差异。

    与另外两层不同，这里不用符号相关（FFT 层只增不减、没有负向调制），
    改为直接比较**图案覆盖处**与**其余位置**的平均幅度，
    差值再除以背景标准差归一化。

    先取 ``log1p``：频谱幅度的动态范围极大（低频比高频高好几个数量级），
    不取对数的话均值会被少数几个低频点完全支配。

    阈值 0.05 用于把高斯模糊后的图案二值化成掩码；
    有效点少于 10 个则样本不足，判 0。
    """
    channel = np.array(image.convert("RGB"), dtype=np.float32)[:, :, 2]
    magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(channel))))
    pattern = fft_pattern(channel.shape, trace_id)
    mask = pattern > 0.05
    if int(mask.sum()) < 10:
        return 0.0
    selected = magnitude[mask]
    background = magnitude[~mask]
    return float(max(0.0, (selected.mean() - background.mean()) / (background.std() + 1e-6)))


def apply_frequency_layers(
    image: Image.Image,
    trace_id: str,
    *,
    apply_dct_layer_fn: Callable[[Image.Image, str], Image.Image] | None = None,
    apply_dwt_layer_fn: Callable[[Image.Image, str], Image.Image] | None = None,
    apply_fft_layer_fn: Callable[[Image.Image, str], Image.Image] | None = None,
) -> Image.Image:
    """依次叠加 DCT → DWT → FFT 三层，返回处理后的图片。

    因为三层各占一个颜色通道，顺序上互不影响；这里的嵌套写法只是
    把三次调用串起来，读作"先 dct，再 dwt，最后 fft"。
    """
    dct_fn = apply_dct_layer_fn or apply_dct_layer
    dwt_fn = apply_dwt_layer_fn or apply_dwt_layer
    fft_fn = apply_fft_layer_fn or apply_fft_layer
    return fft_fn(dwt_fn(dct_fn(image, trace_id), trace_id), trace_id)


def layer_scores_for_image(
    image: Image.Image,
    trace_id: str,
    *,
    dct_layer_score_fn: Callable[[Image.Image, str], float] | None = None,
    dwt_layer_score_fn: Callable[[Image.Image, str], float] | None = None,
    fft_layer_score_fn: Callable[[Image.Image, str], float] | None = None,
) -> dict[str, float]:
    """一次性算出三层的评分，供检测结论附带展示。

    保留 4 位小数：足够区分强弱，又不至于把浮点噪声也写进记录里。
    """
    dct_fn = dct_layer_score_fn or dct_layer_score
    dwt_fn = dwt_layer_score_fn or dwt_layer_score
    fft_fn = fft_layer_score_fn or fft_layer_score
    return {
        "dct": round(dct_fn(image, trace_id), 4),
        "dwt": round(dwt_fn(image, trace_id), 4),
        "fft": round(fft_fn(image, trace_id), 4),
    }
