"""V4 的 DCT 系数调制：把码字比特写进图像，以及从图像读回软判决分数。

这是 V4 真正"改动像素"的地方。核心思路是**系数对比较编码**：

在每个 16×16 单元格的 DCT 频谱中取一对中频系数，人为拉开它们的差值。
差值为正代表比特 1，为负代表比特 0。解码时只看这对系数谁大谁小即可，
**无需知道原图**——这就是盲提取。

之所以编码"相对关系"而非"绝对数值"，是因为亮度整体变化、对比度调整、
JPEG 量化等操作会同时缩放一对系数，它们的**大小关系**却能保住。

**空间层级**::

    整图 → 128×128 分块(tile) → 8×8 网格 → 16×16 单元格(cell) → 1 比特
                                                    共 64 个 = 一个完整码字

只在 Y（亮度）通道操作：人眼对亮度最敏感，但亮度通道在各类压缩中保留得
也最完整，且不受色度二次采样（4:2:0）的影响。

**批处理**。``embed_codeword`` / ``extract_image_tiles`` 把多个分块拼成
一个批次，用三维矩阵乘一次算完所有 DCT。相比逐块循环，
在大图上是数量级的差距。
"""

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from .config import V4Config
from .payload import (
    permute_codeword_bits,
    phase_for_tile,
    phase_permutation,
)


# 与 V4Config 中同名字段一致的几何常量。此处单独定义是为了让本模块的
# 数组形状校验不依赖配置实例（校验函数需要在拿到 config 之前就能工作）。
CELL_SIZE = 16
GRID_SIZE = 8
TILE_SIZE = 128
# 每个分块承载的比特数 = 8×8 网格
BIT_COUNT = 64
# 亮度中心值。DCT 前先减去它把数据移到零附近，
# 否则直流分量（DC）会大到淹没我们要调制的中频分量。
LUMINANCE_CENTER = 128.0
# 每批处理的分块数。取 8 是内存与速度的折中：
# 一批的中间数组约 8×64×16×16×8 字节 ≈ 1MB，能舒适地留在 CPU 缓存里。
DCT_TILE_BATCH = 8


@dataclass(frozen=True, slots=True)
class TileScores:
    """单个分块的提取结果：64 个比特的软判决分数。

    "软判决"指分数不是 0/1，而是带强度的实数：符号表示比特值，
    绝对值表示置信度（1.0 附近说明信号强度与嵌入时的 ``dct_margin`` 相当）。
    保留强度信息是为了让上层能选出最不可信的字节做 RS 擦除。

    ``logical_scores`` 已经过逆置换还原为**逻辑位序**，可直接按码字比特顺序解读。
    """

    tile_x: int
    tile_y: int
    phase: int
    logical_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        """构造校验：坐标、相位、分数数量与有限性。

        分数必须是原生 ``float`` 且有限——NaN 会在后续的排序与阈值比较中
        静默传播，产生看似正常实则错误的判定结果。
        """
        if type(self.tile_x) is not int or type(self.tile_y) is not int:
            raise TypeError("tile coordinates must be integers")
        if self.tile_x < 0 or self.tile_y < 0:
            raise ValueError("tile coordinates must be nonnegative")
        if type(self.phase) is not int:
            raise TypeError("phase must be an integer")
        if self.phase not in range(4):
            raise ValueError("phase must be between 0 and 3")
        if type(self.logical_scores) is not tuple:
            raise TypeError("scores must be a tuple")
        if len(self.logical_scores) != BIT_COUNT:
            raise ValueError("scores must contain exactly 64 values")
        if any(
            type(score) is not float or not np.isfinite(score)
            for score in self.logical_scores
        ):
            raise ValueError("scores must contain only finite floats")


def _dct_basis(size: int) -> np.ndarray:
    """取 DCT 基矩阵的只读视图。

    返回 ``view()`` 而非缓存对象本身：视图与原数组共享内存（零拷贝），
    同时调用方即使误改视图的形状也不会影响缓存里的那份。
    """
    return _cached_dct_basis(size).view()


@lru_cache(maxsize=1)
def _cached_dct_basis(size: int) -> np.ndarray:
    """构造 16×16 的正交 DCT-II 基矩阵（只算一次，永久缓存）。

    第 k 行是第 k 个余弦基函数在 16 个采样点上的取值::

        basis[k][n] = c(k) · cos(π · (n + 0.5) · k / N)

    ``(n + 0.5)`` 的半采样偏移是 DCT-II 的定义部分，使基函数在边界处
    自然延拓为偶对称，避免块边界产生突变。

    归一化系数分两档——直流行乘 √(1/N)、交流行乘 √(2/N)——目的是让矩阵
    **正交**。正交带来两个好处：逆变换直接用转置即可（无需求逆），
    以及变换前后能量守恒，系数的绝对值可直接当作强度来比较。

    结果标记为不可写，防止被意外修改后污染所有后续变换。
    """
    if type(size) is not int:
        raise TypeError("DCT basis size must be an integer")
    if size != CELL_SIZE:
        raise ValueError("DCT basis size must be exactly 16")

    frequencies = np.arange(size, dtype=np.float64)[:, None]
    samples = np.arange(size, dtype=np.float64)[None, :]
    basis = np.cos(np.pi * (samples + 0.5) * frequencies / size)
    basis[0] *= np.sqrt(1.0 / size)
    basis[1:] *= np.sqrt(2.0 / size)
    basis.flags.writeable = False
    return basis


def _forward_dct_blocks(blocks: np.ndarray) -> np.ndarray:
    """批量正变换：``(N, 16, 16)`` 空域块 → DCT 系数。

    二维 DCT 可分解为"先对行做一维 DCT，再对列做"，用矩阵表达就是
    ``B @ X @ Bᵀ``。NumPy 的 ``@`` 支持批量维度，一次调用即算完整批。
    """
    values = _validated_blocks(blocks)
    basis = _dct_basis(CELL_SIZE)
    # 屏蔽溢出/无效警告：极端像素值可能触发中间告警，
    # 但最终结果由 _validated_output 统一把关，无需逐次打印。
    with np.errstate(over="ignore", invalid="ignore"):
        result = basis @ values @ basis.T
    return _validated_output(result)


def _inverse_dct_blocks(blocks: np.ndarray) -> np.ndarray:
    """批量逆变换：DCT 系数 → 空域块。

    因为基矩阵正交，逆变换就是把正变换的两个乘子转置互换：``Bᵀ @ X @ B``。
    """
    values = _validated_blocks(blocks)
    basis = _dct_basis(CELL_SIZE)
    with np.errstate(over="ignore", invalid="ignore"):
        result = basis.T @ values @ basis
    return _validated_output(result)


def embed_tile_bits(
    luminance_tile: np.ndarray,
    bits: tuple[int, ...],
    config: V4Config,
) -> np.ndarray:
    """把 64 个比特写进单个 128×128 亮度分块，返回调制后的分块。

    单块版本，主要供测试与调试使用；生产路径走批处理的
    :func:`embed_codeword`。

    ``[None, ...]`` 是在最前面插一个长度为 1 的批次维，
    好让单块也能复用批量版的 :func:`_embed_coefficients`。
    """
    tile = _validated_tile(luminance_tile)
    _validate_bits(bits)
    _validate_config(config)

    centered_blocks = _tile_to_blocks(tile - LUMINANCE_CENTER)
    coefficients = _forward_dct_blocks(centered_blocks)[None, ...]
    physical_bits = np.asarray(bits, dtype=np.float64)[None, ...]
    _embed_coefficients(coefficients, physical_bits, config)
    restored = _blocks_to_tile(_inverse_dct_blocks(coefficients[0]))
    # 变换全程在"减去中心值"的坐标系里进行，最后加回来还原到 0~255 量纲。
    return _validated_output(restored + LUMINANCE_CENTER)


def extract_tile_scores(
    luminance_tile: np.ndarray,
    config: V4Config,
) -> tuple[float, ...]:
    """从单个分块读出 64 个软判决分数（**物理位序**，未做逆置换）。

    与 :func:`embed_tile_bits` 对应的单块版本。需要逻辑位序的调用方
    请用 :func:`extract_image_tiles`，它会顺带完成逆置换。
    """
    tile = _validated_tile(luminance_tile)
    _validate_config(config)

    centered_blocks = _tile_to_blocks(tile - LUMINANCE_CENTER)
    coefficients = _forward_dct_blocks(centered_blocks)[None, ...]
    scores = _extract_coefficient_scores(coefficients, config)[0]
    return tuple(scores.tolist())


def embed_codeword(
    image: Image.Image,
    codeword: bytes,
    config: V4Config = V4Config(),
) -> Image.Image:
    """把 8 字节码字嵌入整张图片的所有完整分块，返回新图片。

    :param codeword: RS 编码后的 8 字节码字。
    :param config: 算法参数，默认用标准配置。
    :return: 新的 PIL 图片；原图不被修改。
    :raises ValueError: 图片太小，凑不齐最少分块数或最少相位数。

    **每个分块都嵌入同一个码字**（codec 名字里的 "full_repeat"），
    只是各自按所在位置的相位做了不同置换。这带来极强的抗裁剪能力：
    只要还剩下 ``minimum_tiles`` 个完整分块，就能完整还原码字。

    图片右侧和底部不足 128 像素的边角会被忽略——不完整的分块无法解码，
    强行嵌入只会白白损伤画质。
    """
    _validate_image(image)
    if type(codeword) is not bytes:
        raise TypeError("codeword must be bytes")
    if len(codeword) != 8:
        raise ValueError("codeword must contain exactly 8 bytes")
    _validate_config(config)
    tiles = _eligible_tiles(image, config)

    source = np.asarray(image)
    # 复制一份作为输出画布：source 是只读视图，且后续要按分块逐块覆盖，
    # 未被覆盖的边角区域应保持原样。
    output = source.copy()
    for batch_start in range(0, len(tiles), DCT_TILE_BATCH):
        batch = tiles[batch_start : batch_start + DCT_TILE_BATCH]
        tile_count = len(batch)
        # 把本批分块在**纵向拼成一长条**，这样整批只需一次色彩空间转换。
        # cv2.cvtColor 的单次调用开销较大，合并调用比逐块转换快得多。
        rgb_strip = np.concatenate(
            tuple(
                source[
                    tile_y * TILE_SIZE : (tile_y + 1) * TILE_SIZE,
                    tile_x * TILE_SIZE : (tile_x + 1) * TILE_SIZE,
                    :3,
                ]
                for tile_x, tile_y, _ in batch
            ),
            axis=0,
        )
        # 转到 YCrCb，只有 Y（首通道）参与调制，Cr/Cb 原样保留。
        ycrcb_strip = cv2.cvtColor(rgb_strip, cv2.COLOR_RGB2YCrCb)
        luminance_tiles = ycrcb_strip[..., 0].reshape(
            tile_count,
            TILE_SIZE,
            TILE_SIZE,
        )
        centered_blocks = np.stack(
            tuple(
                _tile_to_blocks(tile.astype(np.float64) - LUMINANCE_CENTER)
                for tile in luminance_tiles
            ),
            axis=0,
        )
        # 摊平成 (批×64, 16, 16) 交给 DCT，算完再还原成四维。
        # _forward_dct_blocks 只接受三维输入，这里靠 reshape 适配。
        coefficients = _forward_dct_blocks(
            centered_blocks.reshape(
                tile_count * BIT_COUNT,
                CELL_SIZE,
                CELL_SIZE,
            )
        ).reshape(tile_count, BIT_COUNT, CELL_SIZE, CELL_SIZE)
        # 同一个码字，按各分块自己的相位置换出不同的物理比特排列。
        physical_bits = np.asarray(
            [permute_codeword_bits(codeword, phase) for _, _, phase in batch],
            dtype=np.float64,
        )
        _embed_coefficients(coefficients, physical_bits, config)
        restored_blocks = _inverse_dct_blocks(
            coefficients.reshape(
                tile_count * BIT_COUNT,
                CELL_SIZE,
                CELL_SIZE,
            )
        ).reshape(tile_count, BIT_COUNT, CELL_SIZE, CELL_SIZE)

        # 把调制后的亮度写回长条。三步不可省：
        #   rint  —— 四舍五入到最近整数（直接截断会引入系统性向下偏差）
        #   clip  —— 夹到 0~255，防止调制把接近边界的像素推出有效范围
        #   uint8 —— 转回图像的原生位深
        for tile_index in range(tile_count):
            top = tile_index * TILE_SIZE
            embedded_y = (
                _blocks_to_tile(restored_blocks[tile_index]) + LUMINANCE_CENTER
            )
            ycrcb_strip[top : top + TILE_SIZE, :, 0] = np.clip(
                np.rint(embedded_y),
                0,
                255,
            ).astype(np.uint8)

        # 转回 RGB，再把长条按原坐标散射回输出画布。
        converted_strip = cv2.cvtColor(ycrcb_strip, cv2.COLOR_YCrCb2RGB)
        for tile_index, (tile_x, tile_y, _) in enumerate(batch):
            source_top = tile_index * TILE_SIZE
            target_top = tile_y * TILE_SIZE
            target_left = tile_x * TILE_SIZE
            output[
                target_top : target_top + TILE_SIZE,
                target_left : target_left + TILE_SIZE,
                # 只写 RGB 三通道；RGBA 图的 alpha 通道保持原值不动。
                :3,
            ] = converted_strip[
                source_top : source_top + TILE_SIZE,
                :,
            ]

    return Image.fromarray(output)


def extract_image_tiles(
    image: Image.Image,
    config: V4Config = V4Config(),
) -> tuple[TileScores, ...]:
    """从整图提取每个分块的软判决分数（已还原为逻辑位序）。

    :return: 每个完整分块一条 :class:`TileScores`。
    :raises ValueError: 图片太小，不满足最少分块/相位要求。

    与嵌入不同，这里**一次性**转换整图并处理全部分块，不分批——
    提取只读不写，没有逐块回写的需求，整体处理反而更快。
    """
    _validate_image(image)
    _validate_config(config)
    tiles = _eligible_tiles(image, config)

    rgb = np.asarray(image)[..., :3]
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    centered_blocks = _gather_centered_blocks(ycrcb[..., 0], tiles)
    tile_count = len(tiles)
    coefficients = _forward_dct_blocks(
        centered_blocks.reshape(tile_count * BIT_COUNT, CELL_SIZE, CELL_SIZE)
    ).reshape(tile_count, BIT_COUNT, CELL_SIZE, CELL_SIZE)
    physical_score_batches = _extract_coefficient_scores(coefficients, config)
    records = []
    for tile_index, (tile_x, tile_y, phase) in enumerate(tiles):
        # 逆置换：用置换表当索引数组一次性重排。
        # 正向表满足 permutation[逻辑] = 物理，因此
        # scores[permutation] 取出的第 i 项就是逻辑位 i 的分数——
        # 这正是逆置换，无需另行构造逆表。
        logical_scores = physical_score_batches[tile_index][
            np.asarray(phase_permutation(phase), dtype=np.intp)
        ]
        records.append(
            TileScores(
                tile_x=tile_x,
                tile_y=tile_y,
                phase=phase,
                logical_scores=tuple(logical_scores.tolist()),
            )
        )
    return tuple(records)


def _embed_coefficients(
    coefficients: np.ndarray,
    physical_bits: np.ndarray,
    config: V4Config,
) -> None:
    """**就地**调制 DCT 系数，把比特写进系数对的大小关系中。

    :param coefficients: 形状 ``(分块数, 64, 16, 16)``，会被原地修改。
    :param physical_bits: 形状 ``(分块数, 64)``，取值 0/1。

    这是整个 V4 编码的核心，四步完成：

    1. **比特转符号**：``2b - 1`` 把 ``{0, 1}`` 映射为 ``{-1, +1}``，
       后续可以用乘法统一处理两种比特值，不必写分支。
    2. **算当前差值**：``d = 系数A - 系数B``。
    3. **算修正量**：目标是让 ``sign · d ≥ margin``。缺口为
       ``margin - sign·d``；用 ``maximum(..., 0)`` 截断意味着
       **差值已经够大就不动它**——这是关键的画质优化：图像本身的纹理
       常常已经满足条件，此时嵌入是零代价的。除以 2 是因为下一步要
       从两个系数分头各改一半。
    4. **对称施加**：A 加、B 减。一增一减使两系数之和不变，
       即该单元格的**总能量守恒**，视觉上远比只改一个系数更不可察觉。

    每比特用了两对系数（``coefficient_pairs`` 有两组），两对都按同样规则
    调制，提取时取平均——相当于同一比特写了两遍，天然抗单点损伤。
    """
    pairs = np.asarray(config.coefficient_pairs, dtype=np.intp)
    # 拆成四组下标数组，供 NumPy 的花式索引一次性取出所有系数对。
    first_rows = pairs[:, 0, 0]
    first_columns = pairs[:, 0, 1]
    second_rows = pairs[:, 1, 0]
    second_columns = pairs[:, 1, 1]
    # 末尾补一维，使符号能与"系数对"维度广播对齐。
    signs = (2.0 * physical_bits - 1.0)[..., None]
    differences = (
        coefficients[:, :, first_rows, first_columns]
        - coefficients[:, :, second_rows, second_columns]
    )
    corrections = np.maximum(config.dct_margin - signs * differences, 0.0) / 2.0
    coefficients[:, :, first_rows, first_columns] += signs * corrections
    coefficients[:, :, second_rows, second_columns] -= signs * corrections


def _extract_coefficient_scores(
    coefficients: np.ndarray,
    config: V4Config,
) -> np.ndarray:
    """从 DCT 系数读出软判决分数，是 :func:`_embed_coefficients` 的逆操作。

    :return: 形状 ``(分块数, 64)`` 的分数。符号即比特值（正为 1、负为 0），
        绝对值即置信度。

    两对系数的差值取**平均**，把两次冗余嵌入合并成一个判决——
    一对被局部损伤破坏时，另一对仍能把结果拉向正确方向。

    最后除以 ``dct_margin`` 做归一化，使分数与嵌入强度解耦：
    无论当初用多大强度嵌入，理想情况下读回来都是 ±1.0 附近。
    上层因此可以用固定阈值判断质量，不必关心嵌入参数。
    """
    pairs = np.asarray(config.coefficient_pairs, dtype=np.intp)
    differences = (
        coefficients[:, :, pairs[:, 0, 0], pairs[:, 0, 1]]
        - coefficients[:, :, pairs[:, 1, 0], pairs[:, 1, 1]]
    )
    return np.mean(differences, axis=2) / config.dct_margin


def _gather_centered_blocks(
    luminance: np.ndarray,
    tiles: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    """按分块坐标切出亮度块、减去中心值并切成单元格，堆成一个批次数组。"""
    return np.stack(
        tuple(
            _tile_to_blocks(
                luminance[
                    tile_y * TILE_SIZE : (tile_y + 1) * TILE_SIZE,
                    tile_x * TILE_SIZE : (tile_x + 1) * TILE_SIZE,
                ].astype(np.float64)
                - LUMINANCE_CENTER
            )
            for tile_x, tile_y, _ in tiles
        ),
        axis=0,
    )


def _validated_blocks(blocks: np.ndarray) -> np.ndarray:
    """校验并转成 ``float64`` 的 ``(N, 16, 16)`` 块数组。

    ``astype(copy=False)`` 在已经是 float64 时零拷贝返回原数组。
    转换后再查一次有限性：大整数转浮点可能溢出成 ``inf``，
    这种情况在转换前的整数检查里是发现不了的。
    """
    if not isinstance(blocks, np.ndarray):
        raise TypeError("DCT blocks must be a NumPy array")
    if blocks.ndim != 3 or blocks.shape[1:] != (CELL_SIZE, CELL_SIZE):
        raise ValueError("DCT blocks must have shape (N, 16, 16)")
    if blocks.dtype.kind not in "iuf":
        raise TypeError("DCT blocks must contain real numeric values")
    if not np.isfinite(blocks).all():
        raise ValueError("DCT blocks must contain only finite values")

    values = blocks.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("DCT blocks must be representable as finite float64 values")
    return values


def _validated_tile(luminance_tile: np.ndarray) -> np.ndarray:
    """校验并转成 ``float64`` 的 ``(128, 128)`` 亮度分块。"""
    if type(luminance_tile) is not np.ndarray:
        raise TypeError("luminance tile must be a NumPy array")
    if luminance_tile.shape != (TILE_SIZE, TILE_SIZE):
        raise ValueError("luminance tile must have shape (128, 128)")
    if luminance_tile.dtype.kind not in "iuf":
        raise TypeError("luminance tile must contain real numeric values")
    if not np.isfinite(luminance_tile).all():
        raise ValueError("luminance tile must contain only finite values")

    tile = luminance_tile.astype(np.float64, copy=False)
    if not np.isfinite(tile).all():
        raise ValueError("luminance tile must be representable as finite float64 values")
    return tile


def _validate_bits(bits: tuple[int, ...]) -> None:
    """比特必须是恰好 64 个纯 0/1 整数。

    ``bit not in (0, 1)`` 结合 ``type(bit) is not int``，
    连 ``True``/``False`` 也一并排除（``bool`` 是 ``int`` 子类，
    单靠值检查会放行）。
    """
    if type(bits) is not tuple:
        raise TypeError("bits must be a tuple")
    if len(bits) != BIT_COUNT:
        raise ValueError("bits must contain exactly 64 values")
    if any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only integer zero or one values")


def _validate_config(config: V4Config) -> None:
    """配置必须是 :class:`V4Config` 本身，**不接受子类**。

    子类可能覆写那些标了 ``init=False`` 的格式契约字段，
    从而在不改版本号的情况下悄悄改变编码格式——直接从类型上堵死。
    """
    if type(config) is not V4Config:
        raise TypeError("config must be an exact V4Config instance")


def _validate_image(image: Image.Image) -> None:
    """图片必须是 RGB 或 RGBA 模式的 PIL 图像。

    调色板图（P）、灰度图（L）等模式的像素排布不同，
    需由调用方先转换，本模块不做隐式转换以免掩盖上游问题。
    """
    if type(image) is not Image.Image:
        raise TypeError("image must be an exact PIL Image")
    if image.mode not in ("RGB", "RGBA"):
        raise ValueError("image mode must be RGB or RGBA")


def _eligible_tiles(
    image: Image.Image,
    config: V4Config,
) -> tuple[tuple[int, int, int], ...]:
    """枚举图中所有完整分块，返回 ``(tile_x, tile_y, phase)`` 三元组。

    :raises ValueError: 分块数或相位数不足。

    整除意味着**丢弃右侧与底部的残余边角**：不足 128 像素的区域无法
    构成完整分块，嵌入了也解不出来。

    两个下限缺一不可：分块数太少则冗余不足、抗裁剪能力形同虚设；
    相位数太少则说明分块排布退化（如只有一行），
    置换带来的抗突发错误能力会大打折扣。
    """
    columns = image.width // TILE_SIZE
    rows = image.height // TILE_SIZE
    tiles = tuple(
        (tile_x, tile_y, phase_for_tile(tile_x, tile_y))
        for tile_y in range(rows)
        for tile_x in range(columns)
    )
    if len(tiles) < config.minimum_tiles:
        raise ValueError("image does not contain the minimum number of complete tiles")
    if len({phase for _, _, phase in tiles}) < config.minimum_phases:
        raise ValueError("image does not contain enough distinct tile phases")
    return tiles


def _tile_to_blocks(tile: np.ndarray) -> np.ndarray:
    """把 ``(128, 128)`` 分块切成 ``(64, 16, 16)`` 的单元格序列。

    三步都是纯视图操作，不复制数据::

        (128, 128)
          reshape  → (8, 16, 8, 16)   拆成 行块/行内/列块/列内
          transpose→ (8, 8, 16, 16)   把两个"块"维调到一起
          reshape  → (64, 16, 16)     合并成一维块序号

    转置这一步是关键：不换轴的话，直接 reshape 会把同一行内相邻单元格的
    像素交错在一起，得到的根本不是完整的 16×16 块。
    输出按**行优先**排列，即块序号 = 行块 × 8 + 列块。
    """
    return (
        tile.reshape(GRID_SIZE, CELL_SIZE, GRID_SIZE, CELL_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(BIT_COUNT, CELL_SIZE, CELL_SIZE)
    )


def _blocks_to_tile(blocks: np.ndarray) -> np.ndarray:
    """:func:`_tile_to_blocks` 的逆操作：单元格序列拼回完整分块。

    转置轴序 ``(0, 2, 1, 3)`` 与正向相同——该置换是自逆的。
    """
    return (
        blocks.reshape(GRID_SIZE, GRID_SIZE, CELL_SIZE, CELL_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(TILE_SIZE, TILE_SIZE)
    )


def _validated_output(result: np.ndarray) -> np.ndarray:
    """出口兜底：变换结果不得含 NaN/inf。

    输入合法时数学上不会溢出，但这里仍然检查——一旦上游有 bug 产生了
    异常值，宁可在此明确报错，也不要让 NaN 顺着写进图片、
    最终表现为"水印时灵时不灵"这种极难定位的故障。
    """
    if not np.isfinite(result).all():
        raise ValueError("DCT transform must produce only finite output values")
    return result


__all__ = (
    "TileScores",
    "embed_codeword",
    "embed_tile_bits",
    "extract_image_tiles",
    "extract_tile_scores",
)
