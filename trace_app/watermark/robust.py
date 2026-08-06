"""鲁棒水印：传统链路（非 V4）的抗攻击主力层，含 v1/v2/v3 三代实现。

**共同的空域调制骨架。** 三代都把图片切成 ``ROBUST_TILE`` 方块，方块内再分
``ROBUST_GRID × ROBUST_GRID`` 个 ``ROBUST_CELL`` 小格，每格用
:func:`~trace_app.watermark.frequency.robust_pattern` 生成的 ±1 块状图案
调制 **B 通道**（``ROBUST_CHANNEL``）：比特 1 加、比特 0 减。
解码时把小格与同一图案做相关，符号即比特值。

选 B 通道同样是因为人眼对蓝色最不敏感；用块状图案而非单像素噪声，
是为了在压缩和缩放后仍能存活。

**三代的差别只在"嵌什么内容"，不在"怎么嵌"**：

* **v1** —— 直接嵌 ``魔数 + 溯源号哈希`` 的 64 位码。无认证，
  只能靠汉明距离在候选中找最近的，容易误报。
* **v2** —— 嵌 RS 纠错码字，并按分块位置分三个相位。有纠错能力，
  解码时还能用擦除机制挽救。
* **v3** —— 嵌 HMAC 认证码，带比特置换。有密码学认证，
  攻击者无法伪造，误报率最低。

**两类检测入口**：

* :func:`extract_robust_code` —— **盲检测**，不知道几何变换，
  靠多组网格参数暴力试探。快但只能处理未变形的图。
* :func:`detect_aligned_authenticated_watermark` —— **对齐检测**，
  先用视觉特征把图配准回原始几何，再解码。慢但能处理裁剪、缩放、透视。

**依赖注入。** :class:`RobustDependencies` 里的每个字段都可选，
为 ``None`` 时回落到本模块的默认实现（代码里大量的 ``x or default`` 即为此）。
这样测试可以精确替换任意一环。
"""

import hashlib
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from trace_app.watermark.auth import permuted_code_bits, phase_permutation
from trace_app.watermark.ecc import codeword_phase, decode_expected_codeword, encode_codeword, tile_phase
from trace_app.config import (
    CODE_PAYLOAD_BITS,
    CODE_PHYSICAL_BITS,
    ROBUST_BITS,
    ROBUST_CELL,
    ROBUST_CHANNEL,
    ROBUST_DELTA,
    ROBUST_GRID,
    ROBUST_MAGIC,
    ROBUST_TILE,
)
from trace_app.watermark.frequency import robust_pattern


Record = dict[str, Any]
# 记录集可以是列表，也可以是"返回列表的函数"——后者用于延迟求值，
# 避免在用不到记录的分支上白白查一次库
RecordsSource = Iterable[Record] | Callable[[], Iterable[Record]]


def _resolve_records(records: RecordsSource) -> Iterable[Record]:
    """统一处理两种记录来源：可调用对象就调用，否则原样返回。"""
    return records() if callable(records) else records


@dataclass(frozen=True, slots=True)
class RobustConfig:
    """鲁棒水印的几何与强度参数，默认值取自全局配置。

    做成数据类而非直接引用常量，是为了让测试能构造小尺寸配置
    （小方块、少比特）来快速验证逻辑，无需生成大图。
    """

    # 一个方块承载的比特数（= robust_grid²）
    robust_bits: int = ROBUST_BITS
    # 每个小格的像素边长
    robust_cell: int = ROBUST_CELL
    # 调制哪个颜色通道（B 通道，人眼最不敏感）
    robust_channel: int = ROBUST_CHANNEL
    # 调制幅度
    robust_delta: float = ROBUST_DELTA
    # 方块内的网格边长
    robust_grid: int = ROBUST_GRID
    # 载荷高位的固定魔数，用于快速识别"这是本系统的水印"
    robust_magic: int = ROBUST_MAGIC
    # 方块像素边长（= robust_grid × robust_cell）
    robust_tile: int = ROBUST_TILE
    # 编码层的有效载荷位数
    code_payload_bits: int = CODE_PAYLOAD_BITS
    # 编码层重复展开后的物理位数（冗余 = physical / payload 倍）
    code_physical_bits: int = CODE_PHYSICAL_BITS


@dataclass(frozen=True, slots=True)
class RobustDependencies:
    """可替换的实现回调，全部可选；为 ``None`` 时用本模块的默认实现。

    存在的意义有二：让单元测试能把任意一环换成假实现；
    以及打破与 ``trace_app.compat`` 之间的循环导入。
    """

    robust_code_from_trace: Callable[..., int] | None = None
    robust_bits_from_code: Callable[..., list[int]] | None = None
    robust_payload_bytes: Callable[..., bytes] | None = None
    code_crc16: Callable[[int], int] | None = None
    watermark_payload_from_trace: Callable[..., int] | None = None
    hamming_distance: Callable[[int, int], int] | None = None
    iter_robust_tiles: Callable[..., Iterable[tuple[int, int]]] | None = None
    robust_pattern: Callable[[int, int], np.ndarray] | None = None
    encode_codeword: Callable[[bytes], bytes] | None = None
    codeword_phase: Callable[[bytes, int], bytes] | None = None
    tile_phase: Callable[[int, int], int] | None = None
    permuted_code_bits: Callable[[bytes, int], list[int]] | None = None
    phase_permutation: Callable[[int], Any] | None = None
    extract_robust_from_grid: Callable[..., tuple[int | None, float, int]] | None = None
    scores_to_byte: Callable[[np.ndarray], tuple[int, float]] | None = None
    phase_scores_to_codeword: Callable[..., tuple[bytes, list[float]]] | None = None
    decode_expected_codeword: Callable[..., Any] | None = None
    record_v3_auth_code: Callable[[Record], bytes | None] | None = None


DEFAULT_CONFIG = RobustConfig()
DEFAULT_DEPENDENCIES = RobustDependencies()


def robust_code_from_trace(
    trace_id: str, *, config: RobustConfig = DEFAULT_CONFIG
) -> int:
    """v1 码：``16 位魔数 << 48 | 48 位溯源号哈希``，共 64 位。

    魔数占高 16 位，作用是让检测端能先看这一段就快速排除无关比特串，
    不必去跟每条候选记录逐一比对。

    .. warning::
        这里用的是**无密钥哈希**，任何人知道算法就能算出任意溯源号的码，
        伪造不受限制。v3 改用 HMAC 正是为了堵上这个缺口。
    """
    digest = hashlib.blake2b(trace_id.encode("utf-8"), digest_size=6).digest()
    body = int.from_bytes(digest, "big")
    return (config.robust_magic << 48) | body


def robust_bits_from_code(
    code: int, *, config: RobustConfig = DEFAULT_CONFIG
) -> list[int]:
    """把整数码摊成比特列表，高位在前。"""
    return [(code >> shift) & 1 for shift in range(config.robust_bits - 1, -1, -1)]


def robust_payload_bytes(
    trace_id: str,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> bytes:
    """把 v1 的 64 位码转成 8 字节，作为 v2 RS 编码的输入净荷。

    固定 8 字节大端。长度是格式契约：RS(24,8) 的数据段就按 8 字节切分，
    改动会让所有 v2 水印无法解码。
    """
    code = (
        dependencies.robust_code_from_trace(trace_id)
        if dependencies.robust_code_from_trace
        else robust_code_from_trace(trace_id, config=config)
    )
    return code.to_bytes(8, "big")


def code_crc16(value: int) -> int:
    """16 位校验和（用 BLAKE2b 截断代替传统 CRC 多项式）。

    名字里的 "crc16" 只表明用途和长度；实现上取哈希摘要前 2 字节，
    比 CRC 多项式更均匀，也省得单独维护一张查表。
    """
    return int.from_bytes(hashlib.blake2b(value.to_bytes(4, "big"), digest_size=2).digest(), "big")


def watermark_payload_from_trace(
    trace_id: str,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> int:
    """编码层 / 点阵层共用的 48 位载荷。

    结构：``16 位魔数 | 16 位溯源号哈希 | 16 位校验和``。

    比 v1 的 64 位码短，因为编码层和点阵层的每方块容量更小。
    代价是哈希只有 16 位、碰撞概率明显更高，所以额外加了校验和，
    并且检测时必须再跟候选记录逐一比对确认。
    """
    digest = hashlib.blake2b(trace_id.encode("utf-8"), digest_size=2).digest()
    body = int.from_bytes(digest, "big")
    crc16 = dependencies.code_crc16 or code_crc16
    # 校验和覆盖"魔数 + 哈希"两段，任一段出错都能被察觉
    checksum = crc16((config.robust_magic << 16) | body)
    return (config.robust_magic << 32) | (body << 16) | checksum


def watermark_bits_from_trace(
    trace_id: str,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> list[int]:
    """把 48 位载荷重复展开成 ``code_physical_bits`` 个物理比特。

    :return: 物理比特列表，长度为 ``config.code_physical_bits``。

    这是编码层与点阵层用的**重复码**——最朴素的纠错方式：同一逻辑位写多份，
    解码时多数表决（见 :func:`recover_payload_from_code`）。
    比 RS 弱得多，但实现简单、不需要额外的校验字节，
    适合这两层"每方块容量极小"的场景。
    """
    payload = (
        dependencies.watermark_payload_from_trace(trace_id)
        if dependencies.watermark_payload_from_trace
        else watermark_payload_from_trace(
            trace_id, config=config, dependencies=dependencies
        )
    )
    base = [(payload >> shift) & 1 for shift in range(config.code_payload_bits - 1, -1, -1)]
    # 循环重复填满物理位。用取模而非整块复制，使同一逻辑位的多份副本
    # 在物理上**均匀散开**，局部损伤不会一次毁掉某一位的全部副本。
    repeated = []
    for index in range(config.code_physical_bits):
        repeated.append(base[index % config.code_payload_bits])
    return repeated


def recover_payload_from_code(
    code: int, *, config: RobustConfig = DEFAULT_CONFIG
) -> tuple[int, int]:
    """多数表决还原载荷，:func:`watermark_bits_from_trace` 的逆操作。

    :return: ``(还原的载荷, 纠正的比特数)``。

    每个逻辑位在物理上有若干份副本，按 ``code_payload_bits`` 步长分布。
    收集同一位的全部副本做多数表决，少数派的数量即"纠正数"——
    它是信号质量的度量：为 0 说明所有副本一致，越大说明干扰越重。

    平票时 (``one_votes >= zero_votes``) 判 1，是一个任意但确定的约定，
    保证结果可复现。
    """
    bits = [(code >> shift) & 1 for shift in range(config.code_physical_bits - 1, -1, -1)]
    recovered = 0
    corrections = 0
    for index in range(config.code_payload_bits):
        # 按 payload_bits 步长跳着取，正好取到同一逻辑位的全部副本
        votes = [bits[pos] for pos in range(index, config.code_physical_bits, config.code_payload_bits)]
        one_votes = sum(votes)
        zero_votes = len(votes) - one_votes
        bit = 1 if one_votes >= zero_votes else 0
        corrections += min(one_votes, zero_votes)
        recovered = (recovered << 1) | bit
    return recovered, corrections


def normalize_robust_watermark_version(
    value: str | int | None,
    *,
    version_v1: int,
    version_v2: int,
    version_v3: int,
    version_v4: int,
) -> int:
    """把任意形式的版本值归一到四个已知版本之一。

    无法解析或不认识的值一律回落到 ``version_v1``——**最保守的选择**：
    v1 的检测逻辑最宽松，用它去解新版本的水印顶多是解不出，
    而反过来用新版本逻辑去解旧图则可能给出错误结论。
    """
    try:
        version = int(value if value is not None else version_v1)
    except (TypeError, ValueError):
        version = version_v1
    if version == version_v4:
        return version_v4
    if version == version_v3:
        return version_v3
    if version == version_v2:
        return version_v2
    return version_v1


def robust_code_to_trace(
    code: int,
    *,
    records: RecordsSource,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> str | None:
    """把 v1 码**精确**反查成溯源号；无匹配返回 ``None``。

    先查魔数快速排除，再遍历记录逐一比对。要求完全相等——
    容错版本见 :func:`robust_code_to_trace_fuzzy`。
    """
    if (code >> 48) != config.robust_magic:
        return None
    current_records = _resolve_records(records)
    code_from_trace = dependencies.robust_code_from_trace
    for record in current_records:
        trace_id = record.get("trace_id")
        expected = (
            code_from_trace(trace_id)
            if code_from_trace
            else robust_code_from_trace(trace_id, config=config)
        ) if trace_id else None
        if trace_id and expected == code:
            return trace_id
    return None


def hamming_distance(left: int, right: int) -> int:
    """两个整数的汉明距离：异或后数 1 的个数。"""
    return (left ^ right).bit_count()


def robust_code_to_trace_fuzzy(
    code: int,
    max_errors: int = 18,
    *,
    records: RecordsSource,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> tuple[str | None, int]:
    """容错反查 v1 码：找汉明距离最近且不超过 ``max_errors`` 的溯源号。

    :return: ``(溯源号或 None, 最小距离)``。第二个值即使失败也会返回，
        便于调用方了解"差了多少"来调参或排查。

    先看魔数：距离超过 6 比特就直接放弃——这串比特多半根本不是本系统的
    水印，没必要再遍历全部记录。这道前置检查在记录多时能省下大量比对。

    64 位码允许 18 位误差看似宽松，但要凑巧落到某个特定记录的 18 位邻域内，
    概率仍然极低。
    """
    distance_fn = dependencies.hamming_distance or hamming_distance
    magic_distance = distance_fn(code >> 48, config.robust_magic)
    if magic_distance > 6:
        return None, config.robust_bits + 1
    current_records = _resolve_records(records)
    best_trace = None
    best_distance = config.robust_bits + 1
    for record in current_records:
        trace_id = record.get("trace_id")
        if not trace_id:
            continue
        expected = (
            dependencies.robust_code_from_trace(trace_id)
            if dependencies.robust_code_from_trace
            else robust_code_from_trace(trace_id, config=config)
        )
        distance = distance_fn(code, expected)
        if distance < best_distance:
            best_trace = trace_id
            best_distance = distance
    if best_trace and best_distance <= max_errors:
        return best_trace, best_distance
    return None, best_distance


def robust_candidate_records(records: Iterable[Record]) -> list[Record]:
    """筛出有溯源号且启用了鲁棒水印的记录（不区分版本）。"""
    return [record for record in records if record.get("trace_id") and record.get("robust_watermark")]


def legacy_robust_candidate_records(
    records: RecordsSource,
    *,
    normalize_version: Callable[[str | int | None], int],
    version_v1: int,
) -> list[Record]:
    """在上面的基础上再筛出 **v1** 版本的记录。

    盲检测（:func:`extract_robust_code`）只认 v1 的编码方式，
    因此必须把 v2/v3 的记录排除掉，否则会拿错误的期望码去比对。
    """
    current_records = _resolve_records(records)
    return [
        record
        for record in robust_candidate_records(current_records)
        if normalize_version(record.get("robust_watermark_version", version_v1)) == version_v1
    ]


def iter_robust_tiles(
    width: int, height: int, *, config: RobustConfig = DEFAULT_CONFIG
):
    """按方块尺寸不重叠地枚举所有完整方块的左上角坐标。

    与点阵层不同，这里**不做重叠铺排**：鲁棒水印靠的是"整幅图铺满、
    任意区域都有信号"，不需要靠重叠来对抗裁剪。右侧和底部不足一个方块
    的边角直接忽略。
    """
    for y in range(0, height - config.robust_tile + 1, config.robust_tile):
        for x in range(0, width - config.robust_tile + 1, config.robust_tile):
            yield x, y


def embed_robust_watermark(
    image: Image.Image,
    trace_id: str,
    strength_scale: float = 1.0,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> Image.Image:
    """**v1** 嵌入：把 64 位码铺进图中每个方块。

    :param strength_scale: 强度倍率，与 ``robust_delta`` 相乘。

    所有方块嵌的都是同一个码、同样的排布——完全重复，无相位变化。
    这使得它极易被"多图求平均"之类的统计攻击定位。v2 引入相位正是为此。
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    code = (
        dependencies.robust_code_from_trace(trace_id)
        if dependencies.robust_code_from_trace
        else robust_code_from_trace(trace_id, config=config)
    )
    bits = (
        dependencies.robust_bits_from_code(code)
        if dependencies.robust_bits_from_code
        else robust_bits_from_code(code, config=config)
    )
    tiles = (
        dependencies.iter_robust_tiles(width, height)
        if dependencies.iter_robust_tiles
        else iter_robust_tiles(width, height, config=config)
    )
    pattern_fn = dependencies.robust_pattern or robust_pattern
    for x, y in tiles:
        for bit_index, bit in enumerate(bits):
            row = bit_index // config.robust_grid
            col = bit_index % config.robust_grid
            y0 = y + row * config.robust_cell
            x0 = x + col * config.robust_cell
            patch = arr[y0 : y0 + config.robust_cell, x0 : x0 + config.robust_cell, config.robust_channel]
            pattern = pattern_fn(bit_index, config.robust_cell)
            delta = pattern * ((config.robust_delta * strength_scale) if bit else -(config.robust_delta * strength_scale))
            patch[:, :] = np.clip(patch + delta, 0, 255)
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


def embed_robust_watermark_v2(
    image: Image.Image,
    trace_id: str,
    strength_scale: float = 1.0,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> Image.Image:
    """**v2** 嵌入：把 RS(24,8) 码字按三个相位分散铺进各方块。

    :param strength_scale: 强度倍率，与 ``robust_delta`` 相乘。

    载荷仍是 v1 那个 64 位码（8 字节），经 RS 编码扩成 24 字节码字。
    一个方块只装得下 ``robust_bits`` = 64 位 = 8 字节，所以码字按
    ``tile_phase(tile_x, tile_y) = (tile_x + 2·tile_y) % 3`` 切成三段，
    每个方块只嵌属于自己相位的那 8 字节。

    这样做有两重收益：相邻方块的调制内容互不相同，"多图求平均"
    或"块间比对"这类统计攻击拿不到稳定的重复图案；解码端把三段拼回
    完整码字后 RS 还能纠错，容错力远强于 v1 的纯汉明距离。
    代价是三个相位必须都采到，缺一段就整整少 8 字节。
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    payload = (
        dependencies.robust_payload_bytes(trace_id)
        if dependencies.robust_payload_bytes
        else robust_payload_bytes(trace_id, config=config, dependencies=dependencies)
    )
    codeword = (dependencies.encode_codeword or encode_codeword)(payload)
    tiles = dependencies.iter_robust_tiles(image.width, image.height) if dependencies.iter_robust_tiles else iter_robust_tiles(image.width, image.height, config=config)
    tile_phase_fn = dependencies.tile_phase or tile_phase
    phase_fn = dependencies.codeword_phase or codeword_phase
    pattern_fn = dependencies.robust_pattern or robust_pattern
    for x, y in tiles:
        # 相位由**方块索引**决定（不是像素坐标），解码端按同样方式换算即可对上
        phase = tile_phase_fn(x // config.robust_tile, y // config.robust_tile)
        phase_bytes = phase_fn(codeword, phase)
        value = int.from_bytes(phase_bytes, "big")
        bits = dependencies.robust_bits_from_code(value) if dependencies.robust_bits_from_code else robust_bits_from_code(value, config=config)
        for bit_index, bit in enumerate(bits):
            row, col = divmod(bit_index, config.robust_grid)
            y0 = y + row * config.robust_cell
            x0 = x + col * config.robust_cell
            patch = arr[y0 : y0 + config.robust_cell, x0 : x0 + config.robust_cell, config.robust_channel]
            sign = 1.0 if bit else -1.0
            patch[:, :] = np.clip(
                patch + pattern_fn(bit_index, config.robust_cell) * config.robust_delta * strength_scale * sign,
                0,
                255,
            )
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


def embed_robust_watermark_v3(
    image: Image.Image,
    auth_code: bytes,
    strength_scale: float = 1.0,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> Image.Image:
    """**v3** 嵌入：把 8 字节 HMAC 认证码整份铺进每个方块，载体位置按相位置换。

    :param auth_code: 8 字节 HMAC 认证码。密钥不在本模块，由调用方算好后传入——
        这也是签名收"码"而不是收 ``trace_id`` 的原因。
    :param strength_scale: 强度倍率，与 ``robust_delta`` 相乘。

    与 v2 相反，这里每个方块装的都是**完整**的 64 位认证码（full repeat），
    冗余度是 v2 的三倍，理论上剩一个完好方块就有机会解出来。
    抗统计攻击不靠分段，而靠 ``permuted_code_bits``：三个相位各配一套固定置换，
    同一逻辑位在不同相位落到不同小格上，所以方块之间的调制图案照样互不相同。

    最本质的改进是认证码来自 HMAC——攻击者没有密钥就构造不出能通过校验的码，
    而 v1/v2 的码任何人都能自己算出来。
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    tiles = dependencies.iter_robust_tiles(image.width, image.height) if dependencies.iter_robust_tiles else iter_robust_tiles(image.width, image.height, config=config)
    tile_phase_fn = dependencies.tile_phase or tile_phase
    # bits_fn 返回的是"物理位序"：逻辑位已按相位置换搬到各自的小格上
    bits_fn = dependencies.permuted_code_bits or permuted_code_bits
    pattern_fn = dependencies.robust_pattern or robust_pattern
    for x, y in tiles:
        phase = tile_phase_fn(x // config.robust_tile, y // config.robust_tile)
        bits = bits_fn(auth_code, phase)
        for bit_index, bit in enumerate(bits):
            row, col = divmod(bit_index, config.robust_grid)
            y0 = y + row * config.robust_cell
            x0 = x + col * config.robust_cell
            patch = arr[y0 : y0 + config.robust_cell, x0 : x0 + config.robust_cell, config.robust_channel]
            sign = 1.0 if bit else -1.0
            patch[:, :] = np.clip(
                patch + pattern_fn(bit_index, config.robust_cell) * config.robust_delta * strength_scale * sign,
                0,
                255,
            )
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


def extract_robust_from_grid(
    arr: np.ndarray,
    cell: int,
    offset_x: int,
    offset_y: int,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> tuple[int | None, float, int]:
    """在给定的网格参数下盲提取一个 64 位码（多方块硬判决投票）。

    :param arr: RGB 浮点数组。
    :param cell: 假定的小格边长。方块边长按 ``cell × robust_grid`` 现算，
        **不用** ``config.robust_tile``——盲检测正是靠改这个值试探缩放比例。
    :param offset_x: 网格左上角的水平偏移，用于试探平移。
    :param offset_y: 网格左上角的垂直偏移。
    :return: ``(码或 None, 置信度, 已判定的比特数)``。

    每个方块的每个小格与对应图案做相关，符号即这个方块为该位投的票；
    所有方块投完后按多数决出每一位。相关前先减去小格均值
    （``blue - blue.mean()``）：图案的 ±1 之和不保证为零，
    不去掉直流分量的话局部亮度会直接混进相关值，把水印信号压掉。

    方块少于 2 个直接放弃——单块无从交叉印证，纯噪声也能凑出一个"码"。

    置信度取各位投票优势 ``|票差| / 总票数`` 的平均：全票一致为 1，
    势均力敌趋近 0。一票都没有的位记 0 分且不计入 ``decided``，
    调用方靠 ``decided`` 剔除残缺结果。
    """
    height, width = arr.shape[:2]
    tile = cell * config.robust_grid
    # votes[i] = [判 0 的方块数, 判 1 的方块数]
    votes = [[0, 0] for _ in range(config.robust_bits)]
    pattern_fn = dependencies.robust_pattern or robust_pattern
    tiles = 0
    for y in range(offset_y, height - tile + 1, tile):
        for x in range(offset_x, width - tile + 1, tile):
            tiles += 1
            for bit_index in range(config.robust_bits):
                row = bit_index // config.robust_grid
                col = bit_index % config.robust_grid
                y0 = y + row * cell
                x0 = x + col * cell
                patch = arr[y0 : y0 + cell, x0 : x0 + cell, :]
                if patch.size == 0:
                    continue
                blue = patch[:, :, config.robust_channel]
                if blue.shape != (cell, cell):
                    # 越界导致的残缺小格，尺寸对不上就不要硬算相关
                    continue
                pattern = pattern_fn(bit_index, cell).astype(np.float32)
                # 去直流：只保留围绕局部均值的起伏，那才是调制留下的痕迹
                centered = blue - blue.mean()
                score = float((centered * pattern).mean())
                if score > 0:
                    votes[bit_index][1] += 1
                else:
                    votes[bit_index][0] += 1
    if tiles < 2:
        return None, 0.0, 0
    code = 0
    margins = []
    decided = 0
    for zero_votes, one_votes in votes:
        total = zero_votes + one_votes
        if total == 0:
            # 这一位一票都没收到：占位补 0 保持位宽，但不计入 decided
            code = code << 1
            margins.append(0.0)
            continue
        decided += 1
        bit = 1 if one_votes > zero_votes else 0
        code = (code << 1) | bit
        margins.append(abs(one_votes - zero_votes) / total)
    confidence = sum(margins) / len(margins) if margins else 0.0
    return code, confidence, decided


def decode_aligned_robust_trace(
    alignment: Record,
    record: Record,
    max_errors: int = 4,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> Record | None:
    """**v1** 对齐解码：在配准后的图上验证某条记录的 64 位码。

    :param alignment: 视觉配准结果，需含 ``image``（配准后画布）、
        ``valid_mask``（哪些像素是真实内容而非填充）、``target_scale``。
    :param record: 待验证的候选记录，取其 ``trace_id`` 算期望码。
    :param max_errors: 允许的最大比特错误数；整码距离和魔数段距离各自受限。
    :return: 命中时返回带证据字段的字典，否则 ``None``。

    与盲检测最关键的差别是**软合并**：这里把各方块的相关值直接累加，
    最后才取符号，而不是每块先硬判决再投票。硬判决把"勉强为正"和
    "强烈为正"等同看待，丢掉了幅度信息；累加则让信号强的方块自然占更大权重，
    同一张图能多解出几个比特。

    方块按**原图几何**枚举，再乘 ``target_scale`` 映射到配准画布上，
    取出后统一缩回 ``robust_tile`` 尺寸——这样查询图无论被放大还是缩小，
    小格边界都能和图案对齐。

    ``mask_tile.mean() < 0.70`` 是覆盖率闸门：配准画布边缘有大片黑色填充，
    有效像素不足七成的方块读出来基本是噪声，宁可少几块也不能让它们污染累加和。

    开头那一长串校验是防御性的：``alignment`` 来自外部注入的配准回调，
    形状不匹配或 ``target_scale`` 非正都会让后面的坐标换算变成一堆无意义的切片。

    .. note::
        累加的是**未归一化**的原始相关值，高对比度区域的方块权重天然偏高。
        v3 用中位数归一化加截断解决了这点，v1/v2 这两条路径保持原样、未作改动。
    """
    trace_id = record.get("trace_id")
    aligned = alignment.get("image")
    valid_mask = alignment.get("valid_mask")
    target_scale = float(alignment.get("target_scale", 1.0))
    if (
        not trace_id
        or not isinstance(aligned, np.ndarray)
        or not isinstance(valid_mask, np.ndarray)
        or aligned.ndim != 3
        or aligned.shape[:2] != valid_mask.shape
        or target_scale <= 0
    ):
        return None
    height, width = aligned.shape[:2]
    # 反推原图尺寸：方块划分必须在原图坐标系里做，才能和嵌入时一一对应
    original_width = int(round(width / target_scale))
    original_height = int(round(height / target_scale))
    aggregate_scores = np.zeros(config.robust_bits, dtype=np.float64)
    authenticated_tiles = 0
    tiles = dependencies.iter_robust_tiles(original_width, original_height) if dependencies.iter_robust_tiles else iter_robust_tiles(original_width, original_height, config=config)
    pattern_fn = dependencies.robust_pattern or robust_pattern
    for x, y in tiles:
        x0 = max(0, int(round(x * target_scale)))
        y0 = max(0, int(round(y * target_scale)))
        x1 = min(width, int(round((x + config.robust_tile) * target_scale)))
        y1 = min(height, int(round((y + config.robust_tile) * target_scale)))
        if x1 <= x0 or y1 <= y0:
            continue
        mask_tile = valid_mask[y0:y1, x0:x1]
        # 覆盖率闸门：有效像素不足七成的方块多半压着配准留下的空白填充
        if not mask_tile.size or float(mask_tile.mean()) < 0.70:
            continue
        # 缩回标准方块尺寸，抵消查询图相对原图的缩放，让小格与图案对齐
        tile = cv2.resize(aligned[y0:y1, x0:x1, :], (config.robust_tile, config.robust_tile), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        authenticated_tiles += 1
        for bit_index in range(config.robust_bits):
            row = bit_index // config.robust_grid
            col = bit_index % config.robust_grid
            cell = tile[row * config.robust_cell : (row + 1) * config.robust_cell, col * config.robust_cell : (col + 1) * config.robust_cell, config.robust_channel]
            centered = cell - cell.mean()
            aggregate_scores[bit_index] += float(np.mean(centered * pattern_fn(bit_index, config.robust_cell)))
    if authenticated_tiles < 2:
        return None
    # 累加完再取符号——这一步才是"软合并"落地的地方
    decoded_code = 0
    for score in aggregate_scores:
        decoded_code = (decoded_code << 1) | int(score > 0)
    expected_code = dependencies.robust_code_from_trace(trace_id) if dependencies.robust_code_from_trace else robust_code_from_trace(trace_id, config=config)
    distance_fn = dependencies.hamming_distance or hamming_distance
    bit_errors = distance_fn(decoded_code, expected_code)
    # 魔数段再单独卡一次：整码距离达标但魔数错得离谱的，多半是巧合而非真水印
    magic_errors = distance_fn(decoded_code >> 48, config.robust_magic)
    if bit_errors > max_errors or magic_errors > max_errors:
        return None
    average_scores = aggregate_scores / authenticated_tiles
    return {
        "record": record,
        "trace_id": trace_id,
        "decoded_code": decoded_code,
        "bit_errors": bit_errors,
        "magic_errors": magic_errors,
        "authenticated_tiles": authenticated_tiles,
        "mean_abs_score": round(float(np.mean(np.abs(average_scores))), 6),
    }


def _scores_to_byte(scores: np.ndarray) -> tuple[int, float]:
    """把 8 个相关值判成一个字节（高位在前），并给出这一字节的可信度。

    :param scores: 长度为 8 的相关值数组，顺序即比特的高位到低位。
    :return: ``(字节值, 可信度)``。可信度 = ``min|score| / median|score|``。

    可信度是给 RS 解码挑擦除位置用的：值越小，说明这字节里有某一位
    明显比同伴弱、最可能判反，应该优先当作擦除交给 RS 重建。
    ``max(1e-6, ...)`` 只是防止整字节全零时除零。

    .. note::
        分母取的是**本字节自己**的中位数，所以这衡量的是字节内部的均衡度，
        而非绝对信号强度：8 位一致地弱（纯噪声）反而会拿到接近 1 的高分，
        信号很强但有一位偏弱的字节分数反倒偏低。此处如实保留现有行为，未作改动。
    """
    value = 0
    absolute = np.abs(scores.astype(np.float64))
    scale = max(1e-6, float(np.median(absolute)))
    for score in scores:
        value = (value << 1) | int(score > 0)
    return value, float(np.min(absolute / scale))


def _phase_scores_to_codeword(
    phase_scores: np.ndarray,
    phase_counts: list[int],
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> tuple[bytes, list[float]]:
    """把三个相位累加的相关值拼回 24 字节 RS 观测码字。

    :param phase_scores: 形状 ``(3, robust_bits)``，各相位方块相关值的累加和。
    :param phase_counts: 各相位实际参与累加的方块数。
    :return: ``(24 字节观测码字, 24 个逐字节可信度)``。

    先按方块数取平均，让三个相位在数量不均时仍然可比——某个相位多采到几块
    不该让它的幅度天然更大。再每 8 位切一字节。

    相位顺序 0→1→2 与嵌入端 ``codeword_phase`` 的切片顺序严格一致，
    拼出来正好还原成原码字的字节排列，可以直接喂给 RS。

    ``max(1, ...)`` 是兜底；调用方已保证每个相位至少两块。
    """
    observed = bytearray()
    confidences = []
    for phase in range(3):
        average = phase_scores[phase] / max(1, phase_counts[phase])
        for start in range(0, config.robust_bits, 8):
            value, confidence = (dependencies.scores_to_byte or _scores_to_byte)(average[start : start + 8])
            observed.append(value)
            confidences.append(confidence)
    return bytes(observed), confidences


def decode_aligned_robust_trace_v2(
    alignment: Record,
    record: Record,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> Record | None:
    """**v2** 对齐解码：重建 RS 码字，与该记录的期望码字核对。

    :param alignment: 视觉配准结果，需含 ``image`` / ``valid_mask`` / ``target_scale``。
    :param record: 待验证的候选记录。
    :return: 命中时返回带 RS 纠错细节的字典，否则 ``None``。

    骨架和 v1 完全一样（原图坐标枚举方块、覆盖率闸门、缩回标准尺寸、软合并），
    差别只在相关值按方块相位**分三路**累加，最后拼成 24 字节码字。

    ``min(phase_counts) < 2`` 要求三个相位**各自**至少有两块：
    缺任一相位就意味着码字里整整 8 字节是纯噪声，RS(24,8) 的纠错预算
    会被这一段吃光，剩不下多少余量去对付真正的失真。

    校验走 :func:`~trace_app.watermark.ecc.decode_expected_codeword`——它拿期望载荷去
    **验证**而不是自由解码：纠错结果必须同时等于期望载荷和期望码字才算通过，
    可信度列表用于按档位逐次尝试擦除。这么做是因为 v2 的码本身无密码学保护，
    自由解码出什么就信什么的话误报率压不住。
    """
    trace_id = record.get("trace_id")
    aligned = alignment.get("image")
    valid_mask = alignment.get("valid_mask")
    target_scale = float(alignment.get("target_scale", 1.0))
    if (
        not trace_id
        or not isinstance(aligned, np.ndarray)
        or not isinstance(valid_mask, np.ndarray)
        or aligned.ndim != 3
        or aligned.shape[:2] != valid_mask.shape
        or target_scale <= 0
    ):
        return None
    height, width = aligned.shape[:2]
    original_width = int(round(width / target_scale))
    original_height = int(round(height / target_scale))
    phase_scores = np.zeros((3, config.robust_bits), dtype=np.float64)
    phase_counts = [0, 0, 0]
    tiles = dependencies.iter_robust_tiles(original_width, original_height) if dependencies.iter_robust_tiles else iter_robust_tiles(original_width, original_height, config=config)
    tile_phase_fn = dependencies.tile_phase or tile_phase
    pattern_fn = dependencies.robust_pattern or robust_pattern
    for x, y in tiles:
        x0 = max(0, int(round(x * target_scale)))
        y0 = max(0, int(round(y * target_scale)))
        x1 = min(width, int(round((x + config.robust_tile) * target_scale)))
        y1 = min(height, int(round((y + config.robust_tile) * target_scale)))
        if x1 <= x0 or y1 <= y0:
            continue
        mask_tile = valid_mask[y0:y1, x0:x1]
        # 覆盖率闸门，同 v1：空白填充占比过高的方块读出来是噪声
        if not mask_tile.size or float(mask_tile.mean()) < 0.70:
            continue
        tile = cv2.resize(aligned[y0:y1, x0:x1, :], (config.robust_tile, config.robust_tile), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        # 相位由原图坐标系下的方块索引算出，与嵌入端口径一致
        phase = tile_phase_fn(x // config.robust_tile, y // config.robust_tile)
        phase_counts[phase] += 1
        for bit_index in range(config.robust_bits):
            row, col = divmod(bit_index, config.robust_grid)
            cell = tile[row * config.robust_cell : (row + 1) * config.robust_cell, col * config.robust_cell : (col + 1) * config.robust_cell, config.robust_channel]
            centered = cell - cell.mean()
            phase_scores[phase, bit_index] += float(np.mean(centered * pattern_fn(bit_index, config.robust_cell)))
    if min(phase_counts) < 2:
        return None
    observed, confidences = (
        dependencies.phase_scores_to_codeword(phase_scores, phase_counts)
        if dependencies.phase_scores_to_codeword
        else _phase_scores_to_codeword(phase_scores, phase_counts, config=config, dependencies=dependencies)
    )
    payload = dependencies.robust_payload_bytes(trace_id) if dependencies.robust_payload_bytes else robust_payload_bytes(trace_id, config=config, dependencies=dependencies)
    decoded = (dependencies.decode_expected_codeword or decode_expected_codeword)(observed, payload, confidences)
    if not decoded:
        return None
    average_scores = np.vstack([phase_scores[index] / phase_counts[index] for index in range(3)])
    return {
        "record": record,
        "trace_id": trace_id,
        "corrected_symbols": decoded["corrected_symbols"],
        "erasure_count": decoded["erasure_count"],
        "bit_errors": decoded["bit_errors"],
        "recovery_method": decoded["recovery_method"],
        "phase_tile_counts": phase_counts,
        "mean_abs_score": round(float(np.mean(np.abs(average_scores))), 6),
    }


def _record_v3_auth_code(record: Record) -> bytes | None:
    """从记录里取出并校验 v3 的 8 字节 HMAC 认证码。

    :return: 8 字节认证码；格式不合规返回 ``None``。

    只接受严格的 16 位十六进制（``strip().lower()`` 先归一化大小写和空白）。
    长度、正则、``fromhex``、解出后再验字节数——四道检查看着重复，
    但这个字段直接来自数据库，脏数据一旦漏进解码就会变成难查的形状错误。

    所有异常都收敛成 ``None``，调用方跳过这条记录即可，不必写 try。
    """
    text = str(record.get("robust_auth_code") or "").strip().lower()
    if len(text) != 16 or not re.fullmatch(r"[0-9a-f]{16}", text):
        return None
    try:
        code = bytes.fromhex(text)
    except ValueError:
        return None
    return code if len(code) == 8 else None


def decode_aligned_robust_trace_v3(
    alignment: Record,
    record: Record,
    max_errors: int = 8,
    *,
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> Record | None:
    """**v3** 对齐解码：逐块归一化、逆置换，再核对 HMAC 认证码。

    :param alignment: 视觉配准结果，需含 ``image`` / ``valid_mask`` / ``target_scale``。
    :param record: 待验证的候选记录，其 ``robust_auth_code`` 提供期望的认证码。
    :param max_errors: 允许的最大比特错误数，默认 8（共 ``robust_bits`` = 64 位）。
    :return: 命中时返回带认证证据的字典，否则 ``None``。

    骨架仍是"原图坐标枚举方块 → 覆盖率闸门 → 缩回标准尺寸 → 累加相关值"，
    相对 v1/v2 有两处实质改进：

    1. **逐方块归一化**。每块先算出 64 个原始相关值，除以本块的
       ``median|score|`` 再截断到 ``±3``，然后才累加进总和。这样高对比度区域
       不会仅凭幅度大就压过平坦区域的方块；截断则挡住个别过曝/欠曝小格
       贡献出的极端值。v1/v2 直接累加原始值，没有这层保护。
    2. **逆置换**。嵌入时逻辑位 ``logical`` 被放到物理格 ``permutation[logical]``，
       这里按同一张映射表把物理相关值搬回逻辑位上。

    通过条件三道：

    * 至少 2 个方块参与；
    * 至少 2 个**不同相位**有贡献。v3 每块都装完整码，单个相位就够解出内容了，
      多要一个相位纯粹是防误报——两套不同置换给出一致结论，才不像噪声凑巧对上；
    * 比特错误不超过 ``max_errors``，**且** ``mean_signed_agreement > 0``。

    最后那条是对硬判决的软性补充：把平均相关值乘上期望比特的符号（0→-1，1→+1）
    再取均值，真水印会得到明显为正的值；就算硬判决勉强过线，
    整体相关方向若没有倾向性，照样判失败。

    64 位码允许 8 位误差看着松，但认证码由 HMAC 生成、不可伪造：
    随机比特串落进某个 64 位码的 8 位邻域内的概率约 ``2.7e-10``，
    这点松弛换来的抗噪收益远大于误报风险。
    """
    trace_id = record.get("trace_id")
    auth_code = (dependencies.record_v3_auth_code or _record_v3_auth_code)(record)
    aligned = alignment.get("image")
    valid_mask = alignment.get("valid_mask")
    target_scale = float(alignment.get("target_scale", 1.0))
    if (
        not trace_id
        or auth_code is None
        or not isinstance(aligned, np.ndarray)
        or not isinstance(valid_mask, np.ndarray)
        or aligned.ndim != 3
        or aligned.shape[:2] != valid_mask.shape
        or target_scale <= 0
    ):
        return None
    height, width = aligned.shape[:2]
    original_width = int(round(width / target_scale))
    original_height = int(round(height / target_scale))
    aggregate_scores = np.zeros(config.robust_bits, dtype=np.float64)
    phase_counts = [0, 0, 0]
    authenticated_tiles = 0
    tiles = dependencies.iter_robust_tiles(original_width, original_height) if dependencies.iter_robust_tiles else iter_robust_tiles(original_width, original_height, config=config)
    tile_phase_fn = dependencies.tile_phase or tile_phase
    phase_permutation_fn = dependencies.phase_permutation or phase_permutation
    pattern_fn = dependencies.robust_pattern or robust_pattern
    for x, y in tiles:
        x0 = max(0, int(round(x * target_scale)))
        y0 = max(0, int(round(y * target_scale)))
        x1 = min(width, int(round((x + config.robust_tile) * target_scale)))
        y1 = min(height, int(round((y + config.robust_tile) * target_scale)))
        if x1 <= x0 or y1 <= y0:
            continue
        mask_tile = valid_mask[y0:y1, x0:x1]
        # 覆盖率闸门，同 v1/v2
        if not mask_tile.size or float(mask_tile.mean()) < 0.70:
            continue
        tile = cv2.resize(aligned[y0:y1, x0:x1, :], (config.robust_tile, config.robust_tile), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        # 先按物理格顺序读，逆置换留到归一化之后再做
        physical_scores = np.zeros(config.robust_bits, dtype=np.float64)
        for bit_index in range(config.robust_bits):
            row, col = divmod(bit_index, config.robust_grid)
            cell = tile[row * config.robust_cell : (row + 1) * config.robust_cell, col * config.robust_cell : (col + 1) * config.robust_cell, config.robust_channel]
            centered = cell - cell.mean()
            physical_scores[bit_index] = float(np.mean(centered * pattern_fn(bit_index, config.robust_cell)))
        # 用中位数（而非均值/最大值）归一：中位数不受少数极端小格影响，
        # 能稳定反映本块的相关值量级；截断到 ±3 再挡一道离群值
        scale = max(1e-6, float(np.median(np.abs(physical_scores))))
        physical_scores = np.clip(physical_scores / scale, -3.0, 3.0)
        phase = tile_phase_fn(x // config.robust_tile, y // config.robust_tile)
        permutation = phase_permutation_fn(phase)
        # 逆置换：permutation[logical] 是嵌入时该逻辑位所占的物理格
        for logical, physical in enumerate(permutation):
            aggregate_scores[logical] += physical_scores[physical]
        phase_counts[phase] += 1
        authenticated_tiles += 1
    if authenticated_tiles < 2 or sum(count > 0 for count in phase_counts) < 2:
        return None
    expected_value = int.from_bytes(auth_code, "big")
    expected_bits = np.array([(expected_value >> shift) & 1 for shift in range(config.robust_bits - 1, -1, -1)], dtype=np.int8)
    observed_bits = (aggregate_scores > 0).astype(np.int8)
    bit_errors = int(np.count_nonzero(observed_bits != expected_bits))
    # 期望比特映射成 ±1，与平均相关值同号相乘：整体为正才说明相关方向确实一致
    expected_signs = expected_bits.astype(np.float64) * 2.0 - 1.0
    average_scores = aggregate_scores / authenticated_tiles
    mean_signed_agreement = float(np.mean(average_scores * expected_signs))
    if bit_errors > max_errors or mean_signed_agreement <= 0:
        return None
    return {
        "record": record,
        "trace_id": trace_id,
        "bit_errors": bit_errors,
        "authenticated_tiles": authenticated_tiles,
        "phase_tile_counts": phase_counts,
        "mean_signed_agreement": round(mean_signed_agreement, 6),
        "mean_abs_score": round(float(np.mean(np.abs(average_scores))), 6),
    }


def detect_aligned_authenticated_watermark(
    image: Image.Image,
    candidate_limit: int = 8,
    budget_seconds: float = 5.0,
    *,
    records: RecordsSource,
    rank_candidates: Callable[[Image.Image, list[Record]], list[Record]],
    align_query: Callable[[Image.Image, Record], Record | None],
    decode_v1: Callable[[Record, Record], Record | None],
    decode_v2: Callable[[Record, Record], Record | None],
    decode_v3: Callable[[Record, Record], Record | None],
    normalize_version: Callable[[str | int | None], int],
    with_evidence_fields: Callable[[Record, Record | None], Record],
    now_text: Callable[[], str],
    version_v1: int,
    version_v2: int,
    version_v3: int,
    codec_v2: str,
    codec_v3: str,
    watermark_layers: Any,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> Record | None:
    """**对齐检测**总入口：候选排序 → 视觉配准 → 按版本分派解码 → 组装取证报告。

    :param candidate_limit: 最多尝试几条候选记录（下限 1 条）。
    :param budget_seconds: 时间预算，``<= 0`` 表示不限时。
    :param rank_candidates: 用视觉特征给候选排序，把最可能命中的排前面。
    :param align_query: 把查询图配准到某条记录的原图；失败返回 ``None``。
        返回的字典必须带 ``inliers`` / ``ratio`` / ``coverage``，下面直接下标取用。
    :param decode_v1: v1 对齐解码器。
    :param decode_v2: v2 对齐解码器。
    :param decode_v3: v3 对齐解码器。
    :param normalize_version: 把记录里的版本字段归一成整数版本号。
    :param with_evidence_fields: 给结果补齐通用取证字段。
    :return: 命中时返回检测报告，否则 ``None``。

    每条候选都得单独配准一次——配准是"查询图对某张原图"的，没有可以共用的
    目标几何，所以这条路径很贵。``rank_candidates`` 的排序质量和
    ``candidate_limit`` 直接决定了实际能试到几条、要花多久。

    **必须恰好一个候选通过认证**。多于一个说明有好几条记录都"解得出"，
    此时无从判断哪条才对，宁可返回 ``None`` 让上层继续走别的检测层，
    也绝不猜一个可能错的溯源号——取证给错结论比给不出结论更糟。

    置信度 ``99 - bit_errors × 6`` 夹到 ``[80, 99]``：能走到这一步说明认证已经通过，
    起点就该高；下限守在 80 是因为即便错了几位，HMAC / RS 的认证依然成立，
    不该用低分暗示结果不可靠。
    """
    started = perf_counter()
    current_records = _resolve_records(records)
    candidates = [record for record in current_records if record.get("trace_id") and record.get("robust_watermark")]
    candidates = rank_candidates(image, candidates)[: max(1, candidate_limit)]
    authenticated = []
    for record in candidates:
        # 预算只在开始下一条候选前检查，已经开工的这条会跑完；
        # 也就是说总耗时最多超出一次"配准 + 解码"的时间
        if budget_seconds > 0 and perf_counter() - started >= budget_seconds:
            break
        alignment = align_query(image, record)
        if not alignment:
            continue
        version = normalize_version(record.get("robust_watermark_version", version_v1))
        if version == version_v3:
            decoded = decode_v3(alignment, record)
        elif version == version_v2:
            decoded = decode_v2(alignment, record)
        else:
            decoded = decode_v1(alignment, record)
        if decoded:
            authenticated.append((decoded, alignment))
    trace_ids = {decoded["trace_id"] for decoded, _ in authenticated}
    # 恰好一条才算数。注：len(authenticated) == 1 时 trace_ids 必然也只有一个，
    # 后半个条件是冗余的保险，保持原样
    if len(authenticated) != 1 or len(trace_ids) != 1:
        return None
    decoded, alignment = authenticated[0]
    record = decoded["record"]
    version = normalize_version(record.get("robust_watermark_version", version_v1))
    confidence = max(80, min(99, 99 - decoded["bit_errors"] * 6))
    # 配准指标三件套是各版本共有的证据，版本相关的细节在下面按分支追加
    code_recovery = {
        "visual_inliers": alignment["inliers"],
        "visual_ratio": alignment["ratio"],
        "aligned_coverage": alignment["coverage"],
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
    }
    if version == version_v3:
        code_recovery.update({
            "method": "homography_aligned_hmac64_full_repeat_v3",
            "codec": codec_v3,
            "bit_errors": decoded["bit_errors"],
            "authenticated_tiles": decoded["authenticated_tiles"],
            "phase_tile_counts": decoded["phase_tile_counts"],
            "mean_signed_agreement": decoded["mean_signed_agreement"],
            "mean_abs_score": decoded["mean_abs_score"],
        })
    elif version == version_v2:
        code_recovery.update({
            "method": "homography_aligned_rs_24_8_three_phase",
            "codec": codec_v2,
            "bit_errors": decoded["bit_errors"],
            "corrected_symbols": decoded["corrected_symbols"],
            "erasure_count": decoded["erasure_count"],
            "recovery_method": decoded["recovery_method"],
            "phase_tile_counts": decoded["phase_tile_counts"],
            "mean_abs_score": decoded["mean_abs_score"],
        })
    else:
        code_recovery.update({
            "method": "homography_aligned_robust_64",
            "bit_errors": decoded["bit_errors"],
            "magic_errors": decoded["magic_errors"],
            "authenticated_tiles": decoded["authenticated_tiles"],
            "mean_abs_score": decoded["mean_abs_score"],
        })
    return with_evidence_fields({
        "id": record.get("id"),
        "trace_id": decoded["trace_id"],
        "user_id": record.get("user_id"),
        "mode": "aligned_robust_hmac_v3" if version == version_v3 else "aligned_robust_rs_v2" if version == version_v2 else "aligned_robust_code",
        "mode_label": "几何对齐 HMAC 认证水印" if version == version_v3 else "几何对齐 RS 认证水印" if version == version_v2 else "几何对齐 64-bit 认证水印",
        "created_at": record.get("created_at"),
        "confidence": confidence,
        "phash_match": False,
        "status": "认证水印恢复",
        "extracted_at": now_text(),
        "watermark_layers": record.get("watermark_layers", watermark_layers),
        "code_recovery": code_recovery,
    }, record)


def extract_robust_code(
    image: Image.Image,
    *,
    records: Iterable[Record],
    config: RobustConfig = DEFAULT_CONFIG,
    dependencies: RobustDependencies = DEFAULT_DEPENDENCIES,
) -> tuple[str | None, float, int]:
    """**盲检测**：不知道几何变换，靠暴力试探网格参数提取 v1 码。

    :param records: 候选记录。调用方应当只传 **v1** 的记录
        （见 :func:`legacy_robust_candidate_records`），因为这里比对的是 v1 期望码。
    :return: ``(溯源号或 None, 置信度, 已判定的比特数)``。

    先把所有候选的期望码算成一张 ``码 → 溯源号`` 的表，再在多组网格参数下提取，
    提到的码回表里做**精确**匹配。这里刻意不做模糊匹配：盲提取本身噪声就大，
    再加上容错，误报会直接失控（容错反查请走对齐检测那条路）。

    偏移只需扫一个方块周期（``min(tile, ...)``），再往后就是重复；
    步长取 ``2 × cell`` 是精度与耗时的折中——整个扫描已经是
    "网格组合数 × 方块数 × 64 位"量级的相关运算。

    多组参数可能同时命中，按置信度降序取最高的那一个。

    ``cell`` 候选里 **16 必须排在第一位**：它等于嵌入时用的 ``ROBUST_CELL``，
    对应未经缩放的原尺寸图片，是最常见也最该先试的情形。其余 8/7/9 覆盖的是
    图片被缩到约 50% / 43.75% / 56.25% 的场景。

    .. warning::
        本函数整体开销极高——1080p 图上实测约 200 秒，远超配置的在线预算。
        它只在 ``dense_watermark_fallback_enabled``（默认关闭）打开时才会被
        流水线调用。若要启用该开关，务必先对本函数做性能改造。
    """
    trace_codes = {
        record.get("trace_id"): (
            dependencies.robust_code_from_trace(record.get("trace_id"))
            if dependencies.robust_code_from_trace
            else robust_code_from_trace(record.get("trace_id"), config=config)
        )
        for record in records
        if record.get("trace_id")
    }
    if not trace_codes:
        return None, 0.0, 0
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    if width < config.robust_tile or height < config.robust_tile:
        return None, 0.0, 0
    candidates = []
    # 先试 8，再试 ±1，覆盖轻微缩放导致的小格边长漂移
    # 16 = ROBUST_CELL，对应原尺寸图；其余三档对应被缩放过的图片。
    for cell in (16, 8, 7, 9):
        tile = cell * config.robust_grid
        step = max(1, cell * 2)
        for offset_y in range(0, min(tile, height - tile + 1), step):
            for offset_x in range(0, min(tile, width - tile + 1), step):
                code, confidence, decided = (
                    dependencies.extract_robust_from_grid(arr, cell, offset_x, offset_y)
                    if dependencies.extract_robust_from_grid
                    else extract_robust_from_grid(
                        arr,
                        cell,
                        offset_x,
                        offset_y,
                        config=config,
                        dependencies=dependencies,
                    )
                )
                # 有位没投出票，说明网格根本没对齐，整份结果作废
                if code is None or decided < config.robust_bits:
                    continue
                trace_id = next((item for item, item_code in trace_codes.items() if item_code == code), None)
                if trace_id is not None:
                    candidates.append((trace_id, confidence, decided))
    if not candidates:
        return None, 0.0, 0
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0]


def detect_robust_watermark(
    image: Image.Image,
    *,
    records: Iterable[Record],
    extract_code: Callable[[Image.Image, list[Record]], tuple[str | None, float, int]],
    with_evidence_fields: Callable[[Record, Record | None], Record],
    now_text: Callable[[], str],
    layer_scores_for_image: Callable[[Image.Image, str], Any],
    watermark_layers: Any,
) -> Record | None:
    """盲检测的报告封装：调 ``extract_code`` 拿溯源号，再组装成检测结果。

    :param extract_code: 实际的提取函数，通常是 :func:`extract_robust_code`。
    :param layer_scores_for_image: 补充各频域层的相关分，作为佐证写进报告。
    :return: 命中时返回检测报告，否则 ``None``。

    ``confidence < 0.08`` 是这一层的准入线。盲提取的置信度是投票优势的均值，
    真水印通常远高于此。门槛压这么低是因为它只是初筛，
    ``extract_code`` 内部已经做过精确码比对兜底，
    而漏检的代价（整条链路查不到）比多算一次比对大得多。

    对外报出的置信度映射到 ``[75, 98]``，比对齐检测的 ``[80, 99]`` 低一档——
    这条路径没有密码学认证，可信度天然弱一些。

    报告里的 ``phash_match`` 为 ``False``：本函数完全靠解码得出结论，
    没有做任何图像相似度或哈希比对。只有 ``detect_by_visual_match``
    那类真正比对过图像的检测器才应置 ``True``。
    """
    # 下面要遍历两次，先物化一份，免得传进来的是一次性迭代器
    records = list(records)
    if not records:
        return None
    trace_id, confidence, decided = extract_code(image, records)
    if not trace_id or confidence < 0.08:
        return None
    record = next((item for item in records if item.get("trace_id") == trace_id), None)
    if not record:
        return None
    return with_evidence_fields({
        "id": record.get("id"),
        "trace_id": trace_id,
        "user_id": record.get("user_id"),
        "mode": "robust_dct",
        "mode_label": "30% 局部鲁棒水印",
        "created_at": record.get("created_at"),
        "confidence": int(min(98, max(75, confidence * 100))),
        "phash_match": False,
        "status": "鲁棒水印命中",
        "extracted_at": now_text(),
        "robust_decided_bits": decided,
        "robust_score": round(confidence, 3),
        "watermark_layers": record.get("watermark_layers", watermark_layers),
        "layer_scores": layer_scores_for_image(image, trace_id),
    }, record)
