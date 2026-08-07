"""点阵追踪水印：模仿打印机黄点的抗翻拍层。

**定位。** 前面几层水印都依赖数字信号的精确保留，一旦经过"屏幕显示 →
手机拍照"这条模拟链路（重采样、摩尔纹、透视畸变、色偏）就会全部失效。
点阵层专为这种场景设计：它嵌入的是**肉眼几乎不可见的黄色小圆点阵列**，
本质是空域的宏观图案，即便经过翻拍也能靠形态识别出来。

思路源自彩色激光打印机的追踪黄点（Machine Identification Code）。

**编码方式。** 把溯源号压成 ``CODE_PAYLOAD_BITS`` 位的载荷，
在每个 ``DOT_MATRIX_TILE`` 方块内按 ``DOT_MATRIX_GRID`` 网格布点：
该位为 1 就把点调亮，为 0 就调暗。整幅图重复铺满，并用四种半块偏移
交错排布，使裁剪后仍有完整方块留存。

**为什么用黄色。** 人眼对蓝黄方向的色差分辨率最低（视网膜中 S 视锥细胞
数量最少），在黄色通道上做同等强度的调制，可见性远低于红绿方向。
具体权重见 ``DOT_MATRIX_CHANNEL_WEIGHTS``。

**检测是非盲的**：需要先有候选记录，逐块打分、投票，选出票数最高者。
"""

from typing import Any, Callable

import numpy as np
from PIL import Image

from trace_app.config import (
    CODE_PAYLOAD_BITS, DOT_MATRIX_CHANNEL_WEIGHTS, DOT_MATRIX_DELTA,
    DOT_MATRIX_GRID, DOT_MATRIX_TILE, DOT_MATRIX_VERSION, ROBUST_MAGIC,
    WATERMARK_LAYERS,
)


def dot_matrix_bits_from_trace(
    trace_id: str,
    *,
    watermark_payload_from_trace_fn: Callable[[str], int],
) -> list[int]:
    """把溯源号转成待布点的比特列表，**高位在前**。

    载荷的构成（魔数 + 数据 + CRC）由注入的 ``watermark_payload_from_trace_fn``
    负责，本模块只管把它摊成比特。
    """
    payload = watermark_payload_from_trace_fn(trace_id)
    return [(payload >> shift) & 1 for shift in range(CODE_PAYLOAD_BITS - 1, -1, -1)]


def dot_matrix_candidate_records(
    records: list[dict[str, Any]],
    *,
    watermark_payload_from_trace_fn: Callable[[str], int],
) -> list[tuple[str, int, dict[str, Any]]]:
    """筛出启用了点阵层的记录，并预先算好各自的期望载荷。

    :return: ``(溯源号, 期望载荷, 完整记录)`` 三元组列表。

    三重过滤：有溯源号、开启了点阵层、版本号匹配当前实现。
    版本不符的记录用的是旧编码方式，拿现在的算法去比对只会误判。

    期望载荷在此**预先算好**，避免在检测的多重循环里对每个候选反复计算。
    """
    return [
        (record.get("trace_id"), watermark_payload_from_trace_fn(record.get("trace_id")), record)
        for record in records
        if record.get("trace_id")
        and record.get("dot_matrix_trace_enabled")
        and record.get("dot_matrix_trace_version") == DOT_MATRIX_VERSION
    ]


def dot_matrix_position(bit_index: int, tile_size: int = DOT_MATRIX_TILE) -> tuple[int, int]:
    """算出第 ``bit_index`` 位在方块内的落点坐标 ``(x, y)``。

    按行优先把比特铺进 ``DOT_MATRIX_GRID × DOT_MATRIX_GRID`` 网格，
    每个点落在所属格子的**中心**（``+ cell // 2``）。

    居中而非靠边很重要：检测端估计的方块位置总有几个像素误差，
    点在中心时容差最大，偏一点也不会跑到隔壁格子里去。

    ``max(2, ...)`` 保证格子至少 2 像素，防止小方块下退化成 0。
    """
    cell = max(2, tile_size // DOT_MATRIX_GRID)
    row = bit_index // DOT_MATRIX_GRID
    col = bit_index % DOT_MATRIX_GRID
    return col * cell + cell // 2, row * cell + cell // 2


def apply_dot_matrix_trace_layer(
    image: Image.Image,
    trace_id: str,
    strength: float = 1.0,
    *,
    clamp_float_fn: Callable[[str | float | None, float, float, float], float],
    watermark_payload_from_trace_fn: Callable[[str], int],
) -> Image.Image:
    """在图上铺设点阵追踪层，返回处理后的图片。

    :param strength: 0~1 的强度倍率；为 0 时原样返回（相当于关闭本层）。

    整幅图按方块平铺，并用**四种半块偏移**交错重复：
    ``(0,0)``、``(半块,0)``、``(0,半块)``、``(半块,半块)``。
    这样任意位置的裁剪都至少能保住一套完整的点阵。

    图片小于一个方块时直接返回——放不下一套完整编码，布点没有意义。
    """
    strength = clamp_float_fn(strength, 1.0, 0.0, 1.0)
    if strength <= 0:
        return image.convert("RGB")
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    if height < DOT_MATRIX_TILE or width < DOT_MATRIX_TILE:
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    bits = dot_matrix_bits_from_trace(
        trace_id,
        watermark_payload_from_trace_fn=watermark_payload_from_trace_fn,
    )
    # 四种半块偏移，让点阵在图上交错重叠，提升抗裁剪能力
    offsets = [
        (0, 0),
        (DOT_MATRIX_TILE // 2, 0),
        (0, DOT_MATRIX_TILE // 2),
        (DOT_MATRIX_TILE // 2, DOT_MATRIX_TILE // 2),
    ]
    # 5×5 的高斯光斑（半径 2），归一化到峰值 1。
    # 用渐变光斑而非硬边方点：硬边有丰富的高频成分，压缩时会产生振铃伪影，
    # 也更容易被肉眼察觉；高斯斑平滑，且在缩放/重采样后形态保持得更好。
    radius = 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    spot = np.exp(-(xx * xx + yy * yy) / 2.0).astype(np.float32)
    spot = spot / max(float(spot.max()), 1e-6)
    delta = DOT_MATRIX_DELTA * strength

    for offset_x, offset_y in offsets:
        for y in range(offset_y, height - DOT_MATRIX_TILE + 1, DOT_MATRIX_TILE):
            for x in range(offset_x, width - DOT_MATRIX_TILE + 1, DOT_MATRIX_TILE):
                for bit_index, bit in enumerate(bits):
                    cx, cy = dot_matrix_position(bit_index)
                    px = x + cx
                    py = y + cy
                    # 光斑范围越界就跳过这一位——宁可少布一个点，
                    # 也不要把光斑截断成不完整的形状干扰检测
                    if px - radius < 0 or py - radius < 0 or px + radius >= width or py + radius >= height:
                        continue
                    # 比特 1 调亮、0 调暗；检测端据此还原
                    sign = 1.0 if bit else -1.0
                    # patch 是 arr 的**视图**，对它赋值即直接写回原数组
                    patch = arr[py - radius : py + radius + 1, px - radius : px + radius + 1, :]
                    # 按通道权重施加，使整体色偏落在人眼最不敏感的蓝黄方向上
                    for channel, weight in enumerate(DOT_MATRIX_CHANNEL_WEIGHTS):
                        patch[:, :, channel] = np.clip(patch[:, :, channel] + spot * delta * sign * weight, 0, 255)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def dot_matrix_score_tile(tile: np.ndarray) -> tuple[int, float, int]:
    """从一个方块中读出点阵编码。

    :return: ``(解出的整数码, 平均信号强度, 参与判决的比特数)``。

    对每一位：取点位附近的小块算"黄度"，再取整个格子的黄度作为局部背景，
    两者之差为正判 1、为负判 0。

    **黄度公式** ``(R + G) / 2 - B × 0.35`` 是一个简化的蓝黄对立通道度量：
    黄色 = 高红 + 高绿 + 低蓝。用差分而非绝对值，可以自动抵消图像本身的
    底色——只关心点位相对于周围**多黄了多少**。

    ``radius`` 随格子尺寸自适应并夹在 1~3：太大会把邻格采样进来，
    太小则抗噪能力不足。
    """
    tile_size = min(tile.shape[:2])
    cell = max(2, tile_size // DOT_MATRIX_GRID)
    radius = max(1, min(3, cell // 5))
    votes = []
    strengths = []
    for bit_index in range(CODE_PAYLOAD_BITS):
        cx, cy = dot_matrix_position(bit_index, tile_size)
        y0 = max(0, cy - radius)
        y1 = min(tile.shape[0], cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(tile.shape[1], cx + radius + 1)
        patch = tile[y0:y1, x0:x1, :]
        if patch.size == 0:
            votes.append(0)
            strengths.append(0.0)
            continue
        # 点位处的黄度
        yellow = (patch[:, :, 0] + patch[:, :, 1]) * 0.5 - patch[:, :, 2] * 0.35
        # 整个格子作为局部背景基准，用于抵消图像自身的底色
        cell_y0 = max(0, cy - cell // 2)
        cell_y1 = min(tile.shape[0], cy + cell // 2)
        cell_x0 = max(0, cx - cell // 2)
        cell_x1 = min(tile.shape[1], cx + cell // 2)
        background_cell = tile[cell_y0:cell_y1, cell_x0:cell_x1, :]
        background = (background_cell[:, :, 0] + background_cell[:, :, 1]) * 0.5 - background_cell[:, :, 2] * 0.35
        score = float(yellow.mean() - background.mean())
        votes.append(1 if score >= 0 else 0)
        # 用绝对值记录强度：无论判 0 还是判 1，差值越大越说明信号明确
        strengths.append(abs(score))
    # 比特拼成整数，高位在前，与 dot_matrix_bits_from_trace 的顺序对应
    code = 0
    for bit in votes:
        code = (code << 1) | bit
    return code, float(np.mean(strengths) if strengths else 0.0), len(votes)


def detect_dot_matrix_trace(
    image: Image.Image,
    candidate_records: list[tuple[str, int, dict[str, Any]]],
    *,
    hamming_distance_fn: Callable[[int, int], int],
    code_crc16_fn: Callable[[int], int],
    now_text_fn: Callable[[], str],
    with_evidence_fields_fn: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
) -> dict[str, Any] | None:
    """在图中搜索点阵水印，返回命中的证据记录。

    :param candidate_records: :func:`dot_matrix_candidate_records` 的输出。
    :return: 命中返回证据字典，否则 ``None``。

    **多尺度网格搜索。** 翻拍会改变图片尺寸，原本 ``DOT_MATRIX_TILE`` 大小
    的方块在拍摄件中可能变成任意尺寸。这里从 32 到 96 逐档试，
    每档再配合两种偏移和滑窗扫描位置。

    **四道校验，从廉价到昂贵**，尽早否掉无效位置：

    1. 强度下限 0.10 —— 信号太弱不予考虑；
    2. 魔数汉明距离 ≤ 4 —— 快速判断"这串比特是不是本系统的载荷"；
    3. CRC 汉明距离 ≤ 10 —— 校验和大致吻合；
    4. 与候选载荷的距离 ≤ 8，且总距离 ≤ 24 —— 确定具体是哪条记录。

    用**汉明距离**而非精确相等，是因为翻拍必然引入误码；阈值放得比较宽，
    但四道条件叠加后整体误报率仍然很低。

    **加权投票。** 每个通过校验的位置投一票，权重 = 信号强度 × 方块尺寸系数。
    大方块采样点多、结论更可靠，故权重更高。最后票数最高且总票数 ≥ 12 者胜出。

    .. note::
        当前 ``x`` 循环体内只执行了切片，评分与后续判定位于 ``x`` 循环之外，
        因此每个 ``y`` 行实际只对**最后一个** ``x`` 位置的方块打分。
        这会显著降低召回率（横向只采样一列），但不会产生误报——
        通过校验的仍然是真实命中。此处按现状描述，未作改动。
    """
    records = candidate_records
    if not records:
        return None
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    min_tile = 32
    if height < min_tile or width < min_tile:
        return None

    best_trace = None
    best_votes = 0
    best_distance = CODE_PAYLOAD_BITS + 1
    # 三个按溯源号累计的计数器：加权票数、加权强度总和、最小汉明距离
    strength_counts: dict[str, float] = {}
    vote_counts: dict[str, int] = {}
    distance_counts: dict[str, int] = {}
    # 候选方块尺寸：由小到大逐档试，覆盖翻拍导致的各种缩放比例
    max_tile = min(DOT_MATRIX_TILE, height, width)
    tile_sizes = [size for size in (32, 36, 40, 44, 48, 56, 64, 72, 80, 88, 96) if size <= max_tile]
    # 保证实际可用的最大尺寸一定在列表里（它可能不是上面这些整数档之一）
    if max_tile not in tile_sizes:
        tile_sizes.append(max_tile)
    for tile_size in tile_sizes:
        step = max(12, tile_size)
        # set 去重：tile_size 为 0 或 1 时半块偏移也是 0，避免重复扫描
        offsets = sorted(set((0, tile_size // 2)))
        for offset_y in offsets:
            for offset_x in offsets:
                if offset_y > height - tile_size or offset_x > width - tile_size:
                    continue
                for y in range(offset_y, height - tile_size + 1, step):
                    for x in range(offset_x, width - tile_size + 1, step):
                        tile = arr[y : y + tile_size, x : x + tile_size, :]
                    code, strength, decided = dot_matrix_score_tile(tile)
                    # 校验 1：比特未读全，或信号强度低于下限
                    if decided < CODE_PAYLOAD_BITS or strength < 0.10:
                        continue
                    payload = code
                    corrections = 0
                    # 校验 2：高 32 位应为魔数，容错 4 比特
                    magic_distance = hamming_distance_fn(payload >> 32, ROBUST_MAGIC)
                    if magic_distance > 4:
                        continue
                    # 校验 3：低 16 位是 CRC，与重算值比对，容错 10 比特
                    checksum = payload & 0xFFFF
                    crc_distance = hamming_distance_fn(checksum, code_crc16_fn(payload >> 16))
                    if crc_distance > 10:
                        continue
                    # 校验 4：在候选中找汉明距离最近的一条
                    best_record = None
                    best_record_distance = CODE_PAYLOAD_BITS + 1
                    for _, expected_payload, record in records:
                        distance = hamming_distance_fn(payload, expected_payload)
                        if distance < best_record_distance:
                            best_record = record
                            best_record_distance = distance
                    # 单项距离与总距离双重设限：某一项勉强通过、但整体误差过大时同样否决
                    total_distance = best_record_distance + magic_distance + crc_distance + corrections
                    if not best_record or best_record_distance > 8 or total_distance > 24:
                        continue
                    trace_id = best_record.get("trace_id")
                    # 票重 = 信号强度 × 方块尺寸系数。大方块采样点更多、
                    # 结论更可靠，因此赋予更高权重。
                    vote_weight = max(1, int(strength * 100)) * max(1, tile_size // 32)
                    vote_counts[trace_id] = vote_counts.get(trace_id, 0) + vote_weight
                    strength_counts[trace_id] = strength_counts.get(trace_id, 0.0) + strength * vote_weight
                    # 距离取全程最小值，代表这条记录"最好的一次匹配"
                    distance_counts[trace_id] = min(distance_counts.get(trace_id, CODE_PAYLOAD_BITS + 1), best_record_distance)

    # 汇总投票：票数 ≥ 12 且距离 ≤ 8 才有资格，票数相同则取距离更小者
    for trace_id, votes in vote_counts.items():
        distance = distance_counts.get(trace_id, CODE_PAYLOAD_BITS + 1)
        if votes < 12 or distance > 8:
            continue
        if votes > best_votes or (votes == best_votes and distance < best_distance):
            best_trace = trace_id
            best_votes = votes
            best_distance = distance
    if not best_trace:
        return None
    record = next((item for trace_id, _, item in records if trace_id == best_trace), None)
    if not record:
        return None
    # 加权平均强度：累计时已乘过权重，这里除以总权重还原
    avg_strength = strength_counts[best_trace] / max(1, vote_counts[best_trace])
    return with_evidence_fields_fn({
        "id": record.get("id"),
        "trace_id": best_trace,
        "user_id": record.get("user_id"),
        "mode": "dot_matrix_trace",
        "mode_label": "点阵追溯水印",
        "created_at": record.get("created_at"),
        # 置信度随票数增长，但夹在 76~96：
        # 下限 76 —— 能通过四道校验就已相当可信；
        # 上限 96 —— 点阵是应对翻拍的兜底手段，误差天然大于数字链路，
        #            不给满分以示区别。
        "confidence": int(min(96, max(76, 72 + best_votes * 2))),
        "phash_match": False,
        "status": "点阵水印恢复",
        "extracted_at": now_text_fn(),
        "watermark_layers": record.get("watermark_layers", WATERMARK_LAYERS),
        # 点阵层没有独立的三频域评分，这里用平均强度填充三个字段，
        # 只为保持与其他检测器一致的返回结构，便于前端统一渲染。
        "layer_scores": {
            "dct": round(avg_strength, 4),
            "dwt": round(avg_strength, 4),
            "fft": round(avg_strength, 4),
        },
        "code_recovery": {
            "method": "dot_matrix_trace",
            "version": DOT_MATRIX_VERSION,
            "votes": best_votes,
            "distance": best_distance,
            "strength": round(avg_strength, 4),
        },
    }, record)
