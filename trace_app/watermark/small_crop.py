"""小裁剪追踪层：把溯源号压成短码后密铺全图，让一小块截图也能定位来源。

**要解决的问题。** 频域三件套和鲁棒水印都需要**整张图**（或至少大部分）
才能算出可信的相关性；攻击者只要截取画面中的一小块发出去，这些层就全部失效。
本模块的思路是把信息做小、做密：先把溯源号压成短码，再以
``SMALL_TRACE_TILE``（96 像素）为单位**平铺整幅图**，并用多套半块/四分之一块
偏移交错重叠。这样任意位置的裁剪，只要还剩下一个方块的面积，就有机会还原。

**方块里装了什么。** 单个方块的图案是四种载体的加权叠加
（见 :func:`apply_small_crop_trace_layer`）：

* **marker** —— 全系统共用的固定图案，用作"这里有水印"的快速门控，
  相关性不够高就直接跳过，省下后面昂贵的解码；
* **trace** —— 由溯源号派生的伪随机图案，用于非盲的二次确认；
* **code** —— ``CODE_PHYSICAL_BITS`` 位的完整载荷（魔数 + 数据 + 校验），
  信噪比足够时可以**盲解**出溯源号；
* **short_code** —— 仅 ``SMALL_TRACE_SHORT_BITS``（16）位的短码，权重最高。
  位数少意味着每位能分到更多能量，在小方块、强压缩下比完整载荷更容易活下来；
  代价是 16 位区分度有限，只能在候选集里靠汉明距离匹配，不能独立成立。

解码时先试 code，失败再退到 short_code，并对后者附加更严的 trace 相关性门槛。

**与其他层的关系。** 属于传统链路（非 V4）。V4 链路会**强制关闭本层**——
密铺产生的周期性纹理会在频谱上形成规律峰值，污染 V4 用来估计几何变换的
FFT 同步导频。检测流水线中它排在点阵层和几何对齐检测之后
（见 ``trace_app/watermark/detection.py``）。

**本模块还收留了编码层（code layer）。** ``apply_code_layer`` /
``detect_watermark_code`` 是同一套"平铺 + 相关性投票"思路的大方块版本
（``CODE_TILE`` = 160），两者共用 :func:`normalize_carrier` 等基础设施，
故放在一起。区别是编码层的方块更大、不做偏移交错，只求抗压缩不求抗裁剪。

**检测一律是非盲的**：必须先有候选记录集，且每个候选都要先通过
``record_visual_consistency``（几何/内容一致性）才允许参与投票。
"""

import hashlib
from functools import lru_cache
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from trace_app.config import (
    CODE_CHANNEL_WEIGHTS, CODE_DELTA, CODE_PAYLOAD_BITS, CODE_PHYSICAL_BITS,
    CODE_TILE, CODE_WATERMARK_VERSION, ROBUST_BITS, ROBUST_MAGIC,
    SMALL_TRACE_CHANNEL_WEIGHTS, SMALL_TRACE_DELTA, SMALL_TRACE_SHORT_BITS,
    SMALL_TRACE_TILE, SMALL_TRACE_VERSION, WATERMARK_LAYERS,
)

from trace_app.watermark.frequency import layer_seed


def _code_crc16(value: int) -> int:
    """算出 16 位校验和。

    名字里的 CRC 只是沿用叫法，实际用的是 blake2b 摘要截到 2 字节。
    这里不需要 CRC 多项式那种"检测突发错误"的特性——载荷的比特误差是
    随机分布的，而且下游是按**汉明距离**容错比对（见 :func:`match_small_trace_code`），
    并不要求校验和精确相等，所以只要求它对输入敏感、输出均匀即可。
    用 blake2b 还有个好处：与本系统其他派生逻辑同源，无需另引依赖。

    ``to_bytes(4, "big")`` 固定按 4 字节序列化，调用方传进来的正是
    "魔数 + 数据"这 32 位。
    """
    return int.from_bytes(hashlib.blake2b(value.to_bytes(4, "big"), digest_size=2).digest(), "big")


def _watermark_payload_from_trace(trace_id: str) -> int:
    """把溯源号派生成 ``CODE_PAYLOAD_BITS``（48）位载荷。

    结构固定为三段，高位在前：

    * 高 16 位 ``ROBUST_MAGIC`` —— 魔数，解码端拿它做第一道快速否决，
      避免把随机噪声当成水印；
    * 中 16 位 ``body`` —— 溯源号的 blake2b 摘要，真正携带身份信息；
    * 低 16 位 ``checksum`` —— 前 32 位的校验和。

    数据段只留 16 位是刻意的：位数越少，同样的嵌入能量摊到每位上就越多，
    小方块里的抗噪能力越强。16 位不足以全局唯一，但检测本来就是在
    候选集内比对（非盲），够用。
    """
    digest = hashlib.blake2b(trace_id.encode("utf-8"), digest_size=2).digest()
    body = int.from_bytes(digest, "big")
    checksum = _code_crc16((ROBUST_MAGIC << 16) | body)
    return (ROBUST_MAGIC << 32) | (body << 16) | checksum


def _watermark_bits_from_trace(trace_id: str) -> list[int]:
    """把 48 位载荷摊成 ``CODE_PHYSICAL_BITS``（64）位物理比特，高位在前。

    多出来的 16 位靠 ``index % CODE_PAYLOAD_BITS`` 循环取值填充，
    等于给载荷的**前 16 位**各加了一份副本——这是一种最简单的重复码，
    配合 :func:`_recover_payload_from_code` 的多数表决使用。

    为什么只保护前 16 位：那正是魔数所在的位置，是解码时第一道门控。
    魔数读错则整块方块直接作废，优先加固它的收益最大。
    """
    payload = _watermark_payload_from_trace(trace_id)
    base = [(payload >> shift) & 1 for shift in range(CODE_PAYLOAD_BITS - 1, -1, -1)]
    return [base[index % CODE_PAYLOAD_BITS] for index in range(CODE_PHYSICAL_BITS)]


def _recover_payload_from_code(code: int) -> tuple[int, int]:
    """从 64 位物理码还原 48 位载荷，:func:`_watermark_bits_from_trace` 的逆过程。

    :return: ``(还原出的载荷, 纠正次数)``。纠正次数即副本之间的分歧总数，
        可作为"这块读得有多干净"的附加信号，参与总距离判定。

    对每个载荷位收集它在物理码里的所有副本做多数表决。由于只有前 16 位
    有两份副本、其余 32 位只有一份，实际上只有前 16 位真正受益。

    .. note::
        副本数为 2 且两份不一致时，``one_votes >= zero_votes`` 会判成 1，
        即平票一律偏向 1，并不比单副本更准（但 ``corrections`` 会 +1，
        把这个不确定性传给上层）。此处按现状描述，未作改动。
    """
    bits = [(code >> shift) & 1 for shift in range(CODE_PHYSICAL_BITS - 1, -1, -1)]
    recovered = 0
    corrections = 0
    for index in range(CODE_PAYLOAD_BITS):
        # 步长取 CODE_PAYLOAD_BITS，正好命中同一载荷位的所有副本
        votes = [bits[pos] for pos in range(index, CODE_PHYSICAL_BITS, CODE_PAYLOAD_BITS)]
        one_votes = sum(votes)
        zero_votes = len(votes) - one_votes
        bit = 1 if one_votes >= zero_votes else 0
        # 少数派的数量即"被纠正掉"的比特数，累计成整块的可信度指标
        corrections += min(one_votes, zero_votes)
        recovered = (recovered << 1) | bit
    return recovered, corrections


def _clamp_float(value: str | float | None, default: float, low: float, high: float) -> float:
    """把任意来源的值转成浮点并夹到 ``[low, high]``，转不了就用默认值。

    输入多来自配置项或表单字符串，可能是 ``None``、空串或乱填的文本，
    这里统一兜住，避免把异常抛到嵌入流程里。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _hamming_distance(left: int, right: int) -> int:
    """两个整数的汉明距离：异或后数 1 的个数。"""
    return (left ^ right).bit_count()


def small_trace_short_code(
    trace_id: str,
    *,
    watermark_payload_from_trace_fn: Callable[[str], int] | None = None,
) -> int:
    """取溯源号短码：载荷的低 ``SMALL_TRACE_SHORT_BITS``（16）位。

    低 16 位恰好是校验和段。直接复用它而不另外派生一个短码，是因为
    校验和本身就是"魔数 + 数据"的均匀散列，作为身份指纹的区分度与
    重新派生一份没有差别，还省掉一次哈希。

    16 位在大候选集里必然存在碰撞，所以短码通道只在
    :func:`record_from_short_code_match` 里配合"最优/次优距离差"使用，
    不能单独定论。
    """
    payload_fn = watermark_payload_from_trace_fn or _watermark_payload_from_trace
    return payload_fn(trace_id) & ((1 << SMALL_TRACE_SHORT_BITS) - 1)


def small_trace_short_bits(
    trace_id: str,
    *,
    watermark_payload_from_trace_fn: Callable[[str], int] | None = None,
) -> np.ndarray:
    """把短码摊成 ±1 的浮点向量，高位在前。

    用 ±1 而非 0/1，是因为下游要与载波做张量积叠加：符号相反的两位会往
    相反方向调制像素，检测端做相关时正负自然分离；若用 0/1，比特 0
    就完全不调制，等于白白浪费一半的编码空间。
    """
    code = small_trace_short_code(
        trace_id,
        watermark_payload_from_trace_fn=watermark_payload_from_trace_fn,
    )
    return np.array(
        [1.0 if ((code >> shift) & 1) else -1.0 for shift in range(SMALL_TRACE_SHORT_BITS - 1, -1, -1)],
        dtype=np.float32,
    )


def small_crop_strength_to_scale(value: str | float | None) -> float:
    """把配置里的强度值规范成 ``[0, 1]`` 的倍率，缺省为 1.0（满强度）。

    取 0 相当于关闭本层（见 :func:`apply_small_crop_trace_layer` 的开头）。
    """
    return _clamp_float(value, 1.0, 0.0, 1.0)


def normalize_small_crop_density(value: str | None) -> str:
    """把密度设置规范成 ``low`` / ``medium`` / ``high`` 三档之一。

    同时接受中英文写法，因为这个值既可能来自前端下拉框（英文枚举），
    也可能来自数据库里历史遗留的中文记录。无法识别时一律回落到 ``low``——
    密度越高画质代价越大，拿不准就选最保守的。
    """
    text = str(value or "low").strip().lower()
    if text in {"medium", "中"}:
        return "medium"
    if text in {"high", "高"}:
        return "high"
    return "low"


def small_crop_density_offsets(density: str) -> list[tuple[int, int]]:
    """按密度档位给出方块起始偏移列表，嵌入端和检测端必须用同一套。

    :return: ``(offset_x, offset_y)`` 列表，low 2 套、medium 4 套、high 8 套。

    每一套偏移都会把整幅图重铺一遍。多套交错的意义在于：单套平铺时，
    横跨方块边界的裁剪会把所有方块都切成两半，一块完整的都不剩；
    加上半块偏移后，原本被切断的位置正好落在另一套的方块中心。

    档位递进的设计：

    * ``low`` —— 对角两套（``(0,0)`` 与 ``(半块,半块)``），最省画质预算；
    * ``medium`` —— 补齐另外两个半块方向，凑成完整的 2×2 半块网格；
    * ``high`` —— 再加四分之一块粒度的 4 套，把最坏情况下的裁剪容差
      从半块缩到四分之一块。代价是同一像素被叠加 8 次调制，噪点明显增加。
    """
    quarter = SMALL_TRACE_TILE // 4
    half = SMALL_TRACE_TILE // 2
    offsets = [(0, 0), (half, half)]
    if density in {"medium", "high"}:
        offsets.extend([(half, 0), (0, half)])
    if density == "high":
        offsets.extend([(quarter, quarter), (quarter * 3, quarter), (quarter, quarter * 3), (quarter * 3, quarter * 3)])
    return offsets


def iter_aligned_small_trace_tiles(
    aligned: np.ndarray,
    valid_mask: np.ndarray,
    record: dict[str, Any],
    target_scale: float = 1.0,
):
    """在几何对齐后的图上，按嵌入时的网格逐个吐出方块。

    :param aligned: 已经用单应变换摆正到原图坐标系的 RGB 数组。
    :param valid_mask: 与 ``aligned`` 同尺寸的布尔/数值掩膜，标出哪些像素
        是真实映射过来的（对齐后画面外围通常是空的）。
    :param record: 水印记录，只用来读密度设置。
    :param target_scale: ``aligned`` 相对原图的缩放比。对齐结果不一定是
        原始分辨率，网格必须按原图尺寸推算再映射回来。
    :return: 生成器，逐个产出 ``{"tile", "position", "offset", "coverage"}``。
        ``position`` 是**原图坐标**，便于与记录里的信息对照。

    与 :func:`iter_small_trace_windows` 的盲扫不同，这里已经知道了几何变换，
    可以直接落在嵌入时用过的网格上，方块数量少、每块都对齐得很准。
    这是几何对齐检测链路专用的取块方式。

    形状不合法或 ``target_scale`` 非正时直接返回空生成器——与其抛异常打断
    上层的多候选循环，不如让这个候选自然落空。
    """
    if aligned.ndim != 3 or aligned.shape[2] != 3:
        return
    if valid_mask.shape != aligned.shape[:2] or target_scale <= 0:
        return
    height, width = aligned.shape[:2]
    # 网格必须在原图坐标系里推算：嵌入时用的是原图像素的 SMALL_TRACE_TILE 步长
    original_width = int(round(width / target_scale))
    original_height = int(round(height / target_scale))
    density = normalize_small_crop_density(record.get("small_crop_trace_density"))
    for offset_x, offset_y in small_crop_density_offsets(density):
        for y in range(offset_y, original_height - SMALL_TRACE_TILE + 1, SMALL_TRACE_TILE):
            for x in range(offset_x, original_width - SMALL_TRACE_TILE + 1, SMALL_TRACE_TILE):
                # 原图坐标 → aligned 坐标，并夹进实际数组范围
                x0 = max(0, int(round(x * target_scale)))
                y0 = max(0, int(round(y * target_scale)))
                x1 = min(width, int(round((x + SMALL_TRACE_TILE) * target_scale)))
                y1 = min(height, int(round((y + SMALL_TRACE_TILE) * target_scale)))
                if x1 <= x0 or y1 <= y0:
                    continue
                mask_tile = valid_mask[y0:y1, x0:x1]
                # 覆盖率 = 这块里有多少像素是真实映射来的。低于 70% 说明大半是
                # 对齐后的空白填充，解码只会得到噪声，不如直接跳过省时间。
                coverage = float(mask_tile.mean()) if mask_tile.size else 0.0
                if coverage < 0.70:
                    continue
                tile = aligned[y0:y1, x0:x1, :]
                # 统一缩回 SMALL_TRACE_TILE：解码端的载波都是这个尺寸生成的，
                # 尺寸不一致就无法做相关
                normalized = cv2.resize(
                    tile,
                    (SMALL_TRACE_TILE, SMALL_TRACE_TILE),
                    interpolation=cv2.INTER_CUBIC,
                )
                yield {
                    "tile": normalized.astype(np.float32),
                    "position": (x, y),
                    "offset": (offset_x, offset_y),
                    "coverage": round(coverage, 4),
                }


@lru_cache(maxsize=None)
def small_trace_marker_pattern(size: int) -> np.ndarray:
    """生成全系统共用的标记图案，用作"此处有小裁剪水印"的快速门控。

    :param size: 方阵边长，正常为 ``SMALL_TRACE_TILE``。

    种子只跟 ``ROBUST_MAGIC`` 和层版本有关、**与溯源号无关**，所以所有图片
    嵌的是同一个 marker。检测端因此可以先算一次相关性，不匹配就跳过整块，
    省下昂贵的逐位解码——盲扫时要试成千上万个窗口，这道预筛是性能关键。

    图案由两部分叠加：

    * 10 个随机方向、频率 8~20 周期/块的正弦波。频段选在中高频：太低会在
      平坦区域形成肉眼可见的色斑，太高则被 JPEG 量化抹掉；
    * 一层 12×12 的粗随机块，最近邻放大后再高斯模糊（σ=0.8）软化边缘，
      权重 1.8。这部分是低频成分，缩放和重压缩后存活率最高，
      给相关性提供一个稳定的"底"。

    ``lru_cache`` 是必需的：检测时每个窗口都要用它，重算代价太高，
    而结果对同一 ``size`` 完全确定。

    汉宁窗被重映射到 ``[0.5, 1.0]``（``0.50 + 0.50 * 归一化窗``）而不是原始的
    ``[0, 1]``：完全归零会让方块边缘毫无信号，而裁剪恰恰经常只留下边缘部分；
    保留一半权重是"抑制块效应"与"边缘也要能检出"之间的折中。
    """
    rng = np.random.default_rng(ROBUST_MAGIC * 2039 + SMALL_TRACE_VERSION)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.50 + 0.50 * (window / max(float(window.max()), 1e-6))
    pattern = np.zeros((size, size), dtype=np.float32)
    for _ in range(10):
        # fx/fy 为该方向上跨整块的周期数；两者组合决定平面波的方向与波长
        fx = int(rng.integers(8, 20))
        fy = int(rng.integers(8, 20))
        phase = float(rng.random() * np.pi * 2)
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        pattern += sign * np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)
    coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(12, 12))
    pattern += cv2.GaussianBlur(cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST), (0, 0), sigmaX=0.8, sigmaY=0.8) * 1.8
    return normalize_carrier(pattern * window).astype(np.float32)


@lru_cache(maxsize=None)
def small_trace_pattern(trace_id: str, size: int) -> np.ndarray:
    """生成**溯源号专属**的伪随机图案，用于非盲的身份二次确认。

    与 :func:`small_trace_marker_pattern` 的分工：marker 回答"有没有水印"，
    本图案回答"是不是这个溯源号的"。检测端只有先拿到候选溯源号才能重建它，
    这正是非盲检测的含义。

    结构比 marker 更丰富——正弦波（FFT 域特征）+ 余弦基（DCT 域特征）
    + 粗随机块（近似 DWT 低频），三类成分各 8 个/一层。混合的目的是让图案在
    任意一种失真（重压缩、缩放、模糊）下都还剩一部分成分可用，
    不至于被某一种攻击整体抹平。

    种子里带上层名和版本号做域分隔，避免与其他层的图案雷同（见
    :func:`~trace_app.watermark.frequency.layer_seed`）。

    粗随机块的权重 2.6 高于 marker 的 1.8：本图案只在 marker 门控通过后才计算，
    对区分度的要求高于对速度的要求，低频成分多一些能让不同溯源号之间
    的相关性差距拉得更开。
    """
    seed = layer_seed(trace_id, f"small-crop-trace-v{SMALL_TRACE_VERSION}")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.50 + 0.50 * (window / max(float(window.max()), 1e-6))
    pattern = np.zeros((size, size), dtype=np.float32)
    for _ in range(8):
        fx = int(rng.integers(7, 18))
        fy = int(rng.integers(7, 18))
        phase = float(rng.random() * np.pi * 2)
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        pattern += sign * np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)
    # 第二组用 DCT-II 的基函数（+0.5 是半像素偏移，与 DCT 定义一致）。
    # 与正弦波形态不同，能在 JPEG 的 DCT 量化下保留得更整齐。
    for _ in range(8):
        u = int(rng.integers(5, 16))
        v = int(rng.integers(5, 16))
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        pattern += sign * np.cos((xx + 0.5) * np.pi * u / size) * np.cos((yy + 0.5) * np.pi * v / size)
    coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(10, 10))
    pattern += cv2.GaussianBlur(cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST), (0, 0), sigmaX=0.9, sigmaY=0.9) * 2.6
    return normalize_carrier(pattern * window).astype(np.float32)


@lru_cache(maxsize=None)
def small_trace_code_carriers(size: int) -> np.ndarray:
    """生成完整载荷用的 ``ROBUST_BITS``（64）个正交载波，堆成 ``(64, size, size)``。

    :return: 每个切片都是零均值、单位 RMS 的载波，一位比特一个。

    每位比特有独立载波，嵌入时按 ±1 加权求和叠进图里，解码时分别与每个载波
    做内积就能把各位读回来——典型的扩频（spread spectrum）思路。
    载波之间靠不同随机种子近似正交，位数越多串扰越大，但 64 位在 96×96 的
    方块里仍有足够余量。

    每个载波是三种成分的固定配比 ``fft 0.44 + dct 0.36 + dwt 0.20``：
    正弦波扛缩放、余弦基扛 JPEG、粗块扛模糊。占比递减是因为越低频的成分
    越占能量预算、越容易被看见，所以只给它最小的一份。

    种子里的 2657 / 977 / 43691 都是质数，作为三个维度（层、版本、比特序号）
    的乘数可以让种子空间充分散开，避免不同比特生成出相似的载波。

    .. note::
        这里按 ``ROBUST_BITS`` 生成，而消费方 :func:`decode_small_trace_code_scores`
        按 ``CODE_PHYSICAL_BITS`` 分配结果数组。两个常量当前都等于 64，
        一旦被改成不同值就会形状不匹配。此处按现状描述，未作改动。
    """
    carriers = []
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.50 + 0.50 * (window / max(float(window.max()), 1e-6))
    for bit_index in range(ROBUST_BITS):
        # 每位独立播种，而不是从一个 rng 连续取值：这样增删比特数
        # 不会改变其他位的载波，版本兼容性更好
        rng = np.random.default_rng(ROBUST_MAGIC * 2657 + SMALL_TRACE_VERSION * 977 + bit_index * 43691)

        fx = int(rng.integers(6, 16))
        fy = int(rng.integers(6, 16))
        phase = float(rng.random() * np.pi * 2)
        fft = np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)

        u = int(rng.integers(5, 14))
        v = int(rng.integers(5, 14))
        dct = np.cos((xx + 0.5) * np.pi * u / size) * np.cos((yy + 0.5) * np.pi * v / size)

        coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(8, 8))
        dwt = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST)
        dwt = cv2.GaussianBlur(dwt, (0, 0), sigmaX=0.8, sigmaY=0.8)

        carriers.append(normalize_carrier((fft * 0.44 + dct * 0.36 + dwt * 0.20) * window))
    return np.stack(carriers, axis=0).astype(np.float32)


@lru_cache(maxsize=None)
def small_trace_short_carriers(size: int) -> np.ndarray:
    """生成短码用的 ``SMALL_TRACE_SHORT_BITS``（16）个载波。

    与 :func:`small_trace_code_carriers` 是同一套构造，区别只在**数量**：
    16 位比 64 位稀疏得多，载波之间的串扰小一个量级，因此短码通道
    在信噪比很差的小截图上仍能读出来。这正是它在
    :func:`apply_small_crop_trace_layer` 里拿到最高权重（0.42）的原因。

    频段（8~18 / 6~15）比完整载荷的（6~16 / 5~14）整体偏高一点，
    并且用了另一组种子乘数（3251 / 1543 / 104729），
    目的是让短码载波与完整载荷载波之间也尽量不相关——两者叠在同一块图上，
    互相串扰会同时污染两个通道。

    高斯模糊 σ=0.75 略小于载荷载波的 0.80，保留更多细节以换取正交性。
    """
    carriers = []
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.50 + 0.50 * (window / max(float(window.max()), 1e-6))
    for bit_index in range(SMALL_TRACE_SHORT_BITS):
        rng = np.random.default_rng(ROBUST_MAGIC * 3251 + SMALL_TRACE_VERSION * 1543 + bit_index * 104729)
        fx = int(rng.integers(8, 18))
        fy = int(rng.integers(8, 18))
        phase = float(rng.random() * np.pi * 2)
        fft = np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)

        u = int(rng.integers(6, 15))
        v = int(rng.integers(6, 15))
        dct = np.cos((xx + 0.5) * np.pi * u / size) * np.cos((yy + 0.5) * np.pi * v / size)

        coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(8, 8))
        dwt = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST)
        dwt = cv2.GaussianBlur(dwt, (0, 0), sigmaX=0.75, sigmaY=0.75)
        carriers.append(normalize_carrier((fft * 0.44 + dct * 0.36 + dwt * 0.20) * window))
    return np.stack(carriers, axis=0).astype(np.float32)


def apply_small_crop_trace_layer(
    image: Image.Image,
    trace_id: str,
    strength: float = 0.25,
    density: str = "low",
    fidelity_scale: float = 1.0,
    *,
    watermark_bits_from_trace_fn: Callable[[str], list[int]] | None = None,
    watermark_payload_from_trace_fn: Callable[[str], int] | None = None,
) -> Image.Image:
    """在图上密铺小裁剪追踪层，返回处理后的图片。

    :param strength: 0~1 的强度倍率，为 0 时原样返回（等于关闭本层）。
    :param density: ``low`` / ``medium`` / ``high``，决定铺几套偏移。
    :param fidelity_scale: 画质档位传来的额外衰减，用于在"高保真"模式下
        主动降低嵌入强度。
    :param watermark_bits_from_trace_fn: 载荷比特生成函数，可注入以便测试。
    :param watermark_payload_from_trace_fn: 载荷生成函数，同上。

    **单个方块的图案是四种载体的加权和**（权重见代码），四者归一化后叠加，
    再整体归一化成单位 RMS，这样总能量不随权重配比漂移。

    整幅图所有方块用的是**同一个** ``spread_pattern``。这是密铺的核心：
    检测端在任意位置切一块下来，只要位置对得上，看到的就是同一个图案，
    多块还能平均降噪。

    强度是逐块自适应的：纹理丰富的区域藏得住更强的信号，平坦区域必须收敛，
    否则噪点直接可见。
    """
    strength = small_crop_strength_to_scale(strength)
    if strength <= 0:
        return image.convert("RGB")
    density = normalize_small_crop_density(density)
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    # 装不下一个完整方块就放弃：残缺的方块解不出码，白白损失画质
    if height < SMALL_TRACE_TILE or width < SMALL_TRACE_TILE:
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    marker = small_trace_marker_pattern(SMALL_TRACE_TILE)
    trace = small_trace_pattern(trace_id, SMALL_TRACE_TILE)
    bits_fn = watermark_bits_from_trace_fn or _watermark_bits_from_trace
    bits = np.array([1.0 if bit else -1.0 for bit in bits_fn(trace_id)], dtype=np.float32)
    # tensordot 沿比特维求和：把 64 个载波按 ±1 加权叠成一张图，即扩频调制
    code = normalize_carrier(np.tensordot(bits, small_trace_code_carriers(SMALL_TRACE_TILE), axes=([0], [0])))
    short_code = normalize_carrier(
        np.tensordot(
            small_trace_short_bits(
                trace_id,
                watermark_payload_from_trace_fn=watermark_payload_from_trace_fn,
            ),
            small_trace_short_carriers(SMALL_TRACE_TILE),
            axes=([0], [0]),
        )
    )
    # 权重分配的取舍：
    #   short_code 0.42 —— 最容易读回来的通道，优先保障，权重最高；
    #   marker     0.38 —— 检测的第一道门，读不出它后面全都白搭；
    #   code       0.20 —— 64 位摊薄后每位能量本就有限，只作锦上添花；
    #   trace      0.16 —— 仅作二次确认，不需要很强。
    # 四者相加后再归一化，保证总能量与权重绝对值无关，只由 delta 控制。
    spread_pattern = normalize_carrier(marker * 0.38 + trace * 0.16 + code * 0.20 + short_code * 0.42)
    # fidelity_scale 下限 0.15：再低就基本检不出来了，不如不嵌
    delta = SMALL_TRACE_DELTA * strength * max(0.15, min(1.0, fidelity_scale))
    for offset_x, offset_y in small_crop_density_offsets(density):
        for y in range(offset_y, height - SMALL_TRACE_TILE + 1, SMALL_TRACE_TILE):
            for x in range(offset_x, width - SMALL_TRACE_TILE + 1, SMALL_TRACE_TILE):
                tile_rgb = arr[y : y + SMALL_TRACE_TILE, x : x + SMALL_TRACE_TILE, :]
                tile_gray = tile_rgb.mean(axis=2)
                # 灰度标准差作为纹理复杂度的代理：值越大，视觉掩蔽效应越强
                texture = float(tile_gray.std())
                # 42.0 是把标准差映射到 0~1 的经验刻度（约当纹理适中的照片）。
                # 下限 0.24 保证平坦区域（纯色背景、天空）也留有可检出的信号，
                # 上限 0.82 防止高纹理区被加得过猛，在锐利边缘旁形成可见噪点。
                adaptive = min(0.82, max(0.24, texture / 42.0))
                # SMALL_TRACE_CHANNEL_WEIGHTS 压低 R、放大 G/B，
                # 让色偏落在人眼相对不敏感的方向；三通道同时调制则是为了
                # 让解码端能按权重加权合并，把信噪比再提一档
                for channel, weight in enumerate(SMALL_TRACE_CHANNEL_WEIGHTS):
                    tile = arr[y : y + SMALL_TRACE_TILE, x : x + SMALL_TRACE_TILE, channel]
                    arr[y : y + SMALL_TRACE_TILE, x : x + SMALL_TRACE_TILE, channel] = np.clip(
                        tile + spread_pattern * delta * adaptive * weight,
                        0,
                        255,
                    )
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def apply_code_layer(image: Image.Image, trace_id: str, strength_scale: float = 1.0) -> Image.Image:
    """编码层：按 ``CODE_TILE``（160）平铺 marker + trace 图案。

    :param strength_scale: 强度倍率，由画质档位传入。

    与小裁剪层的三点区别：

    1. 方块大得多（160 vs 96）。大方块相关性积分的样本更多、信噪比更高，
       代价是需要更大的残留面积才能检出，抗裁剪能力弱；
    2. **只铺一套、不做偏移交错**。它的目标是抗压缩/抗缩放，不是抗裁剪，
       多套叠加只会平白增加噪点；
    3. **不嵌入任何比特**，只有 marker(0.70) 与 trace(0.30) 两个图案。
       对应地，:func:`detect_watermark_code` 也是靠图案相关性识别，
       走的是非盲路线。

    .. note::
        本模块里的 :func:`decode_code_tile_scores`、:func:`code_scan_grid`、
        :func:`decode_code_tile` 等按比特解码的函数，针对的是这一层的早期
        比特版本；当前的 ``apply_code_layer`` 已不再写入比特，这些函数在
        主检测链路上没有调用方（只经 ``trace_app/compat.py`` 对外导出）。
        此处按现状描述，未作改动。
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    # marker 占七成：编码层的判定几乎全靠 marker 相关性，trace 只用来区分是谁
    spread_pattern = normalize_carrier(code_marker_pattern(CODE_TILE) * 0.70 + code_trace_pattern(trace_id, CODE_TILE) * 0.30)
    for tile_y in range(0, height - CODE_TILE + 1, CODE_TILE):
        for tile_x in range(0, width - CODE_TILE + 1, CODE_TILE):
            tile_rgb = arr[tile_y : tile_y + CODE_TILE, tile_x : tile_x + CODE_TILE, :]
            tile_gray = tile_rgb.mean(axis=2)
            texture = float(tile_gray.std())
            # 与小裁剪层同样的纹理自适应，但区间更保守（0.15~0.72）：
            # 方块大四倍，同等强度下可见面积也大，更容易被察觉
            adaptive = min(0.72, max(0.15, texture / 44.0))
            for channel, weight in enumerate(CODE_CHANNEL_WEIGHTS):
                tile = arr[tile_y : tile_y + CODE_TILE, tile_x : tile_x + CODE_TILE, channel]
                arr[tile_y : tile_y + CODE_TILE, tile_x : tile_x + CODE_TILE, channel] = np.clip(
                    tile + spread_pattern * CODE_DELTA * adaptive * weight * strength_scale,
                    0,
                    255,
                )
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def normalize_carrier(carrier: np.ndarray) -> np.ndarray:
    """把图案规范成零均值、单位 RMS，本模块所有载波的统一出口。

    **去均值**至关重要：残留的直流分量相当于给整块图加了一个恒定亮度偏移，
    肉眼一眼可见（表现为方块状的明暗格子），而且对解码毫无贡献——
    解码端的高通预处理本来就会把它滤掉。

    **单位 RMS** 让"强度"这个概念在各处可比：无论图案由几个正弦叠加而成、
    权重怎么配，归一化后能量都一样，嵌入强度就完全由 ``*_DELTA`` 一个量决定，
    调参时不用担心改配比会连带改变可见度。

    RMS 过小（接近全零的退化图案）时原样返回，避免除零放大出无意义的数值。
    """
    carrier = carrier.astype(np.float32)
    carrier = carrier - carrier.mean()
    rms = float(np.sqrt(np.mean(carrier * carrier)))
    if rms < 1e-6:
        return carrier
    return carrier / rms


@lru_cache(maxsize=None)
def code_cell_carriers(bit_index: int, size: int) -> dict[str, np.ndarray]:
    """为单个比特生成 dct / dwt / fft 三种载波及其加权合成版。

    :return: 含 ``"dct"``、``"dwt"``、``"fft"``、``"combined"`` 四个键的字典。

    与 :func:`code_tile_carriers` 的区别是这里把三种成分**分开返回**，
    调用方可以只用其中一种（例如在只关心抗压缩时单取 dct）。
    频率取得很低（DCT 的 u,v 在 2~5，FFT 的 fx,fy 在 2~6），因为它面向的是
    "格子（cell）"这种远小于整块的单元，跨格子的周期数自然要少。

    .. note::
        主检测/嵌入链路当前没有调用方，仅经 ``trace_app/compat.py`` 对外导出，
        以及测试里用来验证 ``lru_cache`` 生效。此处按现状描述，未作改动。
    """
    rng = np.random.default_rng(ROBUST_MAGIC * 31 + bit_index * 104729)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)

    # 候选中频对，均关于对角线对称成组出现：JPEG 量化表对 (u,v) 与 (v,u)
    # 的处理权重相近，成对可选保证不同比特的抗压缩能力大致均衡
    mid_freq = [(2, 3), (3, 2), (3, 4), (4, 3), (2, 5), (5, 2), (4, 5), (5, 4)]
    u, v = mid_freq[int(rng.integers(0, len(mid_freq)))]
    dct = np.cos((xx + 0.5) * np.pi * u / size) * np.cos((yy + 0.5) * np.pi * v / size)

    # 用两条随机分割线切出四象限的 ±1 棋盘，模拟小波的方向性细节子带。
    # 分割点限制在中间三分之一，避免切出面积悬殊的象限（那会退化成近似常量）
    split_x = int(rng.integers(size // 3, size * 2 // 3))
    split_y = int(rng.integers(size // 3, size * 2 // 3))
    dwt = np.where(xx < split_x, 1.0, -1.0) * np.where(yy < split_y, 1.0, -1.0)

    fx = int(rng.integers(2, 6))
    fy = int(rng.integers(2, 6))
    phase = float(rng.random() * np.pi * 2)
    fft = np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)

    carriers = {
        "dct": normalize_carrier(dct * window),
        "dwt": normalize_carrier(dwt * window),
        "fft": normalize_carrier(fft * window),
    }
    carriers["combined"] = normalize_carrier(carriers["dct"] * 0.45 + carriers["dwt"] * 0.30 + carriers["fft"] * 0.25)
    return carriers


@lru_cache(maxsize=None)
def code_tile_carriers(size: int) -> np.ndarray:
    """编码层的 ``ROBUST_BITS`` 个整块载波，供 :func:`decode_code_tile_scores` 使用。

    构造与 :func:`small_trace_code_carriers` 同源，差别在于窗函数底为 0.55
    而非 0.50（``CODE_TILE`` 更大，边缘占比更小，可以少压一点），
    以及粗块模糊 σ=1.1 更大（大方块上同样的相对尺度需要更宽的核）。

    .. note::
        当前的 :func:`apply_code_layer` 已不写入比特，本函数的消费方
        因此在主链路上不再被触发。此处按现状描述，未作改动。
    """
    carriers = []
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.55 + 0.45 * (window / max(float(window.max()), 1e-6))
    for bit_index in range(ROBUST_BITS):
        rng = np.random.default_rng(ROBUST_MAGIC * 131 + bit_index * 65537)

        fx = int(rng.integers(7, 18))
        fy = int(rng.integers(7, 18))
        phase = float(rng.random() * np.pi * 2)
        fft = np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)

        u = int(rng.integers(5, 15))
        v = int(rng.integers(5, 15))
        dct = np.cos((xx + 0.5) * np.pi * u / size) * np.cos((yy + 0.5) * np.pi * v / size)

        coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(8, 8))
        dwt = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST)
        dwt = cv2.GaussianBlur(dwt, (0, 0), sigmaX=1.1, sigmaY=1.1)

        carrier = normalize_carrier((fft * 0.45 + dct * 0.35 + dwt * 0.20) * window)
        carriers.append(carrier)
    return np.stack(carriers, axis=0).astype(np.float32)


@lru_cache(maxsize=None)
def code_trace_pattern(trace_id: str, size: int) -> np.ndarray:
    """编码层的溯源号专属图案，:func:`detect_watermark_code` 靠它区分是谁。

    构造与 :func:`small_trace_pattern` 平行（10 个正弦 + 10 个余弦基 + 粗块），
    但成分数量和频段都略大，因为 ``CODE_TILE`` = 160 比小裁剪块宽 67%，
    要维持相近的空间频率密度就得配更多的周期。

    粗块权重 3.0 是三个 pattern 里最高的：编码层不做偏移交错、方块数量少，
    单块的区分度必须够硬，低频成分正是压缩后最可靠的那部分。

    种子里的层名 ``trace-code-v4`` 是历史固定字面量，与
    ``CODE_WATERMARK_VERSION`` 无联动——改动它会让所有存量图片解不出来。
    """
    seed = layer_seed(trace_id, "trace-code-v4")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.55 + 0.45 * (window / max(float(window.max()), 1e-6))
    pattern = np.zeros((size, size), dtype=np.float32)

    for _ in range(10):
        fx = int(rng.integers(6, 19))
        fy = int(rng.integers(6, 19))
        phase = float(rng.random() * np.pi * 2)
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        pattern += sign * np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)

    for _ in range(10):
        u = int(rng.integers(5, 17))
        v = int(rng.integers(5, 17))
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        pattern += sign * np.cos((xx + 0.5) * np.pi * u / size) * np.cos((yy + 0.5) * np.pi * v / size)

    coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(12, 12))
    dwt = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST)
    dwt = cv2.GaussianBlur(dwt, (0, 0), sigmaX=1.3, sigmaY=1.3)
    pattern += dwt * 3.0
    return normalize_carrier(pattern * window).astype(np.float32)


@lru_cache(maxsize=None)
def code_marker_pattern(size: int) -> np.ndarray:
    """编码层的公共标记图案，与溯源号无关，作用同 :func:`small_trace_marker_pattern`。

    用了 16 个正弦分量（小裁剪层只有 10 个）和 16×16 的粗块：方块大，
    容纳得下更复杂的图案，而图案越复杂、与自然图像内容偶然相关的概率越低，
    误报率也就越低。

    在 :func:`detect_watermark_code` 里，它同时承担两个角色——
    ``match_trace_signal`` 的门控（相关性 < 0.055 直接否），以及
    :func:`trace_tile_agreement` 的一致性基准（多少比例的方块都能看到它）。
    """
    rng = np.random.default_rng(ROBUST_MAGIC * 1009 + CODE_WATERMARK_VERSION)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    window_1d = np.hanning(size)
    window = np.outer(window_1d, window_1d).astype(np.float32)
    window = 0.55 + 0.45 * (window / max(float(window.max()), 1e-6))
    pattern = np.zeros((size, size), dtype=np.float32)
    for _ in range(16):
        fx = int(rng.integers(8, 22))
        fy = int(rng.integers(8, 22))
        phase = float(rng.random() * np.pi * 2)
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        pattern += sign * np.sin((xx * fx + yy * fy) * np.pi * 2 / size + phase)
    coarse = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(16, 16))
    pattern += cv2.GaussianBlur(cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST), (0, 0), sigmaX=1.0, sigmaY=1.0)
    return normalize_carrier(pattern * window).astype(np.float32)


def decode_code_tile_signal(tile: np.ndarray) -> np.ndarray:
    """从一个 ``CODE_TILE`` 方块里提取水印残差信号。

    :return: 零均值、单位 RMS 的残差图，可直接与 pattern 做点乘求相关。

    **高斯高通**是关键一步：``plane - GaussianBlur(plane)`` 滤掉图像本身的
    低频内容（大块色域、渐变、光照），只留下与水印图案同频段的细节。
    不做这一步，相关性会被图像内容的巨大能量完全淹没。

    σ=2.6 对应的截止频率大致落在 marker/pattern 的主频段之下，
    既压掉了内容，又没削到信号。

    三通道按 ``CODE_CHANNEL_WEIGHTS`` 加权合并——嵌入时就是按这组权重
    分配到各通道的，用同一组权重合并等于做了一次匹配滤波，信噪比最优。
    最后除以权重和还原到单通道量纲。

    尺寸不足时返回全零而非报错：调用方在密集扫描，返回零信号会自然地
    在后续门控处落空。
    """
    if tile.shape[0] < CODE_TILE or tile.shape[1] < CODE_TILE:
        return np.zeros((CODE_TILE, CODE_TILE), dtype=np.float32)
    signal = np.zeros((CODE_TILE, CODE_TILE), dtype=np.float32)
    total_weight = 0.0
    for channel, weight in enumerate(CODE_CHANNEL_WEIGHTS):
        plane = tile[:CODE_TILE, :CODE_TILE, channel].astype(np.float32)
        low = cv2.GaussianBlur(plane, (0, 0), sigmaX=2.6, sigmaY=2.6)
        signal += (plane - low) * weight
        total_weight += weight
    return normalize_carrier(signal / max(total_weight, 1e-6))


def code_scan_signal_grid(arr: np.ndarray, tile_w: int, tile_h: int, offset_x: int, offset_y: int) -> tuple[np.ndarray, float, int]:
    """按给定网格切块并**叠加平均**残差信号。

    :param tile_w: 假设的方块宽度。图片可能被缩放过，真实方块不一定是 160，
        所以尺寸由调用方遍历尝试。
    :param tile_h: 同上，高度可与宽度不同（非等比缩放）。
    :return: ``(平均后的残差图, 各块残差强度的均值, 参与的块数)``。

    叠加平均是这里唯一重要的事：各块的水印信号完全相同（编码层整幅图铺同一
    个图案），而图像内容各不相同、近似独立。求和后信号线性累加、噪声按
    根号累加，N 块能把信噪比提升约 √N 倍。这正是本层能在低强度下仍被检出
    的原因。

    每块先归一化到 ``CODE_TILE`` 再叠加，抹平尺寸差异。

    少于 2 块时返回**未归一化**的 ``summed`` 和强度 0：样本太少，平均没有意义，
    调用方会用块数做门控（``signal_tiles >= 3``），不会使用这个返回值。
    """
    height, width = arr.shape[:2]
    summed = np.zeros((CODE_TILE, CODE_TILE), dtype=np.float32)
    tiles = 0
    strengths = []
    for y in range(offset_y, height - tile_h + 1, tile_h):
        for x in range(offset_x, width - tile_w + 1, tile_w):
            tile = arr[y : y + tile_h, x : x + tile_w, :]
            normalized = cv2.resize(tile, (CODE_TILE, CODE_TILE), interpolation=cv2.INTER_CUBIC)
            signal = decode_code_tile_signal(normalized)
            summed += signal
            strengths.append(float(np.std(signal)))
            tiles += 1
    if tiles < 2:
        return summed, 0.0, tiles
    return normalize_carrier(summed / tiles), float(np.mean(strengths)) if strengths else 0.0, tiles


def trace_tile_agreement(arr: np.ndarray, tile_w: int, tile_h: int, offset_x: int, offset_y: int, pattern: np.ndarray) -> float:
    """统计有多少比例的方块**各自独立地**与给定图案正相关。

    :return: 正相关块数占比，0~1；样本少于 3 块时返回 0.0。

    这是对 :func:`code_scan_signal_grid` 的补充校验，专门用来否掉一类误报：
    叠加平均的分数可能被**个别**强响应的方块拉高（比如图中某处纹理恰好与
    marker 相似），看起来像命中，实则整幅图并没有铺水印。真正的水印是密铺的，
    应当**每一块都能看到**。

    阈值 0.006 只是"略大于零"的判定线，而非强度要求——这里关心的是
    符号的一致性，强度已由别处把关。留一点余量是为了不把纯噪声块
    （相关性在 0 附近抖动）算成正例。

    要求至少 3 块：块数太少时比例本身没有统计意义（1 块命中就是 100%）。
    """
    height, width = arr.shape[:2]
    scores = []
    for y in range(offset_y, height - tile_h + 1, tile_h):
        for x in range(offset_x, width - tile_w + 1, tile_w):
            tile = arr[y : y + tile_h, x : x + tile_w, :]
            normalized = cv2.resize(tile, (CODE_TILE, CODE_TILE), interpolation=cv2.INTER_CUBIC)
            signal = decode_code_tile_signal(normalized)
            scores.append(float((signal * pattern).mean()))
    if len(scores) < 3:
        return 0.0
    positive = sum(1 for score in scores if score > 0.006)
    return positive / len(scores)


def apply_code_layer_shifted(
    image: Image.Image,
    trace_id: str,
    *,
    apply_code_layer_fn: Callable[[Image.Image, str], Image.Image] | None = None,
) -> Image.Image:
    """再叠一层**错开半个方块**的编码层，弥补 :func:`apply_code_layer` 不做偏移的短板。

    :param apply_code_layer_fn: 可注入的嵌入函数，便于测试。

    做法是"先平移、再嵌、再平移回来"：

    1. 把原图整体往左上挪半块（``CODE_TILE // 2``），右下留出黑边；
    2. 在挪过的图上正常铺一遍编码层——此时方块网格相对**原始内容**
       就偏移了半块；
    3. 再挪回来对齐原始坐标，左上角留出黑边。

    这样得到的水印图案与原始网格错开半块，配合原本那层就形成了 2×2 的
    半块交错，效果类似小裁剪层的多偏移密铺。

    最后按 ``0.55 原图 + 0.45 带水印`` 混合，即只施加 45% 的强度——
    两层叠加的总能量必须控制住，否则可见度翻倍。

    .. note::
        ``mask`` 用"三通道之和 > 0"来判定"这里有有效的水印内容"，
        主要为了排除步骤 3 留下的黑边。副作用是原图中**纯黑像素**
        （R=G=B=0）也会被判为无效而跳过混合，那些位置拿不到这一层水印。
        此处按现状描述，未作改动。

    .. note::
        当前嵌入链路没有调用方，仅经 ``trace_app/compat.py`` 对外导出。
    """
    base = image.convert("RGB")
    # 新建的 RGB 图默认全黑，未被 paste 覆盖的右/下半块即黑边
    shifted = Image.new("RGB", base.size)
    shifted.paste(base.crop((CODE_TILE // 2, CODE_TILE // 2, base.width, base.height)), (0, 0))
    apply_fn = apply_code_layer_fn or apply_code_layer
    marked = apply_fn(shifted, trace_id)
    # 平移回原位：restored[Y,X] = base[Y,X] + 水印[Y-半块, X-半块]，
    # 内容与原图逐像素对齐，只有水印网格错开了半块
    restored = Image.new("RGB", base.size)
    restored.paste(marked.crop((0, 0, base.width - CODE_TILE // 2, base.height - CODE_TILE // 2)), (CODE_TILE // 2, CODE_TILE // 2))
    original_arr = np.array(base, dtype=np.float32)
    restored_arr = np.array(restored, dtype=np.float32)
    mask = restored_arr.sum(axis=2, keepdims=True) > 0
    mixed = np.where(mask, original_arr * 0.55 + restored_arr * 0.45, original_arr)
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), "RGB")


def decode_code_tile_scores(tile: np.ndarray) -> np.ndarray:
    """对编码层方块做逐位解扩，返回 ``ROBUST_BITS`` 个带符号的相关分数。

    :return: 每位一个分数，正号读作比特 1、负号读作 0，绝对值即判决余量。

    与 :func:`decode_code_tile_signal` 用同一套高通预处理，区别是这里直接把
    残差与每个载波做内积（``tensordot`` 在两个空间维上求和），得到逐位分数。

    末尾除以 ``CODE_TILE * CODE_TILE`` 把内积化成**均值**，使分数与方块尺寸
    无关，阈值才能写成常量。

    .. note::
        当前的 :func:`apply_code_layer` 不再写入比特，本函数在主检测链路上
        没有调用方。此处按现状描述，未作改动。
    """
    if tile.shape[0] < CODE_TILE or tile.shape[1] < CODE_TILE:
        return np.zeros(ROBUST_BITS, dtype=np.float32)
    carriers = code_tile_carriers(CODE_TILE)
    scores = np.zeros(ROBUST_BITS, dtype=np.float32)
    total_weight = 0.0
    for channel, weight in enumerate(CODE_CHANNEL_WEIGHTS):
        plane = tile[:CODE_TILE, :CODE_TILE, channel].astype(np.float32)
        low = cv2.GaussianBlur(plane, (0, 0), sigmaX=2.6, sigmaY=2.6)
        high = plane - low
        scores += np.tensordot(carriers, high, axes=([1, 2], [0, 1])).astype(np.float32) * weight
        total_weight += weight
    return scores / (max(total_weight, 1e-6) * CODE_TILE * CODE_TILE)


def code_from_score_vector(scores: np.ndarray) -> tuple[int, float]:
    """把带符号的分数向量硬判成整数码，并给出平均判决余量。

    :return: ``(整数码, 平均 |分数|)``。高位在前，与
        :func:`_watermark_bits_from_trace` 的顺序对应。

    余量取绝对值的均值：无论判 0 还是判 1，离零越远说明该位读得越确定。
    这个值在上层被当作"信号强度"参与打分和门控。
    """
    code = 0
    margins = []
    for score in scores:
        code = (code << 1) | (1 if score > 0 else 0)
        margins.append(abs(float(score)))
    return code, float(np.mean(margins)) if margins else 0.0


def decode_code_tile(tile: np.ndarray) -> tuple[int, float]:
    """解扩 + 硬判的组合快捷方式，用于单块解码。

    .. note::
        主链路无调用方，仅经 ``trace_app/compat.py`` 对外导出。
    """
    scores = decode_code_tile_scores(tile)
    return code_from_score_vector(scores)


def iter_code_scan_offsets(width: int, height: int, tile_w: int, tile_h: int):
    """枚举编码层网格扫描的起始偏移。

    :return: 生成器，逐个产出 ``(offset_x, offset_y)``。

    检测端不知道嵌入时的网格从哪里开始（图片可能被裁过），必须试各种相位。
    偏移只需要覆盖 ``[0, tile)`` 一个周期——再往后就与某个更小的偏移等价了。

    步长取块尺寸的 1/4，即每个方向试 4 个相位。这是"对不准就检不出"与
    "位置越多越慢"之间的折中：半块以内的错位仍有相当部分能量能对上，
    1/4 块的粒度足够。下限 16 像素防止小方块下步长退化成 1、
    把扫描次数炸开。

    ``max_x`` / ``max_y`` 非正说明图比方块还小，直接返回空生成器。
    """
    max_x = width - tile_w + 1
    max_y = height - tile_h + 1
    if max_x <= 0 or max_y <= 0:
        return
    step_x = max(16, tile_w // 4)
    step_y = max(16, tile_h // 4)
    # 上界取 min(tile, max)：既不超过一个相位周期，也不超出图片可放置范围
    for offset_y in range(0, min(tile_h, max_y), step_y):
        for offset_x in range(0, min(tile_w, max_x), step_x):
            yield offset_x, offset_y


def code_scan_grid(arr: np.ndarray, tile_w: int, tile_h: int, offset_x: int, offset_y: int) -> tuple[int, float, int]:
    """按网格切块、把逐位分数叠加平均后硬判出一个码。

    :return: ``(整数码, 强度 × 一致性, 参与块数)``。块数少于 2 时返回 ``(0, 0.0, tiles)``。

    与 :func:`code_scan_signal_grid` 的区别：那个在**图像域**叠加残差，
    这个在**分数域**叠加。分数域叠加更省事，但丢掉了空间信息，
    没法再做图案相关性校验。

    ``agreement`` 是各位分数绝对值相对最大值的均值，衡量"各位读得是否同样确定"。
    真实水印各位强度相近，这个值接近 1；若只是某几位偶然很大、其余接近零
    （典型的噪声形态），值就会很低。乘进强度里等于对这类结果打折。

    .. note::
        主链路无调用方，仅经 ``trace_app/compat.py`` 对外导出。
    """
    height, width = arr.shape[:2]
    summed = np.zeros(ROBUST_BITS, dtype=np.float32)
    tiles = 0
    for y in range(offset_y, height - tile_h + 1, tile_h):
        for x in range(offset_x, width - tile_w + 1, tile_w):
            tile = arr[y : y + tile_h, x : x + tile_w, :]
            normalized = cv2.resize(tile, (CODE_TILE, CODE_TILE), interpolation=cv2.INTER_CUBIC)
            summed += decode_code_tile_scores(normalized)
            tiles += 1
    if tiles < 2:
        return 0, 0.0, tiles
    averaged = summed / max(1, tiles)
    code, strength = code_from_score_vector(averaged)
    agreement = float(np.mean(np.abs(summed) / (np.abs(summed).max() + 1e-6)))
    return code, strength * agreement, tiles


def decode_small_trace_signal(tile: np.ndarray) -> np.ndarray:
    """从小裁剪方块里提取残差信号，供与 marker / trace 图案做相关。

    与 :func:`decode_code_tile_signal` 同构，唯一的实质差别是高通的
    σ=1.8 而非 2.6：``SMALL_TRACE_TILE``（96）比 ``CODE_TILE``（160）小
    约四成，图案的空间频率相应更高，滤波核也要按比例收窄，
    否则会把信号一起滤掉。
    """
    if tile.shape[0] < SMALL_TRACE_TILE or tile.shape[1] < SMALL_TRACE_TILE:
        return np.zeros((SMALL_TRACE_TILE, SMALL_TRACE_TILE), dtype=np.float32)
    signal = np.zeros((SMALL_TRACE_TILE, SMALL_TRACE_TILE), dtype=np.float32)
    total_weight = 0.0
    for channel, weight in enumerate(SMALL_TRACE_CHANNEL_WEIGHTS):
        plane = tile[:SMALL_TRACE_TILE, :SMALL_TRACE_TILE, channel].astype(np.float32)
        low = cv2.GaussianBlur(plane, (0, 0), sigmaX=1.8, sigmaY=1.8)
        signal += (plane - low) * weight
        total_weight += weight
    return normalize_carrier(signal / max(total_weight, 1e-6))


def decode_small_trace_code_scores(tile: np.ndarray) -> np.ndarray:
    """解扩出完整载荷的 ``CODE_PHYSICAL_BITS``（64）位分数。

    :return: 每位一个带符号分数，交给 :func:`code_from_score_vector` 硬判。

    高通去内容 → 与每个载波做内积 → 按通道权重加权合并 → 除以像素数归一化，
    与 :func:`decode_code_tile_scores` 是同一套流程，只是换用小裁剪层的
    载波、方块尺寸和 σ。

    这是**盲解**通道：只要信噪比够，不需要候选集也能读出码，
    再由 :func:`match_small_trace_code` 用魔数和校验和验真。
    """
    if tile.shape[0] < SMALL_TRACE_TILE or tile.shape[1] < SMALL_TRACE_TILE:
        return np.zeros(CODE_PHYSICAL_BITS, dtype=np.float32)
    carriers = small_trace_code_carriers(SMALL_TRACE_TILE)
    scores = np.zeros(CODE_PHYSICAL_BITS, dtype=np.float32)
    total_weight = 0.0
    for channel, weight in enumerate(SMALL_TRACE_CHANNEL_WEIGHTS):
        plane = tile[:SMALL_TRACE_TILE, :SMALL_TRACE_TILE, channel].astype(np.float32)
        low = cv2.GaussianBlur(plane, (0, 0), sigmaX=1.8, sigmaY=1.8)
        high = plane - low
        scores += np.tensordot(carriers, high, axes=([1, 2], [0, 1])).astype(np.float32) * weight
        total_weight += weight
    return scores / (max(total_weight, 1e-6) * SMALL_TRACE_TILE * SMALL_TRACE_TILE)


def decode_small_trace_short_scores(tile: np.ndarray) -> np.ndarray:
    """解扩出 ``SMALL_TRACE_SHORT_BITS``（16）位短码分数。

    流程与 :func:`decode_small_trace_code_scores` 完全一致，只换了载波。
    只有 16 位、且嵌入时权重最高（0.42），所以在完整载荷已经读不出来的
    弱信号场景下，这条通道往往还能给出可用的结果——这正是
    :func:`detect_small_crop_trace` 把它当作 fallback 的原因。
    """
    if tile.shape[0] < SMALL_TRACE_TILE or tile.shape[1] < SMALL_TRACE_TILE:
        return np.zeros(SMALL_TRACE_SHORT_BITS, dtype=np.float32)
    carriers = small_trace_short_carriers(SMALL_TRACE_TILE)
    scores = np.zeros(SMALL_TRACE_SHORT_BITS, dtype=np.float32)
    total_weight = 0.0
    for channel, weight in enumerate(SMALL_TRACE_CHANNEL_WEIGHTS):
        plane = tile[:SMALL_TRACE_TILE, :SMALL_TRACE_TILE, channel].astype(np.float32)
        low = cv2.GaussianBlur(plane, (0, 0), sigmaX=1.8, sigmaY=1.8)
        high = plane - low
        scores += np.tensordot(carriers, high, axes=([1, 2], [0, 1])).astype(np.float32) * weight
        total_weight += weight
    return scores / (max(total_weight, 1e-6) * SMALL_TRACE_TILE * SMALL_TRACE_TILE)


def short_code_from_scores(scores: np.ndarray) -> tuple[int, float]:
    """短码的硬判决，逻辑与 :func:`code_from_score_vector` 完全相同。

    单独留一个函数是为了让短码通道与完整载荷通道各自独立演进
    （例如将来给短码加软判决），互不牵连。
    """
    code = 0
    margins = []
    for score in scores:
        code = (code << 1) | (1 if score > 0 else 0)
        margins.append(abs(float(score)))
    return code, float(np.mean(margins)) if margins else 0.0


def record_from_short_code_match(
    short_code: int,
    code_records: list[tuple[str, int, dict[str, Any]]],
    max_errors: int,
    min_gap: int,
    *,
    hamming_distance_fn: Callable[[int, int], int] | None = None,
    watermark_payload_from_trace_fn: Callable[[str], int] | None = None,
) -> dict[str, Any] | None:
    """用解出的短码在候选中查记录，要求命中足够"独一无二"。

    :param max_errors: 允许的最大汉明距离。
    :param min_gap: 最优与次优的距离差下限。
    :return: 命中的记录；无匹配或存在歧义时 ``None``。

    ``min_gap`` 是这里的关键。短码位数少，随机噪声撞上某条记录并不稀奇；
    但如果它同时跟**两条**记录都差不多近，就说明这次匹配不具区分度——
    宁可判失败，也不能在两条记录里瞎猜一个，那等于随机指认责任人。
    """
    distance_fn = hamming_distance_fn or _hamming_distance
    best_record = None
    best_distance = SMALL_TRACE_SHORT_BITS + 1
    second_distance = SMALL_TRACE_SHORT_BITS + 1
    for trace_id, _, record in code_records:
        distance = distance_fn(
            short_code,
            small_trace_short_code(
                trace_id,
                watermark_payload_from_trace_fn=watermark_payload_from_trace_fn,
            ),
        )
        if distance < best_distance:
            second_distance = best_distance
            best_record = record
            best_distance = distance
        elif distance < second_distance:
            second_distance = distance
    if best_record and best_distance <= max_errors and second_distance - best_distance >= min_gap:
        return best_record
    return None


def match_small_trace_code(
    code: int,
    code_records: list[tuple[str, int, dict[str, Any]]],
    max_errors: int = 10,
    *,
    recover_payload_from_code_fn: Callable[[int], tuple[int, int]] | None = None,
    hamming_distance_fn: Callable[[int, int], int] | None = None,
    code_crc16_fn: Callable[[int], int] | None = None,
) -> tuple[dict[str, Any] | None, int, int]:
    """把读出的完整码匹配到候选记录。

    :return: ``(命中记录或 None, 最优距离, 次优距离)``。后两个值即使失败也返回，
        便于调用方评估"差了多少"以及这次匹配有多接近歧义。

    三道校验层层收紧、由廉价到昂贵：先用魔数（容错 3 比特）快速否掉无关比特串，
    再验 CRC，最后才逐条比对候选。前两道不通过就直接返回哨兵距离
    ``CODE_PAYLOAD_BITS + 1``（即"比最大可能距离还大"），
    调用方据此判定为未命中。
    """
    recover_fn = recover_payload_from_code_fn or _recover_payload_from_code
    distance_fn = hamming_distance_fn or _hamming_distance
    crc_fn = code_crc16_fn or _code_crc16
    payload, corrections = recover_fn(code)
    magic_distance = distance_fn(payload >> 32, ROBUST_MAGIC)
    if magic_distance > 3:
        return None, CODE_PAYLOAD_BITS + 1, CODE_PAYLOAD_BITS + 1
    body_and_magic = payload >> 16
    checksum = payload & 0xFFFF
    crc_distance = distance_fn(checksum, crc_fn(body_and_magic))
    if crc_distance > 12:
        return None, CODE_PAYLOAD_BITS + 1, CODE_PAYLOAD_BITS + 1
    best_record = None
    best_distance = CODE_PAYLOAD_BITS + 1
    second_distance = CODE_PAYLOAD_BITS + 1
    for _, expected_payload, record in code_records:
        distance = distance_fn(payload, expected_payload)
        if distance < best_distance:
            second_distance = best_distance
            best_record = record
            best_distance = distance
        elif distance < second_distance:
            second_distance = distance
    total_distance = best_distance + magic_distance + crc_distance + corrections
    if best_record and best_distance <= max_errors and total_distance <= 28 and second_distance - best_distance >= 3:
        return best_record, best_distance, second_distance
    return None, best_distance, second_distance


def iter_small_trace_windows(width: int, height: int):
    """枚举待扫描的候选窗口 ``(x, y, 宽, 高)``。

    截图的尺寸和长宽比完全不可控，所以这里做**多尺度 + 多长宽比**的穷举：
    8 档宽度 × 3 种长宽比，共 24 种窗口形状；高度对齐到 8 的倍数，
    与嵌入时的网格粒度保持一致。

    步长取窗口边长的一半（下限 20 像素），让相邻窗口**重叠**——
    截图的裁剪边界是任意的，不重叠就很可能每个窗口都恰好跨在两个码之间、
    一个完整的码都框不住。

    这是本模块最昂贵的一步，代价随图片面积和形状数量增长，
    调用方需自行控制何时启用（见检测流水线中的规模开关）。
    """
    tile_shapes = []
    for tile_w in (40, 48, 56, 64, 80, 96, 112, 128):
        for aspect in (0.85, 1.0, 1.15):
            tile_h = int(round(tile_w * aspect / 8)) * 8
            shape = (tile_w, max(40, tile_h))
            if shape not in tile_shapes:
                tile_shapes.append(shape)
    for tile_w, tile_h in tile_shapes:
        if width < tile_w or height < tile_h:
            continue
        step_x = max(20, tile_w // 2)
        step_y = max(20, tile_h // 2)
        for y in range(0, height - tile_h + 1, step_y):
            for x in range(0, width - tile_w + 1, step_x):
                yield x, y, tile_w, tile_h


def detect_small_crop_trace(
    image: Image.Image,
    records: list[dict[str, Any]],
    generated_trace_ids: list[str],
    *,
    watermark_payload_from_trace: Callable[[str], int],
    record_visual_consistency: Callable[[Image.Image, dict[str, Any]], tuple[bool, int, float, float]],
    recover_payload_from_code: Callable[[int], tuple[int, int]],
    hamming_distance: Callable[[int, int], int],
    code_crc16: Callable[[int], int],
    now_text: Callable[[], str],
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
) -> dict[str, Any] | None:
    """小裁剪追踪的检测入口：从一小块截图里反查溯源号。

    :param generated_trace_ids: 近期生成的溯源号，参与候选优先级排序。
    :param record_visual_consistency: 视觉一致性复核回调，返回
        ``(是否一致, 内点数, 比率, 覆盖率)``。
    :return: 命中返回证据字典，否则 ``None``。

    流程：筛出启用了编码层且版本匹配的候选 → 多尺度滑窗扫描 →
    每个窗口解码短码并匹配 → 对胜出者做视觉一致性复核。

    最后那道视觉复核是必要的：短码位数少、又是在大量窗口上反复尝试，
    纯靠码匹配的误报率不可接受。要求图像本身也确实相似，才能定论。
    """
    arr0 = np.array(image.convert("RGB"), dtype=np.float32)
    raw_records = [
        (record.get("trace_id"), watermark_payload_from_trace(record.get("trace_id")), record)
        for record in records
        if record.get("trace_id")
        and record.get("robust_watermark")
        and record.get("watermark_code_version") == CODE_WATERMARK_VERSION
        and record.get("small_crop_trace_enabled")
        and record.get("small_crop_trace_version") == SMALL_TRACE_VERSION
    ]
    persistent_candidate_mode = False
    if generated_trace_ids:
        order = {trace_id: index for index, trace_id in enumerate(generated_trace_ids)}
        candidate_records = [item for item in raw_records if item[0] in order]
        candidate_records.sort(key=lambda item: order[item[0]])
        candidate_records = candidate_records[: min(len(candidate_records), 8)]
    else:
        candidate_records = raw_records[: min(len(raw_records), 80)]
        persistent_candidate_mode = True
    if not candidate_records:
        return None

    visual_evidence: dict[str, tuple[int, float, float]] = {}
    verified_candidates = []
    for trace_id, payload, record in candidate_records:
        consistent, inliers, ratio, residual_score = record_visual_consistency(image, record)
        if not consistent:
            continue
        verified_candidates.append((trace_id, payload, record))
        visual_evidence[trace_id] = (inliers, ratio, residual_score)
    candidate_records = verified_candidates
    if not candidate_records:
        return None

    marker = small_trace_marker_pattern(SMALL_TRACE_TILE)
    scales = (1.0, 0.92, 1.08)
    best_trace = None
    best_votes = 0
    best_strength = 0.0
    vote_counts: dict[str, int] = {}
    hit_counts: dict[str, int] = {}
    strength_counts: dict[str, float] = {}
    for scale in scales:
        if scale == 1.0:
            arr = arr0
        else:
            arr = cv2.resize(arr0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        height, width = arr.shape[:2]
        if height < 120 or width < 120:
            continue
        for x, y, tile_w, tile_h in iter_small_trace_windows(width, height):
            tile = arr[y : y + tile_h, x : x + tile_w, :]
            normalized = cv2.resize(tile, (SMALL_TRACE_TILE, SMALL_TRACE_TILE), interpolation=cv2.INTER_CUBIC)
            signal = decode_small_trace_signal(normalized)
            marker_score = float((signal * marker).mean())
            if marker_score < 0.052:
                continue
            scores = decode_small_trace_code_scores(normalized)
            code, code_strength = code_from_score_vector(scores)
            best_record, _, _ = match_small_trace_code(
                code,
                candidate_records,
                recover_payload_from_code_fn=recover_payload_from_code,
                hamming_distance_fn=hamming_distance,
                code_crc16_fn=code_crc16,
            )
            short_strength = 0.0
            matched_by_short_code = False
            if not best_record:
                short_code, short_strength = short_code_from_scores(decode_small_trace_short_scores(normalized))
                best_record = record_from_short_code_match(
                    short_code,
                    candidate_records,
                    4 if not persistent_candidate_mode else 2,
                    2 if not persistent_candidate_mode else 3,
                    hamming_distance_fn=hamming_distance,
                    watermark_payload_from_trace_fn=watermark_payload_from_trace,
                )
                matched_by_short_code = bool(best_record)
            if not best_record:
                continue
            trace_id = best_record.get("trace_id")
            trace_score = float((signal * small_trace_pattern(trace_id, SMALL_TRACE_TILE)).mean())
            score = marker_score * 0.40 + max(code_strength, short_strength) * 2.8 + max(0.0, trace_score) * 0.24
            if matched_by_short_code and trace_score < 0.034:
                continue
            if (
                marker_score < 0.060
                or max(code_strength, short_strength) < 0.005
                or trace_score < 0.020
                or score < 0.060
            ):
                continue
            weighted_vote = int(max(1, score * 10000) * max(1, min(5, int(round(tile_w * tile_h / (96 * 96))))))
            vote_counts[trace_id] = vote_counts.get(trace_id, 0) + weighted_vote
            hit_counts[trace_id] = hit_counts.get(trace_id, 0) + 1
            strength_counts[trace_id] = strength_counts.get(trace_id, 0.0) + score * weighted_vote
    for trace_id, votes in vote_counts.items():
        hits = hit_counts.get(trace_id, 0)
        min_hits = 5 if not persistent_candidate_mode else 10
        min_votes = 24000 if not persistent_candidate_mode else 36000
        if hits < min_hits or votes < min_votes:
            continue
        avg_strength = strength_counts[trace_id] / max(1, votes)
        if votes > best_votes or (votes == best_votes and avg_strength > best_strength):
            best_trace = trace_id
            best_votes = votes
            best_strength = avg_strength
    min_final_votes = 24000 if not persistent_candidate_mode else 36000
    if not best_trace or best_votes < min_final_votes:
        return None
    record = next((item for trace_id, _, item in candidate_records if trace_id == best_trace), None)
    if not record:
        return None
    evidence = visual_evidence.get(best_trace)
    if not evidence:
        return None
    visual_inliers, visual_ratio, residual_score = evidence
    return with_evidence_fields({
        "id": record.get("id"),
        "trace_id": best_trace,
        "user_id": record.get("user_id"),
        "mode": "watermark_code",
        "mode_label": "小面积截图频域水印码",
        "created_at": record.get("created_at"),
        "confidence": int(min(96, max(78, 72 + best_votes / 1200))),
        "phash_match": False,
        "status": "小截图水印码恢复",
        "extracted_at": now_text(),
        "watermark_layers": record.get("watermark_layers", WATERMARK_LAYERS),
        "layer_scores": {
            "dct": round(best_strength, 4),
            "dwt": round(best_strength, 4),
            "fft": round(best_strength, 4),
        },
        "code_recovery": {
            "method": "small_crop_trace_redundancy",
            "version": SMALL_TRACE_VERSION,
            "votes": best_votes,
            "strength": round(best_strength, 4),
            "visual_inliers": visual_inliers,
            "visual_ratio": round(visual_ratio, 3),
            "residual_score": round(residual_score, 4),
        },
    }, record)


def detect_watermark_code(
    image: Image.Image,
    records: list[dict[str, Any]],
    generated_trace_ids: list[str],
    *,
    watermark_payload_from_trace: Callable[[str], int],
    record_visual_consistency: Callable[[Image.Image, dict[str, Any]], tuple[bool, int, float, float]],
    recover_payload_from_code: Callable[[int], tuple[int, int]],
    hamming_distance: Callable[[int, int], int],
    code_crc16: Callable[[int], int],
    now_text: Callable[[], str],
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
) -> dict[str, Any] | None:
    """编码层的检测入口：按固定网格读取空域短码。

    :return: 命中返回证据字典，否则 ``None``。

    与 :func:`detect_small_crop_trace` 是**姊妹检测器**，区别在假设不同：
    本函数假定图片基本保持原尺寸，只按 ``CODE_TILE`` 网格对齐扫描，
    另外试 ±5% 的缩放容差；小裁剪那边则不假设网格位置，做多尺度滑窗穷举。
    因此本函数快得多，在检测流水线中排在前面。
    """
    arr0 = np.array(image.convert("RGB"), dtype=np.float32)
    # 三档缩放容差：原尺寸优先，再试 ±5% 覆盖轻微重采样带来的偏差
    scales = (1.0, 0.95, 1.05)
    best_trace = None
    best_strength = 0.0
    best_votes = 0
    raw_code_records = [
        (record.get("trace_id"), watermark_payload_from_trace(record.get("trace_id")), record)
        for record in records
        if record.get("trace_id")
        and record.get("robust_watermark")
        and record.get("watermark_code_version") == CODE_WATERMARK_VERSION
    ]
    persistent_candidate_mode = False
    if generated_trace_ids:
        order = {trace_id: index for index, trace_id in enumerate(generated_trace_ids)}
        code_records = [item for item in raw_code_records if item[0] in order]
        code_records.sort(key=lambda item: order[item[0]])
        code_records = code_records[: min(len(code_records), 8)]
    else:
        code_records = raw_code_records[: min(len(raw_code_records), 100)]
        persistent_candidate_mode = True
    if not code_records:
        return None

    visual_evidence: dict[str, tuple[int, float, float]] = {}
    verified_code_records = []
    for trace_id, payload, record in code_records:
        consistent, inliers, ratio, residual_score = record_visual_consistency(image, record)
        if not consistent:
            continue
        verified_code_records.append((trace_id, payload, record))
        visual_evidence[trace_id] = (inliers, ratio, residual_score)
    code_records = verified_code_records
    if not code_records:
        return None

    def match_trace_code(code: int, max_errors: int = 7) -> tuple[str | None, int, int]:
        """把读出的码匹配到溯源号，返回 ``(溯源号或 None, 最优距离, 次优距离)``。

        与模块级的 :func:`match_small_trace_code` 同构，但阈值更严
        （魔数只容错 2 比特、``max_errors`` 为 7）：本检测器假定图片未被
        大幅改动，信号本就该更干净，因此可以收紧，换取更低的误报率。

        写成闭包是为了直接捕获外层的 ``code_records`` 与各注入回调。
        """
        payload, corrections = recover_payload_from_code(code)
        magic_distance = hamming_distance(payload >> 32, ROBUST_MAGIC)
        if magic_distance > 2:
            return None, CODE_PAYLOAD_BITS + 1, CODE_PAYLOAD_BITS + 1
        body_and_magic = payload >> 16
        checksum = payload & 0xFFFF
        crc_distance = hamming_distance(checksum, code_crc16(body_and_magic))
        if crc_distance > 12:
            return None, CODE_PAYLOAD_BITS + 1, CODE_PAYLOAD_BITS + 1
        best = None
        best_distance = CODE_PAYLOAD_BITS + 1
        second_distance = CODE_PAYLOAD_BITS + 1
        for trace_id, expected_code, _ in code_records:
            distance = hamming_distance(payload, expected_code)
            if distance < best_distance:
                second_distance = best_distance
                best = trace_id
                best_distance = distance
            elif distance < second_distance:
                second_distance = distance
        total_distance = best_distance + magic_distance + crc_distance + corrections
        if best and best_distance <= max_errors and total_distance <= 24 and second_distance - best_distance >= 4:
            return best, best_distance, second_distance
        return None, best_distance, second_distance

    def match_trace_signal(signal: np.ndarray) -> tuple[str | None, float, float]:
        """相关性匹配：不解码，直接拿信号与各候选的图案比相似度。

        :return: ``(最佳溯源号或 None, 最佳得分, 次佳得分)``。

        这是解码失败时的兜底路径。先用**标记图案**探一下这块区域到底有没有
        编码层信号（得分低于 0.055 就直接放弃，省下逐候选比对的开销），
        再对每个候选算一次相关。

        同时返回次佳得分，是为了让调用方能判断领先幅度——
        与短码匹配里的 ``min_gap`` 同一个道理：最佳和次佳咬得太紧就是歧义，
        不能据此定论。
        """
        marker_score = float((signal * code_marker_pattern(CODE_TILE)).mean())
        if marker_score < 0.055:
            return None, marker_score, -1.0
        best_trace = None
        best_score = -1.0
        second_score = -1.0
        for trace_id, _, _ in code_records:
            pattern = code_trace_pattern(trace_id, CODE_TILE)
            score = float((signal * pattern).mean())
            if score > best_score:
                second_score = best_score
                best_trace = trace_id
                best_score = score
            elif score > second_score:
                second_score = score
        return best_trace, marker_score * 0.7 + best_score * 0.3, second_score

    for scale in scales:
        if scale == 1.0:
            arr = arr0
        else:
            arr = cv2.resize(arr0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        height, width = arr.shape[:2]
        if height < 120 or width < 120:
            continue
        vote_counts: dict[str, int] = {}
        hit_counts: dict[str, int] = {}
        strength_counts: dict[str, float] = {}
        distance_counts: dict[str, int] = {}
        tile_shapes = []
        for tile_w in (144, 152, 160, 168, 176):
            for aspect in (0.94, 1.0, 1.06):
                tile_h = int(round(tile_w * aspect / 8)) * 8
                shape = (tile_w, max(48, tile_h))
                if shape not in tile_shapes:
                    tile_shapes.append(shape)
        for tile_w, tile_h in tile_shapes:
            if height < tile_h or width < tile_w:
                continue
            for offset_x, offset_y in iter_code_scan_offsets(width, height, tile_w, tile_h):
                signal, signal_strength, signal_tiles = code_scan_signal_grid(arr, tile_w, tile_h, offset_x, offset_y)
                if signal_tiles >= 3:
                    signal_trace, score, second_score = match_trace_signal(signal)
                    if signal_trace and score >= 0.05 and (score - second_score) >= 0.003:
                        agreement = trace_tile_agreement(
                            arr,
                            tile_w,
                            tile_h,
                            offset_x,
                            offset_y,
                            code_marker_pattern(CODE_TILE),
                        )
                        if agreement < 0.40:
                            continue
                        weighted_vote = int(max(1, score * 10000) * min(signal_tiles, 16))
                        vote_counts[signal_trace] = vote_counts.get(signal_trace, 0) + weighted_vote
                        hit_counts[signal_trace] = hit_counts.get(signal_trace, 0) + 1
                        strength_counts[signal_trace] = strength_counts.get(signal_trace, 0.0) + score * weighted_vote
                        distance_counts[signal_trace] = distance_counts.get(signal_trace, 0) + 0
        for trace_id, votes in vote_counts.items():
            min_hits = 2 if persistent_candidate_mode else 3
            min_votes = 30000 if persistent_candidate_mode else 30000
            strong_single_hit = persistent_candidate_mode and hit_counts.get(trace_id, 0) >= 1 and votes >= 23000
            if hit_counts.get(trace_id, 0) < min_hits and not strong_single_hit:
                continue
            if votes < min_votes and not strong_single_hit:
                continue
            avg_distance = distance_counts[trace_id] / max(1, hit_counts[trace_id])
            if avg_distance > 7:
                continue
            avg_strength = strength_counts[trace_id] / max(1, votes)
            if votes > best_votes or (votes == best_votes and avg_strength > best_strength):
                best_trace = trace_id
                best_votes = votes
                best_strength = avg_strength
    min_final_votes = 23000 if persistent_candidate_mode else 30000
    if not best_trace or best_votes < min_final_votes:
        return None
    record = next((item for trace_id, _, item in code_records if trace_id == best_trace), None)
    if not record:
        return None
    evidence = visual_evidence.get(best_trace)
    if not evidence:
        return None
    visual_inliers, visual_ratio, residual_score = evidence
    confidence = int(min(98, max(75, 70 + best_votes * 8)))
    return with_evidence_fields({
        "id": record.get("id"),
        "trace_id": best_trace,
        "user_id": record.get("user_id"),
        "mode": "watermark_code",
        "mode_label": "多尺度频域水印码",
        "created_at": record.get("created_at"),
        "confidence": confidence,
        "phash_match": False,
        "status": "水印码恢复",
        "extracted_at": now_text(),
        "watermark_layers": record.get("watermark_layers", WATERMARK_LAYERS),
        "layer_scores": {
            "dct": round(best_strength, 4),
            "dwt": round(best_strength, 4),
            "fft": round(best_strength, 4),
        },
        "code_recovery": {
            "method": "multi_scale_dct_dwt_fft",
            "version": CODE_WATERMARK_VERSION,
            "votes": best_votes,
            "strength": round(best_strength, 4),
            "visual_inliers": visual_inliers,
            "visual_ratio": round(visual_ratio, 3),
            "residual_score": round(residual_score, 4),
        },
    }, record)
