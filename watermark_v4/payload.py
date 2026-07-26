"""V4 净荷：认证标签生成、RS 纠错编解码、比特置换。

这是 V4 "写什么、怎么纠错、怎么打散"的定义所在，与图像处理完全解耦——
本模块只跟字节和比特打交道，因此可以独立做穷尽测试。

**数据流（嵌入方向）**::

    trace_id ──HMAC-SHA256──> 4 字节认证标签（净荷）
             ──RS(8,4)────> 8 字节码字（4 数据 + 4 校验）
             ──bytes_to_bits─> 64 个逻辑比特
             ──phase 置换──> 64 个物理比特（写入某个分块）

提取方向逆序而行，并在 RS 解码这一步利用"擦除"机制额外挽救错误。

**为什么只嵌 4 字节？** 分块容量有限（64 比特/块），必须在"能装下"和
"够安全"之间取舍。4 字节认证标签配合候选集限制（最多 3 个候选），
误匹配概率约为 3/2³²，见 :func:`candidate_match_probability`。

**为什么要认证而不是直接嵌溯源号？** 认证标签由服务端密钥算出，
攻击者即使完全掌握本文件的算法，没有密钥也构造不出能通过校验的水印。
检测端因此能可靠区分真水印和巧合噪声。
"""

import hashlib
import hmac
import random
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite

from reedsolo import RSCodec, ReedSolomonError


# 认证标签长度：取 HMAC 摘要的前 4 字节（32 位）
AUTH_TAG_BYTES = 4
# 密钥长度下限，低于此值拒绝工作——弱密钥会让整个认证机制失去意义
AUTH_KEY_MIN_BYTES = 32
# HMAC 消息前缀（域分隔符）：保证同一把密钥在别处的用法不会与这里产生
# 可互换的摘要，避免跨用途的重放攻击
AUTH_MESSAGE_PREFIX = b"robust-v4:"

# RS(8,4)：4 字节数据 + 4 字节校验 = 8 字节码字。
# 该配置最多能纠正 2 个完全未知的错误字节；若能指出错误位置（擦除），
# 则最多可纠正 4 个——这正是解码时逐级尝试擦除的价值所在。
RS_DATA_BYTES = 4
RS_PARITY_BYTES = 4
RS_CODEWORD_BYTES = 8
# 解码时依次尝试的擦除数量：从 0（不擦除）逐级放宽到 4（纠错能力上限）
RS_ERASURE_COUNTS = (0, 1, 2, 3, 4)

# 4 个相位：相邻分块使用不同的比特置换，见 phase_for_tile
PHASE_COUNT = 4
# 码字比特数：8 字节 × 8 位
CODEWORD_BITS = 64
# 置换随机种子的前缀，同样起域分隔作用；改动它会导致所有旧图无法解码
PHASE_PERMUTATION_PREFIX = (
    b"hmac32_rs_8_4_full_repeat_sync_v4:carrier-permutation:"
)

# 全模块共用一个编解码器实例：RSCodec 构造时要建伽罗华域查找表，开销不小，
# 且它在编解码过程中无可变状态，可安全复用。
_RS_CODEC = RSCodec(RS_PARITY_BYTES, nsize=RS_CODEWORD_BYTES)


@dataclass(frozen=True, slots=True)
class CandidateDecode:
    """一次成功的候选解码结果，附带用于评估可信度的质量指标。"""

    # 纠错后还原出的 4 字节净荷（即认证标签）
    payload: bytes
    # 纠错后的完整 8 字节码字
    corrected_codeword: bytes
    # RS 实际纠正了几个符号（字节）——越少说明信号越干净
    corrected_symbols: int
    # 本次解码用了几个擦除位——越少说明越可信
    erasure_count: int
    # 观测码字与理论码字的汉明距离（错了多少比特），是最直观的信号质量度量
    bit_errors: int


def authentication_tag(trace_id: str, key: str | bytes) -> bytes:
    """由溯源号与服务端密钥计算 4 字节认证标签，即真正嵌入图片的净荷。

    :param trace_id: 溯源号，必须非空且首尾无空白。
    :param key: 认证密钥，字符串或字节，至少 32 字节。
    :return: HMAC-SHA256 摘要的前 4 字节。
    :raises TypeError: 类型不符。
    :raises ValueError: 溯源号为空/带空白，或密钥过短。

    对 ``trace_id`` 的空白检查看似苛刻，实则关键：``"TR-A"`` 与 ``"TR-A "``
    会算出完全不同的标签。嵌入时若混入了空白，提取时按干净的溯源号去比对
    就永远对不上，且这种错误在现场极难排查——所以在入口直接拒绝。

    消息前缀 ``robust-v4:`` 是域分隔符：同一把密钥若还用于其他用途，
    也不会产生可互换的摘要。
    """
    if type(trace_id) is not str:
        raise TypeError("trace_id must be a string")
    if not trace_id or trace_id != trace_id.strip():
        raise ValueError("trace_id must be nonempty and have no surrounding whitespace")

    if type(key) is str:
        encoded_key = key.encode("utf-8")
    elif type(key) is bytes:
        encoded_key = key
    else:
        raise TypeError("authentication key must be a string or bytes")
    if len(encoded_key) < AUTH_KEY_MIN_BYTES:
        raise ValueError("authentication key must contain at least 32 bytes")

    message = AUTH_MESSAGE_PREFIX + trace_id.encode("utf-8")
    # 截取前 4 字节：HMAC 的任意子串仍保持不可预测性，
    # 缩短只降低搜索空间（2³²），不削弱其单向性。
    return hmac.new(encoded_key, message, hashlib.sha256).digest()[:AUTH_TAG_BYTES]


def encode_codeword(payload: bytes) -> bytes:
    """把 4 字节净荷编成 8 字节 RS 码字（追加 4 字节校验）。

    :raises RuntimeError: 编码结果长度异常——这属于依赖库行为与预期不符，
        不是调用方的错，因此用 ``RuntimeError`` 而非 ``ValueError``。
        这一步校验不能省：长度不对会让后续的比特置换整体错位，
        而错位产生的是"能解出来但内容全错"的结果，比直接报错危险得多。
    """
    if type(payload) is not bytes:
        raise TypeError("v4 payload must be bytes")
    if len(payload) != RS_DATA_BYTES:
        raise ValueError("v4 payload must be exactly 4 bytes")
    encoded = bytes(_RS_CODEC.encode(payload))
    if len(encoded) != RS_CODEWORD_BYTES:
        raise RuntimeError("unexpected v4 RS codeword length")
    return encoded


def decode_candidate_codeword(
    observed: bytes,
    expected_payload: bytes,
    byte_confidences: list[float] | tuple[float, ...],
) -> CandidateDecode | None:
    """验证观测码字是否解码为**指定候选**的净荷。

    :param observed: 从图像中读出的 8 字节码字（可能含错）。
    :param expected_payload: 待验证候选记录的 4 字节认证标签。
    :param byte_confidences: 8 个字节各自的读取置信度，取值 ``[0, 1]``。
    :return: 验证通过时返回 :class:`CandidateDecode`，否则 ``None``。

    这不是"盲解码"而是"**定向验证**"：调用方已有候选名单，本函数逐一回答
    "这堆比特能不能解成这个候选"。这样设计使得单纯的 RS 纠错成功不足以
    构成命中——还必须与某个已登记的候选完全吻合，误报率因此大幅下降。

    **擦除逐级放宽是核心技巧。** RS(8,4) 在不知错误位置时最多纠 2 个字节，
    但若能指出哪些位置可疑（擦除），最多可纠 4 个。图像检测天然能给出
    每个字节的置信度，于是把置信度最低的若干字节标为擦除，就把纠错能力
    翻了一倍。从 0 个擦除开始逐级增加，是为了**优先采纳最保守的解释**：
    不擦除就能解通，说明信号本就干净，没必要冒险放宽。

    所有非法入参一律返回 ``None`` 而不抛异常——本函数在检测循环里会被
    高频调用，用返回值表达"此候选不匹配"比用异常控制流更合适。
    """
    if type(observed) is not bytes or len(observed) != RS_CODEWORD_BYTES:
        return None
    if type(expected_payload) is not bytes or len(expected_payload) != RS_DATA_BYTES:
        return None
    if type(byte_confidences) not in (list, tuple) or len(byte_confidences) != RS_CODEWORD_BYTES:
        return None
    if any(not _is_finite_confidence(value) for value in byte_confidences):
        return None

    expected_codeword = encode_codeword(expected_payload)
    # 按置信度升序排列字节下标，最不可信的排在最前面，优先被当作擦除。
    # 次级键用 index 保证置信度相同时顺序稳定，使整个解码过程可复现。
    confidence_order = sorted(
        range(RS_CODEWORD_BYTES),
        key=lambda index: (byte_confidences[index], index),
    )
    for erasure_count in RS_ERASURE_COUNTS:
        erasures = confidence_order[:erasure_count]
        try:
            decoded, corrected, errata = _RS_CODEC.decode(
                observed,
                erase_pos=erasures,
            )
        except (ReedSolomonError, ValueError, IndexError, TypeError):
            # 本轮擦除数解不通很正常，多擦几个再试；捕获范围放宽到四种异常，
            # 是因为 reedsolo 在不同的畸形输入下抛的类型并不统一。
            continue
        # 双重比对，缺一不可：
        #   decoded == expected_payload   —— 数据段对上了
        #   corrected == expected_codeword —— 连校验段也对上了
        # 只比数据段是不够的：RS 在超出纠错能力时可能"纠"出一个碰巧
        # 数据段相符、校验段却不同的合法码字，那属于误纠正。
        # 用 compare_digest 做等长常数时间比较，避免通过耗时差异侧信道
        # 逐字节试探出有效标签。
        if not hmac.compare_digest(bytes(decoded), expected_payload) or not hmac.compare_digest(
            bytes(corrected), expected_codeword
        ):
            continue
        return CandidateDecode(
            payload=bytes(decoded),
            corrected_codeword=bytes(corrected),
            # 用集合去重：reedsolo 的 errata 可能重复列出同一位置
            corrected_symbols=len({int(index) for index in errata}),
            erasure_count=erasure_count,
            # 异或后数 1 的个数即汉明距离，逐字节累加得到总比特错误数
            bit_errors=sum(
                (left ^ right).bit_count()
                for left, right in zip(observed, expected_codeword)
            ),
        )
    # 擦除数一路加到上限仍解不通，判定此候选不匹配。
    return None


def _is_finite_confidence(value: object) -> bool:
    """置信度必须是 ``[0, 1]`` 区间内的有限数。

    单独放行 ``int``，是为了让调用方能直接传 ``0`` / ``1`` 这种整型极值。
    NaN 会被 ``isfinite`` 挡掉——它参与比较恒为 ``False``，
    若放行会让擦除排序的结果变得不确定。
    """
    if type(value) is int:
        return 0 <= value <= 1
    if type(value) is float:
        return isfinite(value) and 0.0 <= value <= 1.0
    return False


def bytes_to_bits(codeword: bytes) -> tuple[int, ...]:
    """把 8 字节码字展开成 64 个比特，**高位在前**（MSB first）。

    位序是嵌入端与提取端必须一致的约定，改了就解不出旧图。
    ``range(7, -1, -1)`` 即从第 7 位取到第 0 位，实现高位优先。
    """
    if type(codeword) is not bytes:
        raise TypeError("v4 codeword must be bytes")
    if len(codeword) != RS_CODEWORD_BYTES:
        raise ValueError("v4 codeword must be exactly 8 bytes")
    return tuple(
        (byte >> shift) & 1
        for byte in codeword
        for shift in range(7, -1, -1)
    )


@lru_cache(maxsize=PHASE_COUNT)
def phase_permutation(phase: int) -> tuple[int, ...]:
    """生成某个相位的比特置换表：``逻辑位序 -> 物理位序``。

    置换的作用是**抗突发错误**。图像上的局部损伤（一块被涂抹、一条压缩条带）
    会连续毁掉相邻若干位；打散之后，这些连续的物理位对应的是分散的逻辑位，
    集中的突发错误就被摊成了零星错误，正好落进 RS 的纠错能力范围内。

    置换由 SHA-256 种子确定性生成，因此嵌入端和提取端无需传递任何数据、
    仅凭相位号就能算出同一张表。只有 4 个相位，用 ``lru_cache`` 全量缓存。

    .. note::
        这里依赖 ``random.Random`` 的 ``shuffle`` 实现细节——Python 版本
        若改变该算法，历史图片将无法解码。这是与 ``PHASE_PERMUTATION_PREFIX``
        同等级别的格式契约。
    """
    _validate_phase(phase)
    seed = hashlib.sha256(
        PHASE_PERMUTATION_PREFIX + str(phase).encode("ascii")
    ).digest()
    values = list(range(CODEWORD_BITS))
    # 用独立的 Random 实例而非全局 random，避免影响（或被影响于）
    # 进程内其他地方的随机数状态。
    random.Random(int.from_bytes(seed, "big")).shuffle(values)
    return tuple(values)


def inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    """求置换表的逆表，供提取端把物理位序还原回逻辑位序。

    :raises ValueError: 入参不是 0~63 的一个完整排列。

    校验用 ``sorted(...) == list(range(64))``，同时排除了长度不符、
    元素重复、存在缺口三种情况——一次比较覆盖全部非法形态。
    正向表满足 ``permutation[逻辑] = 物理``，因此逆表按
    ``inverse[物理] = 逻辑`` 填写即可。
    """
    if type(permutation) is not tuple:
        raise TypeError("permutation must be a tuple")
    if (
        len(permutation) != CODEWORD_BITS
        or any(type(value) is not int for value in permutation)
        or sorted(permutation) != list(range(CODEWORD_BITS))
    ):
        raise ValueError("permutation must contain each index from 0 through 63")
    inverse = [0] * CODEWORD_BITS
    for logical, physical in enumerate(permutation):
        inverse[physical] = logical
    return tuple(inverse)


def permute_codeword_bits(codeword: bytes, phase: int) -> tuple[int, ...]:
    """把码字展开为比特并按相位置换，得到可直接写入分块的 64 个物理比特。

    即 ``bytes_to_bits`` 与 :func:`phase_permutation` 的组合，
    是嵌入端调用的入口。
    """
    _validate_phase(phase)
    logical_bits = bytes_to_bits(codeword)
    physical_bits = [0] * CODEWORD_BITS
    for logical, physical in enumerate(phase_permutation(phase)):
        physical_bits[physical] = logical_bits[logical]
    return tuple(physical_bits)


def phase_for_tile(tile_x: int, tile_y: int) -> int:
    """由分块坐标推出它该用哪个相位。

    ``(x + 2y) % 4`` 这个系数组合的用意是让**上下左右四个邻块相位各不相同**：
    水平相邻差 1，垂直相邻差 2，因此任一分块与其四邻的相位都不重复。
    这样一来，局部区域内四个相位齐备，即便只截取到图片的一小块，
    也大概率能凑齐 ``minimum_phases`` 要求的相位数量。

    整个 V4 每个分块嵌的都是同一个码字（"full_repeat"），
    靠相位差异来避免规则重复图案在频域中形成可被察觉、可被针对性擦除的峰。
    """
    if type(tile_x) is not int or type(tile_y) is not int:
        raise TypeError("tile coordinates must be integers")
    if tile_x < 0 or tile_y < 0:
        raise ValueError("tile coordinates must be nonnegative")
    return (tile_x + 2 * tile_y) % PHASE_COUNT


def candidate_match_probability(candidate_count: int) -> float:
    """计算给定候选数下的误匹配概率，用于给检测结论标注可信度。

    认证标签 32 位，随机噪声碰巧撞上某个特定标签的概率是 1/2³²；
    同时比对 N 个候选，概率线性放大到 N/2³²。

    候选数上限 8 就是由此而来：候选越多误报概率越高，必须硬性设限，
    不能为了提高召回率而无限扩大候选集。
    """
    if type(candidate_count) is not int:
        raise TypeError("candidate count must be an integer")
    if not 1 <= candidate_count <= 8:
        raise ValueError("candidate count must be between 1 and 8")
    return candidate_count / (2**32)


def _validate_phase(phase: int) -> None:
    """相位必须是 0~3 的整数。"""
    if type(phase) is not int:
        raise TypeError("phase must be an integer")
    if phase not in range(PHASE_COUNT):
        raise ValueError("phase must be between 0 and 3")


__all__ = (
    "AUTH_KEY_MIN_BYTES",
    "AUTH_MESSAGE_PREFIX",
    "AUTH_TAG_BYTES",
    "CandidateDecode",
    "CODEWORD_BITS",
    "PHASE_COUNT",
    "PHASE_PERMUTATION_PREFIX",
    "RS_CODEWORD_BYTES",
    "RS_DATA_BYTES",
    "RS_ERASURE_COUNTS",
    "RS_PARITY_BYTES",
    "authentication_tag",
    "bytes_to_bits",
    "candidate_match_probability",
    "decode_candidate_codeword",
    "encode_codeword",
    "inverse_permutation",
    "phase_for_tile",
    "phase_permutation",
    "permute_codeword_bits",
)
