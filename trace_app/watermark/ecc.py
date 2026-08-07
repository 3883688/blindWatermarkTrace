"""传统链路 robust 水印 v2 的 RS 纠错层：编码、定向解码与相位切分。

本模块只跟字节打交道，不涉及任何图像操作，由
``trace_app/watermark/robust.py`` 的 v2 嵌入/提取流程调用。

**数据流（嵌入方向）**::

    8 字节净荷 ──RS(24,8)──────> 24 字节码字（8 数据 + 16 校验）
              ──codeword_phase─> 按相位切成 3 段，每段 8 字节
              ──tile_phase─────> 分块按自身坐标算出相位，只嵌对应那一段

提取方向按相位把三段拼回 24 字节，再交给 :func:`decode_expected_codeword`
连同候选净荷一起做定向验证。

**为什么要把码字切成三段？** 一个分块只有 64 比特（8 字节）载荷容量，
装不下 24 字节码字。v2 的取舍是"牺牲单块自足性换纠错强度"：16 字节校验
能纠 8 个错误字节，代价是必须覆盖到全部 3 个相位才拼得出完整码字，
截图太小就直接解不出来。V4 走的是相反的路子——每块重复嵌同一个短码字，
见 ``watermark_v4/payload.py``。

**为什么是"定向验证"而不是盲解码？** 调用方已经通过特征匹配拿到了候选记录，
本模块回答的是"这堆比特能不能解成这个候选的净荷"。盲解码出来的任意合法码字
不构成命中，必须与某个已登记候选完全吻合，误报率因此大幅下降。
"""

from typing import Any

from reedsolo import RSCodec, ReedSolomonError


# 净荷长度：v2 嵌的是由溯源号派生出的 8 字节数据
RS_DATA_BYTES = 8
# 校验字节数。RS 的纠错额度满足 2e + f <= 16（e 为位置未知的错误字节数，
# f 为已指出位置的擦除数），即最多纠 8 个未知错误、或 16 个擦除。
RS_PARITY_BYTES = 16
RS_CODEWORD_BYTES = 24
# 解码时依次尝试的擦除数量。步长取 2 而非 1，是因为每多一档就要多跑一遍 RS
# 解码，而检测循环里每个候选都要走这一趟——用档位粒度换耗时。
RS_ERASURE_COUNTS = (0, 2, 4, 6)
# 兜底路径的汉明距离上限：RS 彻底解不通时，若观测码字与理论码字相差不超过
# 32 比特（共 192 比特），仍判为命中。随机噪声落进这个半径的概率在 1e-20
# 量级，见 tests/test_watermark_ecc.py。
RS_CANDIDATE_MAX_BIT_ERRORS = 32
# 相位数与每相位携带的字节数，3 * 8 = 24 恰好覆盖整个码字。
# 这两个值属于格式契约，改了就解不出旧图。
RS_PHASES = 3
RS_PHASE_BYTES = 8

# 全模块共用一个编解码器实例：RSCodec 构造时要建伽罗华域查找表，开销不小，
# 且它在编解码过程中无可变状态，可安全复用。
_CODEC = RSCodec(RS_PARITY_BYTES, nsize=RS_CODEWORD_BYTES)


def encode_codeword(payload: bytes) -> bytes:
    """把 8 字节净荷编成 24 字节 RS 码字（追加 16 字节校验）。

    :param payload: 恰好 8 字节的净荷。
    :return: 24 字节码字；RS 是系统码，前 8 字节就是原净荷本身。
    :raises ValueError: 净荷长度不是 8。
    :raises RuntimeError: 编码结果长度异常——这属于依赖库行为与预期不符，
        不是调用方的错，因此用 ``RuntimeError`` 而非 ``ValueError``。
        这一步校验不能省：长度不对会让后续按相位切段整体错位，
        而错位产生的是"能解出来但内容全错"的结果，比直接报错危险得多。
    """
    if len(payload) != RS_DATA_BYTES:
        raise ValueError("RS watermark payload must be exactly 8 bytes")
    encoded = bytes(_CODEC.encode(payload))
    if len(encoded) != RS_CODEWORD_BYTES:
        raise RuntimeError("unexpected RS watermark codeword length")
    return encoded


def decode_expected_codeword(
    observed: bytes,
    expected_payload: bytes,
    byte_confidences: list[float],
) -> dict[str, Any] | None:
    """验证观测码字是否解得出**指定候选**的净荷。

    :param observed: 从图像中按相位拼回的 24 字节码字（可能含错）。
    :param expected_payload: 待验证候选记录的 8 字节净荷。
    :param byte_confidences: 24 个字节各自的读取置信度，用于挑选擦除位置。
    :return: 验证通过时返回结果字典，含 ``payload``、``corrected_codeword``、
        ``corrected_symbols``、``erasure_count``、``bit_errors``、
        ``recovery_method`` 六个键；不匹配时返回 ``None``。

    **擦除逐级放宽是核心技巧。** RS(24,8) 在不知错误位置时最多纠 8 个字节，
    但若能指出哪些位置可疑（擦除），额度按 ``2e + f <= 16`` 折算，
    等于把纠错能力翻倍。图像检测天然能给出每个字节的置信度，于是把置信度
    最低的若干字节标为擦除。从 0 个擦除开始逐级增加，是为了**优先采纳最保守
    的解释**：不擦除就能解通，说明信号本就干净，没必要冒险放宽。

    **双重比对缺一不可。** 只比数据段是不够的：RS 在超出纠错能力时可能"纠"出
    一个碰巧数据段相符、校验段却不同的合法码字，那属于误纠正。

    **为什么还要汉明距离兜底？** RS 是按字节（符号）纠错的，一个字节里错 1 位
    和错 8 位对它是一回事。图像里的弱信号常常表现为"很多字节各错一两位"——
    出错符号数早已超过 8 个，RS 必然失败，但总比特错误率其实很低。这种情况下
    退回汉明距离判定（``recovery_method == "expected_codeword_distance"``）
    能救回大量真实命中，而 32/192 的半径把误报概率压在 1e-20 量级。

    所有非法入参一律返回 ``None`` 而不抛异常——本函数在检测循环里会被
    高频调用，用返回值表达"此候选不匹配"比用异常控制流更合适。

    .. note::
        兜底路径返回的 ``corrected_symbols`` 是观测码字与理论码字**相差**的
        字节数，而非 RS 实际纠正的符号数；``payload`` 也是直接回填候选净荷，
        并非从观测数据里解出来的。这两个字段的语义与 RS 路径不同，
        调用方若把它们当作统一的信号质量指标需留意。此处如实描述现状，
        未作改动。

    .. note::
        本函数不校验 ``byte_confidences`` 的元素是否为有限数。传入 NaN 时
        排序结果不确定（NaN 参与比较恒为 ``False``），擦除位置的选取会失去
        可复现性。V4 的同名逻辑补上了这项检查，此处未作改动。
    """
    if len(observed) != RS_CODEWORD_BYTES or len(expected_payload) != RS_DATA_BYTES:
        return None
    if len(byte_confidences) != RS_CODEWORD_BYTES:
        return None

    expected_codeword = encode_codeword(expected_payload)
    # 按置信度升序排列字节下标，最不可信的排在最前面，优先被当作擦除。
    # Python 的排序稳定，置信度相同的字节保持原下标次序，结果可复现。
    confidence_order = sorted(
        range(RS_CODEWORD_BYTES),
        key=lambda index: byte_confidences[index],
    )
    for erasure_count in RS_ERASURE_COUNTS:
        erasures = confidence_order[:erasure_count]
        try:
            decoded, corrected, errata = _CODEC.decode(observed, erase_pos=erasures)
        except (ReedSolomonError, ValueError, IndexError):
            # 本轮擦除数解不通很正常，多擦几个再试；捕获范围放宽到三种异常，
            # 是因为 reedsolo 在不同的畸形输入下抛的类型并不统一。
            continue
        # 双重比对：数据段与校验段都得对上，只对上数据段可能是误纠正的结果。
        if bytes(decoded) != expected_payload or bytes(corrected) != expected_codeword:
            continue
        return {
            "payload": bytes(decoded),
            "corrected_codeword": bytes(corrected),
            # 用集合去重：reedsolo 的 errata 可能重复列出同一位置
            "corrected_symbols": len(set(int(index) for index in errata)),
            "erasure_count": erasure_count,
            # 异或后数 1 的个数即汉明距离，逐字节累加得到总比特错误数
            "bit_errors": sum(
                (left ^ right).bit_count()
                for left, right in zip(observed, expected_codeword)
            ),
            "recovery_method": "reed_solomon",
        }
    # 走到这里说明擦除数一路加到上限 RS 仍解不通，改用比特级的汉明距离判定。
    bit_errors = sum(
        (left ^ right).bit_count()
        for left, right in zip(observed, expected_codeword)
    )
    if bit_errors <= RS_CANDIDATE_MAX_BIT_ERRORS:
        return {
            "payload": expected_payload,
            "corrected_codeword": expected_codeword,
            "corrected_symbols": sum(
                left != right for left, right in zip(observed, expected_codeword)
            ),
            "erasure_count": 0,
            "bit_errors": bit_errors,
            "recovery_method": "expected_codeword_distance",
        }
    # 连汉明半径都超了，判定此候选不匹配。
    return None


def tile_phase(tile_x: int, tile_y: int) -> int:
    """由分块坐标推出它该携带码字的哪一段（相位 0/1/2）。

    ``(x + 2y) % 3`` 这个系数组合让水平相邻的分块相位差 1、垂直相邻差 2，
    于是任一分块的上下左右四邻相位都与它不同，任意 2x2 的小区域内三个相位
    必定齐备。这样即便只截取到图片的一小块，也大概率能凑齐三段拼出完整码字
    ——对"必须集齐 3 个相位才能解码"的 v2 来说这是硬需求。
    """
    return (int(tile_x) + 2 * int(tile_y)) % RS_PHASES


def codeword_phase(codeword: bytes, phase: int) -> bytes:
    """取出码字中属于该相位的 8 字节段。

    :param codeword: 24 字节完整码字。
    :param phase: 相位 0~2。
    :return: 该相位对应的 8 字节，可直接铺进一个分块的 64 比特。
    :raises ValueError: 码字长度不是 24，或相位不在 0~2。

    切法是简单的连续分段（相位 0 取 ``[0:8]``、1 取 ``[8:16]``、2 取 ``[16:24]``），
    不做任何打散——v2 靠"每块只嵌 1/3 码字 + 16 字节 RS 校验"抗损伤，
    比特级置换是 v3 才引入的手段（见 ``trace_app.watermark.auth``）。
    分段边界属于格式契约，改了就解不出旧图。
    """
    if len(codeword) != RS_CODEWORD_BYTES:
        raise ValueError("RS watermark codeword must be exactly 24 bytes")
    if phase not in range(RS_PHASES):
        raise ValueError("RS watermark phase must be 0, 1, or 2")
    start = phase * RS_PHASE_BYTES
    return codeword[start : start + RS_PHASE_BYTES]
