"""传统链路 robust 水印 v3 的认证层：HMAC 认证码与比特置换。

v3 与 v2（``trace_app.watermark.ecc``）的根本区别在于**嵌什么**：v2 嵌的是溯源号
派生出的数据加 RS 校验，v3 嵌的是一段由服务端密钥算出的 8 字节 HMAC 认证码。

**为什么要认证而不是直接嵌溯源号？** 认证码由密钥决定，攻击者即使完全掌握
本文件的算法，没有密钥也构造不出能通过校验的水印，检测端因此能可靠区分
真水印和巧合噪声。反过来说，认证码本身不携带任何可读信息，检测时必须拿候选
记录里登记的认证码去比对——这也正好把 v3 变成"定向验证"而非盲解码。

**为什么这里没有纠错码？** 8 字节认证码恰好铺满一个分块的 64 比特容量，
没有余量再放校验字节。v3 改用另一条路子换鲁棒性：每个分块都嵌**完整的**
认证码（不像 v2 那样只嵌 1/3），检测时把所有分块的软判决分数累加，靠多块
平均把噪声压下去，最后用汉明距离半径判定命中，
误报概率由 :func:`candidate_radius_probability` 量化。

**置换的作用是抗突发错误。** 图像上的局部损伤（一块被涂抹、一条压缩条带）
会连续毁掉相邻若干位。相邻分块使用不同的置换表，同一个逻辑比特在不同分块里
落在不同的物理位置上，于是集中的突发损伤被摊成分散的零星误差，
多块累加投票时更容易被压过去。

被 ``trace_app/watermark/robust.py`` 的 v3 嵌入/提取流程使用。
"""

import hashlib
import hmac
import math
import random


# 认证码长度：取 HMAC-SHA256 摘要的前 8 字节（64 位），
# 正好铺满一个分块的 64 比特载荷容量
AUTH_CODE_BYTES = 8
AUTH_CODE_BITS = AUTH_CODE_BYTES * 8
# 密钥长度下限，低于此值拒绝工作——弱密钥会让整个认证机制失去意义
AUTH_KEY_MIN_BYTES = 32
# HMAC 消息前缀（域分隔符）：保证同一把密钥在别处的用法不会与这里产生
# 可互换的摘要，避免跨用途的重放攻击
AUTH_MESSAGE_PREFIX = b"robust-v3:"
# 置换随机种子的前缀，同样起域分隔作用。
# 这两个前缀都属于格式契约，改动它们会导致所有旧图无法解码。
AUTH_PERMUTATION_PREFIX = b"robust-v3-carrier-permutation:"


def _validated_key(key: str | bytes | None) -> bytes:
    """把密钥规整成字节串并校验长度。

    :raises ValueError: 密钥不足 32 字节。

    非 ``str``/``bytes``（含 ``None``）一律当成空串处理，随后被长度检查拦下。
    这样做是让"根本没配密钥"和"密钥太短"走同一条报错路径，
    调用方只需处理一种异常。

    .. note::
        代价是 ``bytearray``、``memoryview`` 这类同样合法的字节容器也会被当成
        空串，报出"至少 32 字节"这种有误导性的信息。此处如实描述现状，
        未作改动。
    """
    if isinstance(key, str):
        encoded = key.encode("utf-8")
    elif isinstance(key, bytes):
        encoded = key
    else:
        encoded = b""
    if len(encoded) < AUTH_KEY_MIN_BYTES:
        raise ValueError("WATERMARK_AUTH_KEY must contain at least 32 bytes")
    return encoded


def auth_code_from_trace(trace_id: str, key: str | bytes | None) -> bytes:
    """由溯源号与服务端密钥计算 8 字节认证码，即真正嵌入图片的那 64 个比特。

    :param trace_id: 溯源号。
    :param key: 认证密钥，字符串或字节，至少 32 字节。
    :return: HMAC-SHA256 摘要的前 8 字节。
    :raises ValueError: 密钥缺失或不足 32 字节。

    截取前 8 字节：HMAC 的任意子串仍保持不可预测性，
    缩短只降低搜索空间（2⁶⁴），不削弱其单向性。

    .. note::
        这里对 ``trace_id`` 不做空白检查，``"TR-A"`` 与 ``"TR-A "`` 会算出
        完全不同的认证码。嵌入时若混入了空白，提取时按干净的溯源号去比对就
        永远对不上，而这种错误在现场极难排查。V4 的 ``authentication_tag``
        在入口直接拒绝带首尾空白的溯源号，此处未作改动。
    """
    validated = _validated_key(key)
    message = AUTH_MESSAGE_PREFIX + str(trace_id).encode("utf-8")
    return hmac.new(validated, message, hashlib.sha256).digest()[:AUTH_CODE_BYTES]


def phase_permutation(phase: int) -> tuple[int, ...]:
    """生成某个相位的比特置换表：``逻辑位序 -> 物理位序``。

    :param phase: 相位 0~2，由分块坐标经 ``trace_app.watermark.ecc.tile_phase`` 算出。
    :return: 长度 64 的置换表，``表[逻辑位] = 物理位``。
    :raises ValueError: 相位不在 0~2。

    置换由 SHA-256 种子确定性生成，因此嵌入端与提取端无需传递任何数据、
    仅凭相位号就能算出同一张表；相位号又完全由分块坐标决定，
    于是整条链路上不需要额外的同步信息。

    种子刻意**不掺入密钥**：置换只负责打散突发错误，安全性由 HMAC 认证码本身
    提供，把置换做成公开可复算的反而简化了实现，也不损失什么。

    .. note::
        这里依赖 ``random.Random`` 的 ``shuffle`` 实现细节——Python 版本若改变
        该算法，历史图片将无法解码。这是与 ``AUTH_PERMUTATION_PREFIX``
        同等级别的格式契约。

    .. note::
        本函数没有缓存，而检测时每个分块都会调用一次，等于对每个分块重跑一遍
        SHA-256 加 64 元素 shuffle。相位只有 3 个、结果又完全确定，本可以像
        V4 的同名函数那样加 ``lru_cache``。此处如实记录，未作改动。
    """
    if phase not in range(3):
        raise ValueError("watermark carrier phase must be 0, 1, or 2")
    seed = hashlib.sha256(
        AUTH_PERMUTATION_PREFIX + str(phase).encode("ascii")
    ).digest()
    values = list(range(AUTH_CODE_BITS))
    # 用独立的 Random 实例而非全局 random，避免影响（或被影响于）
    # 进程内其他地方的随机数状态。
    random.Random(int.from_bytes(seed, "big")).shuffle(values)
    return tuple(values)


def inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    """求置换表的逆表，把物理位序还原回逻辑位序。

    :raises ValueError: 入参不是 0~63 的一个完整排列。

    校验用 ``sorted(...) == list(range(64))``，一次比较同时排除了长度不符、
    元素重复、存在缺口三种非法形态。正向表满足 ``permutation[逻辑] = 物理``，
    因此逆表按 ``inverse[物理] = 逻辑`` 填写即可。

    robust.py 的 v3 解码并不调用本函数——它直接按
    ``aggregate[逻辑] += 分数[permutation[逻辑]]`` 累加，等价于隐式用了逆表。
    本函数主要供测试与外部工具校验置换表的自洽性。
    """
    if len(permutation) != AUTH_CODE_BITS or sorted(permutation) != list(range(AUTH_CODE_BITS)):
        raise ValueError("watermark carrier permutation must contain 0 through 63")
    inverse = [0] * AUTH_CODE_BITS
    for logical, physical in enumerate(permutation):
        inverse[physical] = logical
    return tuple(inverse)


def permuted_code_bits(code: bytes, phase: int) -> tuple[int, ...]:
    """把 8 字节认证码展开成 64 比特并按相位置换，得到可直接写入分块的物理比特。

    :param code: 8 字节认证码。
    :param phase: 分块相位 0~2。
    :return: 64 个 0/1，下标即分块内的物理位序。
    :raises ValueError: 认证码长度不是 8，或相位非法（由
        :func:`phase_permutation` 抛出）。

    展开位序是**高位在前**（MSB first）：``shift`` 从 63 递减到 0。
    位序是嵌入端与提取端必须一致的约定，改了就解不出旧图。
    """
    if len(code) != AUTH_CODE_BYTES:
        raise ValueError("v3 watermark auth code must be exactly 8 bytes")
    logical_bits = tuple(
        (int.from_bytes(code, "big") >> shift) & 1
        for shift in range(AUTH_CODE_BITS - 1, -1, -1)
    )
    physical_bits = [0] * AUTH_CODE_BITS
    for logical, physical in enumerate(phase_permutation(phase)):
        physical_bits[physical] = logical_bits[logical]
    return tuple(physical_bits)


def candidate_radius_probability(max_errors: int, bits: int = AUTH_CODE_BITS) -> float:
    """计算随机比特串落进某个汉明球内的概率，用于给命中阈值定量。

    :param max_errors: 判定命中所允许的最大比特错误数，即汉明半径。
    :param bits: 码长，默认 64。
    :return: 一段随机比特恰好落在该半径内的概率。
    :raises ValueError: 半径为负或超过码长。

    v3 没有纠错码，命中判据就是"观测比特与候选认证码的汉明距离 <= 半径"。
    半径放得越大召回越高，误报概率也随之膨胀，本函数就是用来量化这个取舍的：
    分子是半径内的码字总数 ``Σ C(bits, k)``，分母是整个码字空间 ``2^bits``。
    robust.py 默认用的半径 8 对应约 1e-9 量级，商用上足够小。

    注意这只是**单个候选**的误报概率；同时比对 N 个候选时需再乘以 N。
    """
    if max_errors < 0 or max_errors > bits:
        raise ValueError("max_errors must be within the code length")
    return sum(math.comb(bits, count) for count in range(max_errors + 1)) / (2**bits)
