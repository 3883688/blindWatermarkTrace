"""LSB（最低有效位）水印：把完整元信息明文写进像素最低位。

**特点与定位。** LSB 是所有水印层里信息量最大、速度最快的一层——它直接把
整个 JSON 载荷写进去，提取时无需查库、无需比对，一次读取即可拿到全部信息。
代价是极其脆弱：任何有损压缩、重新编码、甚至改一次亮度都会毁掉它。

因此它在检测流水线中排在最前面（最快路径），但**从不作为唯一依赖**：
真正抗攻击的信息由鲁棒水印层与 V4 承载。

**两级冗余布局**：

1. **全图 LSB** —— 从图片左上角开始按像素顺序连续写入。原图未被改动时，
   一次线性扫描就能读出。
2. **分块 LSB** —— 在图中多个 ``BLOCK_SIZE`` 大小的方块内各写一份完整副本。
   图片被裁剪后，只要还剩下任意一个完整方块，仍能恢复。

**数据包格式**::

    MAGIC(4字节) + 长度(4字节，大端) + JSON载荷(UTF-8)

魔数用于快速排除"这串比特不是水印"，长度字段界定载荷边界。
"""

import json
import re
from typing import Any, Callable

from PIL import Image

from trace_app.config import BLOCK_SIZE, BLOCK_STRIDE, MAGIC


class PayloadTooLargeError(ValueError):
    """图片像素数不足以容纳整个载荷。"""


class WatermarkNotFoundError(LookupError):
    """全图与分块两条路径都没读出有效水印。"""


def bits_from_bytes(data: bytes) -> list[int]:
    """字节序列展开成比特列表，**高位在前**。

    位序是嵌入与提取必须一致的约定，与 :func:`bytes_from_bits` 严格互逆。
    """
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def bytes_from_bits(bits: list[int]) -> bytes:
    """比特列表合回字节，:func:`bits_from_bytes` 的逆操作。

    通过 ``byte = (byte << 1) | bit`` 左移累积实现高位在前。
    末尾不足 8 位的残余会被补零成一个字节——调用方需自行保证长度对齐，
    本函数不做校验。
    """
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i : i + 8]:
            byte = (byte << 1) | bit
        result.append(byte)
    return bytes(result)


def lsb_bits_from_pixels(pixels: Any):
    """从像素序列逐个吐出 R、G、B 三通道的最低位。

    写成生成器是为了**惰性求值**：解码时先只取 64 位读包头，
    魔数不对就立刻停止，不必把整张图的比特都取出来。
    大图上这个差别很可观。
    """
    for pixel in pixels:
        yield pixel[0] & 1
        yield pixel[1] & 1
        yield pixel[2] & 1


def valid_watermark_payload(payload: Any) -> bool:
    """校验解出的载荷是否为本系统写入的合法水印。

    这道校验是**必需的**，不是锦上添花：任意图片的像素最低位本就是随机的，
    偶尔会碰巧凑出魔数并解析成 JSON。若不做结构校验，就会把无关图片
    误判为命中，产生错误的溯源结论。

    四项检查逐级收紧：必填字段非空 → ``id`` 是 32 位小写十六进制 →
    ``evidence_uuid`` 是 32 位**大写**十六进制 → 模式在枚举内。
    大小写不同是格式约定，正好又多一重区分度。
    """
    if not isinstance(payload, dict):
        return False
    required_strings = ("id", "trace_id", "user_id", "mode", "created_at")
    for key in required_strings:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    if not re.fullmatch(r"[0-9a-f]{32}", payload["id"]):
        return False
    if not re.fullmatch(r"[0-9A-F]{32}", str(payload.get("evidence_uuid", ""))):
        return False
    if payload.get("mode") not in {"lsb", "dct", "dwt", "fft", "hybrid"}:
        return False
    return True


def packet_from_payload(payload: dict[str, Any]) -> bytes:
    """把载荷字典序列化成 ``魔数 + 长度 + JSON`` 的数据包。

    ``separators=(",", ":")`` 去掉 JSON 默认的空格，
    ``ensure_ascii=False`` 让中文直接以 UTF-8 存储而不转成 ``\\uXXXX``——
    两者都是为了压缩体积。LSB 容量按像素数硬性受限，
    每省一个字节就多一分嵌进小图的可能。
    """
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return MAGIC + len(data).to_bytes(4, "big") + data


def write_packet_to_pixels(pixels: list[tuple[int, int, int]], packet: bytes) -> list[tuple[int, int, int]]:
    """把数据包写进像素序列的最低位（全图模式）。

    :raises PayloadTooLargeError: 容量不足（每像素 3 比特）。

    ``(channel & 0xFE) | bit`` 是标准的置位操作：先用 ``0xFE``
    （二进制 11111110）清掉最低位，再或上目标比特。
    每个通道值最多变动 1，肉眼完全不可察觉。

    比特写完后**剩余像素原样保留**，不做任何填充——
    保持不变既省算力，也不引入多余的画质损失。
    """
    bits = bits_from_bytes(packet)
    capacity = len(pixels) * 3
    if len(bits) > capacity:
        raise PayloadTooLargeError("图片尺寸过小，无法嵌入水印信息")

    out = []
    idx = 0
    for pixel in pixels:
        channels = list(pixel)
        for channel in range(3):
            if idx < len(bits):
                channels[channel] = (channels[channel] & 0xFE) | bits[idx]
                idx += 1
        out.append(tuple(channels))
    return out


def embed_lsb(
    image: Image.Image,
    payload: dict[str, Any],
    *,
    embed_block_lsb_fn: Callable[[Image.Image, dict[str, Any]], Image.Image] | None = None,
    write_packet_to_pixels_fn: Callable[
        [list[tuple[int, int, int]], bytes], list[tuple[int, int, int]]
    ] | None = None,
) -> Image.Image:
    """嵌入 LSB 水印：先写全图，再叠加分块副本。

    :param embed_block_lsb_fn: 可注入的分块嵌入实现（测试用）。
    :param write_packet_to_pixels_fn: 可注入的全图写入实现（测试用）。

    两级都写。分块副本会覆盖掉全图副本在相应区域的比特，
    但这不构成问题——两者内容完全相同，覆盖后仍是同一份数据。
    """
    rgb = image.convert("RGB")
    packet = packet_from_payload(payload)
    pixels = list(rgb.getdata())
    write_fn = write_packet_to_pixels_fn or write_packet_to_pixels
    rgb.putdata(write_fn(pixels, packet))
    block_fn = embed_block_lsb_fn or embed_block_lsb
    return block_fn(rgb, payload)


def iter_block_origins(width: int, height: int):
    """按 ``BLOCK_STRIDE`` 步长枚举所有完整方块的左上角坐标。

    图片小于一个方块时直接不产出（``return`` 空生成器）。
    步长小于块尺寸意味着方块之间**互相重叠**，这是刻意的：
    重叠布局让任意位置的裁剪都更可能完整保住至少一个方块。
    """
    if width < BLOCK_SIZE or height < BLOCK_SIZE:
        return
    for y in range(0, height - BLOCK_SIZE + 1, BLOCK_STRIDE):
        for x in range(0, width - BLOCK_SIZE + 1, BLOCK_STRIDE):
            yield x, y


def embed_block_lsb(image: Image.Image, payload: dict[str, Any]) -> Image.Image:
    """在每个方块内各写一份完整数据包副本，用于抗裁剪。

    单个方块装不下载荷时**原样返回**，不报错——分块层是尽力而为的增强层，
    装不下就退化为只有全图 LSB，不该因此让整个嵌入流程失败。
    """
    rgb = image.convert("RGB")
    packet = packet_from_payload(payload)
    bits = bits_from_bytes(packet)
    width, height = rgb.size
    # 用 load() 拿到像素访问对象，支持按坐标随机读写（分块必需，
    # 而全图模式用的 getdata/putdata 只能顺序处理）
    pixels = rgb.load()
    required = len(bits)
    block_capacity = BLOCK_SIZE * BLOCK_SIZE * 3
    if block_capacity < required:
        return rgb

    for ox, oy in iter_block_origins(width, height) or []:
        idx = 0
        for y in range(oy, oy + BLOCK_SIZE):
            for x in range(ox, ox + BLOCK_SIZE):
                channels = list(pixels[x, y])
                for channel in range(3):
                    if idx < required:
                        channels[channel] = (channels[channel] & 0xFE) | bits[idx]
                        idx += 1
                pixels[x, y] = tuple(channels)
                # 写满即停：方块剩余像素保持原样，减少不必要的画质损失
                if idx >= required:
                    break
            if idx >= required:
                break
    return rgb


def decode_bits_from_pixels(pixels: Any) -> dict[str, Any] | None:
    """从像素序列的最低位解出水印载荷，失败一律返回 ``None``。

    分三步且**层层早退**，这对性能至关重要——本函数会被分块扫描调用成千
    上万次，绝大多数位置都不含水印，必须尽快否掉：

    1. 只取 64 位（8 字节包头），魔数不符立即返回；
    2. 读长度字段，超出 8192 字节判为异常（真实载荷远小于此）；
    3. 按长度取载荷比特，UTF-8 解码 + JSON 解析 + 结构校验。

    任何一步失败都返回 ``None``，不抛异常——"这个位置没有水印"是扫描过程中
    的常态，用返回值表达比异常合适得多。
    """
    bit_iter = lsb_bits_from_pixels(pixels)
    header_bits = []
    for _ in range(64):
        try:
            header_bits.append(next(bit_iter))
        except StopIteration:
            # 像素不够 22 个（64 位 ÷ 3 通道），连包头都读不满
            return None
    if len(header_bits) < 64:
        return None
    header = bytes_from_bits(header_bits)
    if len(header) < 8 or header[:4] != MAGIC:
        return None

    size = int.from_bytes(header[4:8], "big")
    # 长度上限 8192：正常载荷仅几百字节。设限可防止随机比特凑出的
    # 巨大长度值导致这里去读几百兆比特。
    if size <= 0 or size > 8192:
        return None
    payload_bits = []
    for _ in range(size * 8):
        try:
            payload_bits.append(next(bit_iter))
        except StopIteration:
            return None
    payload = bytes_from_bits(payload_bits)
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not valid_watermark_payload(decoded):
        return None
    return decoded


def extract_lsb(
    image: Image.Image,
    *,
    extract_full_lsb_fn: Callable[[Image.Image], dict[str, Any] | None] | None = None,
    extract_block_lsb_fn: Callable[[Image.Image], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """提取 LSB 水印：先试全图，再试分块。

    :raises WatermarkNotFoundError: 两条路径都没读出。

    顺序按代价排：全图只需一次线性扫描；分块要遍历大量候选位置，
    贵得多，所以放在后面。
    """
    full_fn = extract_full_lsb_fn or extract_full_lsb
    block_fn = extract_block_lsb_fn or extract_block_lsb
    payload = full_fn(image)
    if payload:
        return payload
    payload = block_fn(image)
    if payload:
        return payload
    raise WatermarkNotFoundError("未检测到可识别的隐式水印")


def extract_full_lsb(image: Image.Image) -> dict[str, Any] | None:
    """全图模式提取：从左上角开始按像素顺序读。

    只有图片完全未被裁剪、未被重编码时才能成功，但代价极低，
    值得每次都先试一下。
    """
    rgb = image.convert("RGB")
    payload = decode_bits_from_pixels(list(rgb.getdata()))
    if payload:
        return payload
    return None


def extract_block_lsb(image: Image.Image) -> dict[str, Any] | None:
    """分块模式提取：扫描候选位置，找出任意一个完整的方块副本。

    **搜索步长按图片规模三档自适应**，本质是召回率与耗时的权衡：

    * ≤ 25 万像素（约 500×500）：步长 1，逐像素穷举。小图代价可接受，
      任意偏移的裁剪都能找回；
    * 25 万 ~ 100 万像素：步长 8，牺牲少量召回换取 64 倍加速；
    * > 100 万像素：步长直接放大到 ``BLOCK_STRIDE``，只对齐嵌入时用过的
      网格位置。此时只有"裁剪边界恰好对齐网格"的情况能命中，
      但大图逐像素扫描会耗时数十秒，完全不可接受。

    命中即返回，不继续扫描——各副本内容相同，找到一个就够了。
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < BLOCK_SIZE or height < BLOCK_SIZE:
        return None
    pixels = rgb.load()
    step = 1 if width * height <= 250_000 else 8
    y_origins = range(0, height - BLOCK_SIZE + 1, step)
    x_origins = range(0, width - BLOCK_SIZE + 1, step)
    if width * height > 1_000_000:
        y_origins = range(0, height - BLOCK_SIZE + 1, BLOCK_STRIDE)
        x_origins = range(0, width - BLOCK_SIZE + 1, BLOCK_STRIDE)
    for oy in y_origins:
        for ox in x_origins:
            # 传生成器而非列表：配合 decode_bits_from_pixels 的惰性读取，
            # 魔数不符时只会实际取出前 22 个像素，不会构造整块的副本。
            # 扫描量动辄上万次，这个差别决定了本函数可用与否。
            payload = decode_bits_from_pixels(
                pixels[x, y]
                for y in range(oy, oy + BLOCK_SIZE)
                for x in range(ox, ox + BLOCK_SIZE)
            )
            if payload:
                return payload
    return None
