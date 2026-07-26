"""水印成品的编码与落盘。

职责很窄：把已经嵌好水印的 :class:`PIL.Image.Image` 写成磁盘文件，并把**实际
写出去的那张图**交还给调用方。入口只有一个——
:meth:`trace_app.watermark.service.WatermarkService.embed` 在嵌入链路的最后一步。
读入侧的对应模块是 ``trace_app/imaging/io.py``。

**落盘这一步直接决定水印能否存活**，所以值得单独成模块：

* **格式选择**：只有 v4 链路才允许输出 JPEG。v4 的 DCT 认证码字是针对有损压缩
  设计的；传统链路的 LSB 层与点阵层在 JPEG 量化下会被彻底摧毁，因此一律转 PNG
  无损保存。这个判断在调用方，本模块只接收 ``jpeg_output`` 这个布尔结论。
* **色度采样率沿用源图**：源图是 4:2:0，输出也必须是 4:2:0。采样率一旦不一致，
  图片再被平台二次压缩时色度通道会经历额外的重采样，把嵌在其中的水印信号抹掉。
* **体积贴近源图**：在 quality 90~95 区间里挑**能塞进体积预算的最高质量**，
  既不让成品明显大于原图，又尽量少丢信息。
* **回读落盘结果**：JPEG 编码有损，写出去的像素和内存里的已经不同。后续算哈希、
  建特征索引、生成缩略图都必须基于实际写出的那张图，否则提取端拿到的图与登记的
  特征对不上，检测必然落空。这就是 :class:`WatermarkedOutput` 要带回 ``image``
  字段的原因。
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image, JpegImagePlugin


# quality 下限 90：再低，DCT 量化步长会大到开始吃掉 v4 嵌进系数里的码字幅度。
# 上限 95：95 以上体积快速膨胀，肉眼画质与水印存活率都几乎没有额外收益。
JPEG_MIN_QUALITY = 90
JPEG_MAX_QUALITY = 95
# 体积预算 = 源图字节数 × 1.25。留 25% 余量是因为嵌入水印本身引入了额外的高频
# 细节，同 quality 下必然比原图大一些；把预算卡死在 100% 会逼着质量无谓地往下掉。
JPEG_TARGET_RATIO = 1.25
# PIL 认得的三档色度采样：0 = 4:4:4，1 = 4:2:2，2 = 4:2:0。其余取值一律按 0 处理。
JPEG_SUBSAMPLING_VALUES = (0, 1, 2)
JpegEncoder = Callable[[Image.Image, int], bytes]


@dataclass(frozen=True, slots=True)
class WatermarkedOutput:
    """一次落盘的结果。

    :param path: 实际写出的文件路径。调用方只给不带后缀的基名，后缀由输出格式决定。
    :param image: **从磁盘回读的图像**，不是传进来的那个对象。JPEG 有损，二者像素
        不同；一切基于成品图的后续计算（image 级哈希、视觉特征索引、缩略图）
        都必须用这一份，否则登记下来的特征与用户手里的文件对不上。
    :param quality: 实际采用的 JPEG quality；PNG 输出时为 ``None``。

    ``frozen=True`` 只防止字段被重新绑定，``image`` 指向的 PIL 对象本身仍然可变，
    调用方不应就地修改它。
    """

    path: Path
    image: Image.Image
    quality: int | None


def _normalize_jpeg_subsampling(value: object) -> int:
    """把任意来源的 subsampling 值收敛到 PIL 支持的 0/1/2，其余一律取 0。

    用 ``type(value) is int`` 而不是 ``isinstance``：``bool`` 是 ``int`` 的子类，
    ``True`` 会被当成 1（4:2:2）传给编码器，这显然不是调用方的本意。严格类型判断
    把 ``True``、``None``、``"keep"``、越界整数统统挡在外面。

    兜底值取 0 而非 2：4:4:4 完全不做色度抽样，色度通道信息保留最全。在"源图采样率
    未知"时这是对水印最安全的选择——宁可文件大一些，也不能先把信号丢掉。
    """
    return (
        value
        if type(value) is int and value in JPEG_SUBSAMPLING_VALUES
        else 0
    )


def jpeg_subsampling(image: Image.Image) -> int:
    """读出一张图原本的 JPEG 色度采样档位。

    :return: 0（4:4:4）、1（4:2:2）或 2（4:2:0）；非 JPEG 图，或采样布局不落在这
        三档内（如 L 灰度、CMYK 的 JPEG），返回 0。

    ``JpegImagePlugin.get_sampling`` 对非 JPEG 图像会返回 -1 之类的哨兵值，必须经
    :func:`_normalize_jpeg_subsampling` 过一道才敢用。

    调用点在 embed 链路开头：源图是 JPEG 时先把采样率记下来，落盘时原样沿用，
    免得二次压缩时色度通道被重新抽样，连带把嵌在里面的水印信号一起抹掉。
    """
    return _normalize_jpeg_subsampling(JpegImagePlugin.get_sampling(image))


def encode_jpeg(
    image: Image.Image,
    quality: int,
    *,
    subsampling: int = 0,
) -> bytes:
    """把图编码成 JPEG 字节串（只在内存里，不落盘）。

    :param subsampling: 期望的色度采样档位，非法值自动归 0。
    :return: 完整的 JPEG 字节。

    三个固定参数都是有意为之：

    * ``optimize=True`` 重算 Huffman 表，同画质下体积更小。熵编码是无损的，
      不改变解码出来的像素，因此对水印信号零影响，属于白拿的收益；
    * ``progressive=True`` 渐进式扫描，同样只改变字节的组织方式，不动像素值；
    * 非 RGB 一律先 ``convert("RGB")``。JPEG 不支持 alpha，直接存 RGBA 会报错；
      显式统一通道布局也保证提取端看到的排列与嵌入端一致。
    """
    buffer = BytesIO()
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    rgb_image.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling=_normalize_jpeg_subsampling(subsampling),
    )
    return buffer.getvalue()


def encode_adaptive_jpeg(
    image: Image.Image,
    source_size: int,
    *,
    subsampling: int = 0,
    encoder: JpegEncoder | None = None,
) -> tuple[bytes, int]:
    """在质量区间内自适应挑一档，产出尽量不超过体积预算的 JPEG。

    :param source_size: 源图字节数，体积预算 = ``source_size * JPEG_TARGET_RATIO``。
    :param subsampling: 色度采样档位；**只在使用默认编码器时生效**，注入了
        ``encoder`` 时由调用方自己负责传递。
    :param encoder: 可注入的编码函数 ``(image, quality) -> bytes``；测试用它绕开
        真实 JPEG 编码，直接构造出指定大小的内容来验证挑选逻辑。
    :return: ``(字节内容, 实际采用的 quality)``。
    :raises RuntimeError: 质量区间为空。上下界写死在常量里，实际不会发生，
        纯粹是防御性分支。

    **从高质量往低走**是关键：第一个塞得进预算的就返回，所以拿到的是"预算内的
    最高质量"，而不是"刚好达标的最低质量"。水印强度随量化步长增大而衰减，
    质量能高就不该低。

    整个区间都超预算时，返回 90 这一档并接受体积超标——**画质下限比体积约束更硬**。
    低于 90 的量化会开始吃掉 DCT 域里的码字，那还不如让文件大一点。

    默认路径把 ``convert("RGB")`` 提到循环外只做一次：最多要编码 6 个质量档，
    每档都转一次纯属浪费。注入自定义 ``encoder`` 时不做这个预转换，把图原样交出去，
    由 encoder 自行决定怎么处理。
    """
    if encoder is None:
        effective_image = image if image.mode == "RGB" else image.convert("RGB")

        def default_encoder(image: Image.Image, quality: int) -> bytes:
            """默认 JPEG 编码器：把外层的 ``subsampling`` 闭包进来。

            这样它就符合 ``(image, quality) -> bytes`` 的统一签名，
            质量搜索循环无需关心色度采样率从哪来。
            """
            return encode_jpeg(image, quality, subsampling=subsampling)

        effective_encoder = default_encoder
    else:
        effective_image = image
        effective_encoder = encoder

    target_size = max(1, int(source_size * JPEG_TARGET_RATIO))
    minimum_content: bytes | None = None
    for quality in range(JPEG_MAX_QUALITY, JPEG_MIN_QUALITY - 1, -1):
        content = effective_encoder(effective_image, quality)
        # 顺手留住最低档的编码结果，免得区间跑完后为了兜底再编一次。
        if quality == JPEG_MIN_QUALITY:
            minimum_content = content
        if len(content) <= target_size:
            return content, quality
    if minimum_content is None:
        raise RuntimeError("JPEG quality range is empty")
    return minimum_content, JPEG_MIN_QUALITY


def save_watermarked_output(
    image: Image.Image,
    output_base: Path,
    *,
    jpeg_output: bool,
    source_size: int,
    jpeg_subsampling: int = 0,
) -> WatermarkedOutput:
    """把水印成品写到磁盘，并回读出实际落盘的那张图。

    :param output_base: 不带后缀的目标路径，后缀由本函数按输出格式补齐。
    :param jpeg_output: 是否输出 JPEG。**只有"源图本身是 JPEG 且走 v4 链路"时调用方
        才会传 True**：传统链路的 LSB 层与点阵层扛不住有损压缩，必须走 PNG。
    :param source_size: 源文件字节数，用于算 JPEG 体积预算；PNG 分支忽略。
    :param jpeg_subsampling: 源图的色度采样档位，原样沿用，避免二次压缩时色度通道
        被重新抽样而抹掉水印信号。
    :return: :class:`WatermarkedOutput`，其中 ``image`` 是**从磁盘回读的图**。

    末尾那次"写完再读回来"是本函数存在的理由。JPEG 编码有损，磁盘上的像素与传进来的
    ``image`` 已经不是同一份数据；调用方接下来要算 image 级哈希、建视觉特征索引、
    生成缩略图，这些都必须基于提取端将来真正会看到的那张图，否则登记的特征与实际
    文件对不上，检测必然落空。

    ``loaded.copy()`` 不可省：``with`` 块退出后文件句柄关闭，继续持有 ``loaded``
    会拿到一个底层文件已失效的图像对象。

    PNG 分支本身是无损的，回读拿到的就是原像素，这一步看似多余——但让两条分支返回
    同构的结果，调用方就不必区分格式各写一套后处理，这点开销很划算。

    .. note::
       关键字参数 ``jpeg_subsampling`` 与模块级函数 :func:`jpeg_subsampling` 同名，
       在函数体内会遮蔽该函数。当前函数体没有调用它，所以不构成 bug，但后续在这里
       加代码时要留意。此处未做重命名。
    """
    quality = None
    if jpeg_output:
        path = output_base.with_suffix(".jpg")
        content, quality = encode_adaptive_jpeg(
            image,
            source_size,
            subsampling=jpeg_subsampling,
        )
        path.write_bytes(content)
    else:
        path = output_base.with_suffix(".png")
        image.save(path, format="PNG")
    with Image.open(path) as loaded:
        loaded.load()
        persisted = loaded.copy()
    return WatermarkedOutput(path=path, image=persisted, quality=quality)
