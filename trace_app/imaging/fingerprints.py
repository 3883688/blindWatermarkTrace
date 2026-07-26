"""文件与像素指纹：检测流水线的"原件快速命中"通道。

**这里的哈希全是精确哈希（MD5 / SHA-256），不是感知哈希。** 它们只回答
"是不是同一份东西"，容不下任何改动。模糊的、能抗改动的相似性匹配在
:mod:`trace_app.imaging.feature_matching` 里，两者分工不同。

**为什么值得为它单开一条路径。** 实际使用中最高频的场景是：用户把当初从本系统
下载的那张图原样传回来求证归属。这种情况下文件字节逐字节相同，一次哈希查表就能
给出 100% 置信度的结论，完全不必解码图片、更不必跑 LSB / 频域 / 点阵那一整套
检测。因此 ``WatermarkService.extract_upload`` 在进流水线之前先调
:func:`matched_file_fingerprint`，命中就直接返回。

**两级指纹，容忍度依次放宽：**

1. **文件字节指纹** ``file_md5`` + ``file_sha256`` —— 改一个字节就失配。
2. **像素内容指纹** :func:`image_content_sha256` —— 对解码后的 RGB 像素取哈希，
   能穿透"无损重新封装"：改 EXIF、剥掉元数据、PNG 换个压缩级别、
   JPEG 无损旋转回来……容器变了但像素没变，仍然命中。

再往下（画质有损、被裁剪、被重新压缩）就超出本模块能力，交给水印检测与感知匹配。

**为什么同时留 MD5 和 SHA-256。** 早期只存 SHA-256；后来加快速路径时补上 MD5，
用它做**查表键**（算得快），再用 SHA-256 做**确认**——单靠 MD5 的碰撞可能导致
错误归属，这是溯源系统不能接受的。旧记录里没有 MD5 字段，所以匹配逻辑必须
兼容"只有 SHA-256"的情况。

摘要一律输出**大写十六进制**，入库、比对两端都按这个约定，比对前统一 ``.upper()``。
"""

import hashlib
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from trace_app.imaging.io import load_image_from_bytes


def file_md5(content: bytes) -> str:
    """算文件字节的 MD5，大写十六进制。

    只用作**候选查表键**，命中后必须再用 SHA-256 确认，不单独作为归属依据。
    """
    return hashlib.md5(content).hexdigest().upper()


def path_md5(path: Path) -> str:
    """算磁盘文件的 MD5。

    嵌入时对**已落盘的文件**取指纹，而不是对内存里的图像对象重新编码后取——
    只有这样，存下来的摘要才等于用户之后下载到、并可能再传回来的那串字节。
    """
    return hashlib.md5(path.read_bytes()).hexdigest().upper()


def file_sha256(content: bytes) -> str:
    """算文件字节的 SHA-256，大写十六进制。这是最终确认用的强摘要。"""
    return hashlib.sha256(content).hexdigest().upper()


def path_sha256(path: Path) -> str:
    """算磁盘文件的 SHA-256，理由同 :func:`path_md5`。"""
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def image_content_sha256(image: Image.Image) -> str:
    """算**解码后像素**的 SHA-256，忽略文件容器层面的差异。

    :return: 大写十六进制摘要。

    统一 ``convert("RGB")`` 后再取字节：把调色板图、灰度图、带 alpha 的 RGBA
    都归一到同一种表示，同一幅画面不会因为存储模式不同而算出两个摘要。

    摘要前先喂入 ``"宽x高:RGB:"`` 这个前缀，是为了**绑定尺寸**。
    ``tobytes()`` 只吐出扁平的像素流，不含形状信息——不加前缀的话，
    100×200 和 200×100 两张不同的图有可能得到同一串字节，从而误判为同一张。
    前缀相当于给哈希做了域分隔。
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    digest = hashlib.sha256()
    # 前缀绑定尺寸与色彩模式：tobytes() 本身不带形状信息，
    # 不加前缀时 100x200 与 200x100 可能撞出同一串字节
    digest.update(f"{width}x{height}:RGB:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest().upper()


def matched_file_fingerprint(
    content: bytes,
    *,
    read_records: Callable[[], list[dict[str, Any]]],
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
    now_text: Callable[[], str],
    watermark_layers: list[str],
    file_md5_fn: Callable[[bytes], str] | None = None,
    file_sha256_fn: Callable[[bytes], str] | None = None,
    image_content_sha256_fn: Callable[[Image.Image], str] | None = None,
    load_image_from_bytes_fn: Callable[[bytes], Image.Image] | None = None,
) -> dict[str, Any] | None:
    """拿上传的原始字节去比对全部记录的指纹，命中就直接给出溯源结论。

    :param content: 上传文件的原始字节。
    :param read_records: 读取全部溯源记录的回调。
    :param with_evidence_fields: 给结果补齐通用证据字段的回调。
    :param now_text: 取当前时间字符串的回调。
    :param watermark_layers: 记录里没有 ``watermark_layers`` 时使用的默认层列表。
    :return: 命中返回证据字典，否则 ``None``（调用方随后走完整检测流水线）。

    **两趟扫描，先字节后像素。** 第一趟遍历全部记录比文件字节指纹，
    整趟走完没命中，第二趟才比像素指纹。这个顺序不能合并成一趟：
    字节相同是更强的证据（连元数据都一致），应当优先于"仅像素相同"的结论，
    哪怕后者出现在记录列表更靠前的位置。

    **每条记录有两份文件要比**：``original``（用户上传的原图）和
    ``watermarked``（系统产出的带水印图）。两者命中都算成功归属，
    结果里用 ``matched_file_type`` 和 ``matched_file_url`` 区分是哪一份。
    原图也算命中，是因为持有原图同样能证明这条记录的归属关系。

    **字节匹配的两种形态：**

    * ``file_md5_sha256`` —— 新记录。MD5 选中候选，SHA-256 确认。
      两者都要过，是因为单靠 MD5 的碰撞会造成错误归属，
      而错误归属在溯源系统里比漏判严重得多。
    * ``file_sha256`` —— 旧记录（入库时还没有 MD5 字段）。仅凭 SHA-256 判定。

    **像素匹配** ``image_pixels`` 处理"重新封装但画面没变"的情况。这一趟才需要
    解码图片，且解码结果只算一次、缓存复用；一条带像素指纹的记录都没有时，
    连解码都不会发生。解码失败（上传的根本不是图片）直接返回 ``None``——
    后面所有比对都依赖这张图，继续也没有意义。

    命中一律给 ``confidence: 100``：指纹相等是确定性结论，不存在概率成分。

    .. note::
        ``mode_label`` 与 ``status`` 三种命中方式都写死成"文件指纹一样"，
        包括 ``image_pixels`` 这种**文件字节其实不同、只是像素相同**的情况。
        真正的区分靠 ``matched_hash_type`` 字段。此处按现状描述，未作改动。

    .. note::
        某条记录若 MD5 字段存在但与上传值不符，即便它的 SHA-256 相符也不会命中
        （第二个分支要求 ``not stored_md5``）。正常数据下两个摘要不可能一个中
        一个不中，这是刻意收紧的写法：出现这种矛盾说明记录本身可疑，
        宁可漏判也不给结论。
    """
    hash_md5 = file_md5_fn or file_md5
    hash_file = file_sha256_fn or file_sha256
    hash_image = image_content_sha256_fn or image_content_sha256
    load_image = load_image_from_bytes_fn or load_image_from_bytes
    # 两个摘要都只算一次，后面对着所有记录复用
    md5_digest = hash_md5(content)
    sha256_digest = hash_file(content)
    records = read_records()

    def match_result(
        record: dict[str, Any],
        file_type: str,
        matched_hash_type: str,
        matched_hash: str,
        image_hash: str | None,
    ) -> dict[str, Any]:
        """按统一结构组装命中结果。

        :param file_type: ``original`` 或 ``watermarked``，决定回填哪个 URL。
        :param matched_hash_type: 命中方式，供上层区分证据强度。
        :param image_hash: 像素摘要；字节命中时为 ``None``（那一趟根本没解码图片）。

        ``layer_scores`` 留空字典：指纹命中是绕过水印检测拿到的结论，
        没有任何层的评分可填，但字段要保留，前端按同一套结构渲染所有检测结果。
        """
        return with_evidence_fields({
            "id": record.get("id"),
            "trace_id": record.get("trace_id"),
            "user_id": record.get("user_id"),
            "mode": "file_fingerprint",
            "mode_label": "文件指纹一样",
            "created_at": record.get("created_at"),
            "confidence": 100,
            "phash_match": False,
            "status": "文件指纹一样",
            "extracted_at": now_text(),
            "file_md5": md5_digest,
            "file_hash": sha256_digest,
            "image_hash": image_hash,
            "matched_hash": matched_hash,
            "matched_hash_type": matched_hash_type,
            "matched_file_type": file_type,
            # 命中的是原图就给原图链接，命中带水印图则给下载链接
            "matched_file_url": record.get(
                "original_url" if file_type == "original" else "download_url"
            ),
            "watermark_layers": record.get("watermark_layers", watermark_layers),
            "layer_scores": {},
        }, record)

    # ===== 第一趟：文件字节指纹。最强的证据，整趟走完才考虑像素 =====
    for record in records:
        for file_type in ("original", "watermarked"):
            # 统一大写后再比：历史数据里存在小写摘要
            stored_md5 = str(
                record.get(f"{file_type}_file_md5") or ""
            ).upper()
            stored_sha256 = str(
                record.get(f"{file_type}_file_sha256") or ""
            ).upper()
            # 新记录：MD5 选候选、SHA-256 做确认，必须两项同时成立
            if (
                stored_md5 == md5_digest
                and stored_sha256
                and stored_sha256 == sha256_digest
            ):
                return match_result(
                    record,
                    file_type,
                    "file_md5_sha256",
                    sha256_digest,
                    None,
                )
            # 旧记录：入库时还没有 MD5 字段，只能凭 SHA-256 判定
            if (
                not stored_md5
                and stored_sha256
                and stored_sha256 == sha256_digest
            ):
                return match_result(
                    record,
                    file_type,
                    "file_sha256",
                    sha256_digest,
                    None,
                )

    # ===== 第二趟：像素内容指纹，覆盖"重新封装但画面没变"的情况 =====
    # 惰性求值并缓存：没有任何记录带像素指纹时，一次解码都不会发生
    query_image_digest = None
    for record in records:
        for file_type in ("original", "watermarked"):
            stored_image_digest = str(
                record.get(f"{file_type}_image_sha256") or ""
            ).upper()
            if not stored_image_digest:
                continue
            try:
                if query_image_digest is None:
                    query_image_digest = hash_image(load_image(content))
            except Exception:
                # 上传的根本解不出图片。后面每一条比对都要用这张图，
                # 继续循环没有意义，直接放弃指纹路径交给上层处理
                return None
            if stored_image_digest == query_image_digest:
                return match_result(
                    record,
                    file_type,
                    "image_pixels",
                    query_image_digest,
                    query_image_digest,
                )
    return None
