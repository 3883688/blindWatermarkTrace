"""水印嵌入与提取的总编排层。

这是整个溯源系统的中枢：接口层把请求交给这里，这里再按顺序调度各个水印算法、
写入证据记录、维护运行期状态。

**依赖倒置是本模块的核心设计。** 所有具体算法都不 import 进来，而是通过
:class:`WatermarkOperations` 这个冻结数据类以回调形式注入（见
``trace_app/watermark/default_operations.py`` 的默认装配）。好处有三：

* 打破 ``service ↔ 算法实现`` 的循环依赖；
* 测试能用假算法替换任意一层，无需真的做 DCT 运算；
* 算法版本演进（v2→v3→v4）时本文件的编排骨架保持不变。

**两条互斥的嵌入链路**（由 ``robust_watermark_version`` 决定）：

* **v4 链路**：FFT 同步导频 + DCT 认证码字，两层即完成，
  并强制关闭小裁剪与点阵追踪层（它们的信号会干扰 v4 的同步检测）；
* **传统链路**：鲁棒水印 → 频域层 → 编码层 → 小裁剪层 → 点阵层 → LSB 载荷，
  层层叠加，逐级增强不同攻击场景下的存活率。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException, UploadFile
from PIL import Image

from trace_app.config import Settings
from trace_app.database.repositories import Repository
from trace_app.imaging.output import WatermarkedOutput
from trace_app.runtime import Runtime


@dataclass(frozen=True, slots=True)
class WatermarkOperations:
    """水印服务所需的全部外部能力，以回调集合的形式一次性注入。

    ``frozen=True`` 保证装配完成后不可篡改，``slots=True`` 省内存并防止误加字段。
    字段按"参数解析 → 嵌入 → 落盘 → 提取"的调用顺序分组排列。
    """

    # ---- 通用工具：目录、解析、格式化 ----
    ensure_dirs: Callable[[], None]
    # 把完整证据 UUID 拆成首尾两段（对外只露片段，防拼接盗用）
    evidence_uuid_fields: Callable[[str], dict[str, str]]
    load_upload_image: Callable[[UploadFile], Any]
    normalize_mode: Callable[[str], str]
    apply_visible_copyright: Callable[..., Image.Image]
    # 表单字段都是字符串，需用这两个函数解析并夹紧到安全区间
    parse_bool: Callable[[Any], bool]
    clamp_float: Callable[..., float]
    now_text: Callable[[], str]
    mode_label: Callable[[str], str]

    # ---- 强度与版本换算 ----
    # 保真度（越高越保画质）反推为嵌入强度（越高越抗攻击），二者是此消彼长关系
    fidelity_to_strength: Callable[[str], float]
    robust_strength_to_scale: Callable[[Any], float]
    normalize_robust_watermark_version: Callable[[Any], int]

    # ---- V4 专用：配置与认证码 ----
    v4_config: Callable[[], Any]
    # 由 trace_id + 密钥算 HMAC 认证标签，是 v4 真正嵌进图里的净荷
    v4_authentication_tag: Callable[[str, str], bytes]
    # v3 的同类物，算法不同、长度不同，两者不可混用
    auth_code_from_trace: Callable[[str, str], bytes]
    state_value: Callable[[str], Any]

    # ---- 小裁剪追踪层参数 ----
    small_crop_strength_to_scale: Callable[[Any], float]
    normalize_small_crop_density: Callable[[str], str]

    # ---- 嵌入算子（按链路分两组）----
    # v4 链路：先打同步导频，再嵌 DCT 认证码字
    embed_v4_pilot: Callable[..., Image.Image]
    encode_v4_codeword: Callable[[bytes], Any]
    embed_v4_codeword: Callable[..., Image.Image]
    # 传统链路：三个鲁棒水印版本 + 四个叠加层 + LSB 明文载荷
    embed_robust_watermark: Callable[..., Image.Image]
    embed_robust_watermark_v2: Callable[..., Image.Image]
    embed_robust_watermark_v3: Callable[..., Image.Image]
    apply_frequency_layers: Callable[[Image.Image, str], Image.Image]
    apply_code_layer: Callable[..., Image.Image]
    apply_small_crop_trace_layer: Callable[..., Image.Image]
    apply_dot_matrix_trace_layer: Callable[..., Image.Image]
    embed_lsb: Callable[[Image.Image, dict[str, Any]], Image.Image]

    # ---- 落盘与证据固化 ----
    # 读取源图的 JPEG 采样率，输出时沿用，避免二次压缩引入额外色度损失
    jpeg_subsampling: Callable[[Image.Image], int]
    save_watermarked_output: Callable[..., WatermarkedOutput]
    save_thumbnail: Callable[[Image.Image, Path], None]
    # 特征索引：给"图被改过但还认得出"的视觉匹配兜底用，v4 用独立格式
    save_record_feature_index: Callable[[Image.Image, str], str]
    save_record_feature_index_v4: Callable[[Image.Image, str], str]
    # 文件级与像素级两套哈希：前者验字节完整性，后者在重新编码后仍然可比
    path_md5: Callable[[Path], str]
    path_sha256: Callable[[Path], str]
    image_content_sha256: Callable[[Image.Image], str]
    layer_scores_for_image: Callable[[Image.Image, str], dict[str, float]]

    # ---- 提取侧：入口与各级检测器 ----
    # 字节指纹直查，命中就无需跑任何检测算法（最快路径）
    matched_file_fingerprint: Callable[
        [bytes, list[dict[str, Any]]], dict[str, Any] | None
    ]
    load_image_from_bytes: Callable[[bytes], Image.Image]
    load_image_from_url: Callable[[str], Image.Image]
    v4_candidate_records: Callable[[list[dict[str, Any]]], Any]
    detect_v4_watermark: Callable[
        [Image.Image, Any, list[dict[str, Any]]], dict[str, Any] | None
    ]
    extract_full_lsb: Callable[[Image.Image], dict[str, Any] | None]
    extract_block_lsb: Callable[[Image.Image], dict[str, Any] | None]
    is_registered_original_image: Callable[
        [Image.Image, list[dict[str, Any]]], bool
    ]
    should_run_frequency_fallbacks: Callable[[Image.Image], bool]
    should_run_visual_match_fallback: Callable[
        [Image.Image, list[dict[str, Any]]], bool
    ]
    detect_dot_matrix_trace: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    detect_aligned_authenticated_watermark: Callable[..., dict[str, Any] | None]
    detect_by_visual_match: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    detect_small_crop_trace: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    detect_watermark_code: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    detect_robust_watermark: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    detect_by_residual_match: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    detect_visible_copyright: Callable[
        [Image.Image, list[dict[str, Any]]], dict[str, Any] | None
    ]
    with_evidence_fields: Callable[..., dict[str, Any]]
    # 检测流水线本身：按序调用上面各检测器，命中即停
    watermark_detection_pipeline: Callable[..., dict[str, Any]]

    # ---- 常量：密钥与各层版本号 ----
    default_watermark_auth_key: str
    robust_version_v2: int
    robust_version_v3: int
    robust_version_v4: int
    robust_codec_v2: str
    robust_codec_v3: str
    small_trace_version: int
    dot_matrix_version: int
    code_watermark_version: int
    watermark_layers: dict[str, bool]


class WatermarkService:
    """水印业务服务：编排嵌入链路与提取流水线，并维护证据记录。"""

    def __init__(
        self,
        *,
        settings: Settings,
        repository: Repository,
        runtime: Runtime,
        operations: WatermarkOperations | None = None,
    ) -> None:
        """
        :param runtime: 进程级运行状态（数据库句柄、最近生成的溯源号等）。
        :param operations: 算法回调集合；允许为 ``None``，这样应用可以先建服务
            实例、稍后再装配算法，打破模块间的循环导入。
        """
        self.settings = settings
        self.repository = repository
        self.runtime = runtime
        self.operations = operations

    def _operations(self) -> WatermarkOperations:
        """取算法集合；未装配时明确报错，而不是抛 ``NoneType`` 属性错误。"""
        if self.operations is None:
            raise RuntimeError("Watermark operations are not configured")
        return self.operations

    def _remember_generated_trace(self, trace_id: str) -> None:
        """把刚生成的溯源号记入最近列表（最新在前，只保留 24 条）。

        这份列表供提取端做"近期生成优先"的候选排序：刚嵌完就来验证是最常见的
        使用场景，先试最近的能显著缩短检测耗时。用切片删除而非 ``pop``，
        一次性截断，避免列表随运行时间无限增长。
        """
        self.runtime.generated_trace_ids.insert(0, trace_id)
        del self.runtime.generated_trace_ids[24:]

    async def embed(
        self,
        *,
        file: UploadFile,
        owner_user_id: int | None,
        user_id: str,
        mode: str,
        copyright_enabled: str,
        copyright_text: str,
        copyright_opacity: str,
        copyright_complexity: str,
        copyright_irregular_enabled: str,
        copyright_prominent_corner_enabled: str,
        fidelity_level: str,
        robust_watermark_strength: str,
        robust_watermark_version: str,
        small_crop_trace_enabled: str,
        small_crop_trace_strength: str,
        small_crop_trace_density: str,
        dot_matrix_trace_enabled: str,
        dot_matrix_trace_strength: str,
    ) -> dict[str, Any]:
        """执行一次完整嵌入，返回落库后的完整证据记录。

        :param owner_user_id: 记录归属人；``None``（匿名调用）时回落到管理员账号。
        :param user_id: 业务侧的用户标识字符串，会写进 LSB 载荷随图走。
        :return: 含全部内部字段的记录字典；接口层负责裁剪后再对外。

        整体分七步：生成标识 → 读图存原件 → 打明水印 → 解析强度参数 →
        按版本走嵌入链路 → 落盘与算哈希 → 组装记录并入库。
        """
        op = self._operations()
        op.ensure_dirs()
        # 三个标识各司其职，不可互相替代：
        #   image_id     —— 内部主键，用于拼文件名
        #   trace_id     —— 对外溯源号，也是隐水印真正编码进图里的内容
        #   evidence_uuid—— 证据链编号，拆成首尾两段对外展示
        image_id = uuid.uuid4().hex
        trace_id = f"TR-{uuid.uuid4().hex[:16].upper()}"
        evidence_uuid = uuid.uuid4().hex.upper()
        evidence_fields = op.evidence_uuid_fields(evidence_uuid)
        # 只取文件名部分，剥掉客户端可能带来的路径成分，防目录穿越。
        safe_name = Path(file.filename or "image.png").name
        # 文件名统一以 image_id 打头，保证同名上传不互相覆盖。
        original_path = self.settings.original_dir / f"{image_id}-{safe_name}"
        output_base = self.settings.watermarked_dir / f"{image_id}-watermarked"
        thumbnail_path = self.settings.thumbnail_dir / f"{image_id}-thumb.png"

        uploaded_size = getattr(file, "size", None)
        image = await op.load_upload_image(file)
        # 记住源格式与色度采样：JPEG 源要按原采样率输出，否则二次压缩会额外
        # 损伤色度通道，把嵌在其中的水印信号一并抹掉。
        source_format = str(image.format or "").upper()
        source_subsampling = (
            op.jpeg_subsampling(image) if source_format == "JPEG" else 0
        )
        image.save(original_path)
        # 目标体积以客户端上报的大小为准，拿不到再退回落盘后的实际大小；
        # 该值用于让输出图的体积贴近原图，避免"加了水印后文件明显变大"。
        source_size = (
            uploaded_size
            if isinstance(uploaded_size, int) and uploaded_size > 0
            else original_path.stat().st_size
        )
        normalized_mode = op.normalize_mode(mode)
        # 明水印必须最先打：它是像素级的可见改动，必须成为后续所有隐水印层的
        # 载体基底。反过来先嵌隐水印再叠明水印，会把隐水印信号覆盖掉一部分。
        visible = op.apply_visible_copyright(
            image,
            op.parse_bool(copyright_enabled),
            copyright_text,
            op.clamp_float(copyright_opacity, 0.16, 0.02, 0.90),
            copyright_complexity,
            op.parse_bool(copyright_irregular_enabled),
            op.parse_bool(copyright_prominent_corner_enabled),
        )
        created_at = op.now_text()
        # payload 是要写进 LSB 的**明文载荷**（传统链路），同时也是数据库记录的
        # 公共前缀。字段刻意保持精简：LSB 容量有限，且这部分内容一旦图片被
        # 重编码就会丢失，真正的抗攻击信息靠鲁棒水印层承载。
        payload = {
            "id": image_id,
            "trace_id": trace_id,
            **evidence_fields,
            "user_id": user_id,
            "mode": normalized_mode,
            "mode_label": op.mode_label(normalized_mode),
            "created_at": created_at,
        }
        strength_scale = op.fidelity_to_strength(fidelity_level)
        robust_strength = op.robust_strength_to_scale(robust_watermark_strength)
        robust_version = op.normalize_robust_watermark_version(
            robust_watermark_version
        )
        # v3/v4 嵌入的不是溯源号本身，而是由溯源号 + 服务端密钥算出的认证码。
        # 这样即便攻击者猜到编码方式，没有密钥也伪造不出能通过校验的水印，
        # 检测端因此可以区分"真水印"与"精心构造的噪声"。v2 及更早版本直接
        # 嵌 trace_id，无此保护，故 robust_auth_code 保持 None。
        robust_auth_code = None
        v4_config = op.v4_config()
        if robust_version == op.robust_version_v4:
            try:
                robust_auth_code = op.v4_authentication_tag(
                    trace_id, op.default_watermark_auth_key
                )
            except (TypeError, ValueError) as exc:
                # 密钥缺失或格式非法属于**服务端配置问题**，因此是 503 而非 400：
                # 不是用户请求的错，重试也没用，需要运维介入。
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        elif robust_version == op.robust_version_v3:
            try:
                robust_auth_code = op.auth_code_from_trace(
                    trace_id, op.default_watermark_auth_key
                )
            except ValueError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        # 空串 = 前端未表态，采用服务端全局默认开关；
        # 非空 = 用户显式指定，以用户为准。所以这里不能简单写 parse_bool。
        small_crop_enabled = (
            op.state_value("small_crop_trace_default_enabled")
            if str(small_crop_trace_enabled or "").strip() == ""
            else op.parse_bool(small_crop_trace_enabled)
        )
        small_crop_strength = op.small_crop_strength_to_scale(
            small_crop_trace_strength
        )
        small_crop_density = op.normalize_small_crop_density(
            small_crop_trace_density
        )
        dot_matrix_enabled = op.parse_bool(dot_matrix_trace_enabled)
        dot_matrix_strength = op.clamp_float(
            dot_matrix_trace_strength, 0.85, 0.0, 1.0
        )
        # ===== 嵌入链路分岔：v4 与传统链路互斥 =====
        if robust_version == op.robust_version_v4:
            # v4 强制关闭这两层：它们的周期性纹理会污染 FFT 频谱，
            # 使同步导频的峰值检测失准，反而降低 v4 的整体检出率。
            small_crop_enabled = False
            dot_matrix_enabled = False
            try:
                # 两步嵌入，顺序不可换：
                #   1. embed_v4_pilot   —— 先打 FFT 同步导频，供提取端估计
                #      缩放/旋转/平移，把被改动过的图"摆正"回原始几何；
                #   2. embed_v4_codeword—— 再把纠错编码后的认证码字写进 DCT 系数。
                # 导频必须先在未编码的干净图上打，否则码字信号会削弱导频峰值。
                watermarked = op.embed_v4_codeword(
                    op.embed_v4_pilot(visible, v4_config),
                    op.encode_v4_codeword(robust_auth_code),
                    v4_config,
                )
            except (TypeError, ValueError) as exc:
                # 这里是 400：通常因为图太小、容不下 v4 所需的分块数量，
                # 属于用户输入问题，换张大图即可。
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            # ---- 传统链路：多层叠加，每层针对不同攻击场景 ----
            # 第 1 层：鲁棒水印（抗压缩/缩放的主力层），按版本选算法
            if robust_version == op.robust_version_v3:
                robust = op.embed_robust_watermark_v3(
                    visible, robust_auth_code, robust_strength
                )
            elif robust_version == op.robust_version_v2:
                robust = op.embed_robust_watermark_v2(
                    visible, trace_id, robust_strength
                )
            else:
                robust = op.embed_robust_watermark(
                    visible, trace_id, robust_strength
                )
            # 第 2 层：频域层，在多个频段重复写入，抗滤波与轻度模糊
            frequency_marked = op.apply_frequency_layers(robust, trace_id)
            # 第 3 层：编码层，空域可解码的短码，抗色彩调整
            code_marked = op.apply_code_layer(
                frequency_marked, trace_id, strength_scale
            )
            # 第 4 层（可选）：小裁剪追踪，把短码密铺全图，
            # 使得只截取一小块也能还原出溯源号
            small_crop_marked = (
                op.apply_small_crop_trace_layer(
                    code_marked,
                    trace_id,
                    small_crop_strength,
                    small_crop_density,
                    strength_scale,
                )
                if small_crop_enabled
                else code_marked
            )
            # 第 5 层（可选）：点阵追踪，类似打印机黄点，抗翻拍与屏摄
            dot_matrix_marked = (
                op.apply_dot_matrix_trace_layer(
                    small_crop_marked, trace_id, dot_matrix_strength
                )
                if dot_matrix_enabled
                else small_crop_marked
            )
            # 第 6 层：LSB 明文载荷，最脆弱但信息量最大，
            # 原图未被重编码时可一次性读出全部元信息（最快检测路径）
            watermarked = op.embed_lsb(dot_matrix_marked, payload)

        # ===== 落盘：格式、体积与采样率都尽量贴近源图 =====
        saved_output = op.save_watermarked_output(
            watermarked,
            output_base,
            # 只有 v4 才允许输出 JPEG：它的抗压缩能力经过针对性设计，
            # 传统链路的 LSB 与点阵层在 JPEG 有损压缩下会被摧毁，因此一律转无损。
            jpeg_output=(
                source_format == "JPEG"
                and robust_version == op.robust_version_v4
            ),
            source_size=source_size,
            jpeg_subsampling=source_subsampling,
        )
        # 关键：回读保存函数返回的图像对象。JPEG 编码是有损的，落盘后的像素
        # 与内存里的 watermarked 已经不同；后续的哈希与特征索引都必须基于
        # **实际写出的那张图**，否则提取端比对必然失败。
        output_path = saved_output.path
        watermarked = saved_output.image
        op.save_thumbnail(watermarked, thumbnail_path)
        # 特征索引供视觉匹配兜底：图被改到水印全废时，仍可靠图像特征找回来源。
        feature_index_path = (
            op.save_record_feature_index_v4(watermarked, image_id)
            if robust_version == op.robust_version_v4
            else op.save_record_feature_index(watermarked, image_id)
        )
        # 六个哈希构成完整证据链：
        #   *_file_md5 / *_file_sha256 —— 文件字节级，证明文件未被改动
        #   *_image_sha256             —— 像素级，容器格式变了但画面没变时仍能匹配
        # 原图与成品图各留一份，用于事后证明"这张成品确实由这张原图生成"。
        original_file_md5 = op.path_md5(original_path)
        watermarked_file_md5 = op.path_md5(output_path)
        original_file_sha256 = op.path_sha256(original_path)
        watermarked_file_sha256 = op.path_sha256(output_path)
        original_image_sha256 = op.image_content_sha256(image)
        watermarked_image_sha256 = op.image_content_sha256(watermarked)

        # ===== 组装入库记录 =====
        # 除业务元信息外，把**所有嵌入参数**一并存档（强度、版本、各层开关）。
        # 这是提取端能正确解码的前提：不同强度/版本的解码阈值不同，
        # 也是事后复现和司法举证时还原当时处理过程的依据。
        record = {
            **payload,
            "name": safe_name,
            "image_width": image.width,
            "image_height": image.height,
            "size": f"{original_path.stat().st_size / 1024 / 1024:.1f} MB",
            "status": "保护中",
            "confidence": 98,
            "original_url": f"/uploads/originals/{original_path.name}",
            "download_url": f"/uploads/watermarked/{output_path.name}",
            "thumbnail_url": f"/uploads/thumbnails/{thumbnail_path.name}",
            "feature_index_path": feature_index_path,
            "original_file_md5": original_file_md5,
            "watermarked_file_md5": watermarked_file_md5,
            "original_file_sha256": original_file_sha256,
            "watermarked_file_sha256": watermarked_file_sha256,
            "original_image_sha256": original_image_sha256,
            "watermarked_image_sha256": watermarked_image_sha256,
            "copyright_enabled": op.parse_bool(copyright_enabled),
            "copyright_text": copyright_text.strip() or "© QQ:757675150",
            "copyright_opacity": op.clamp_float(
                copyright_opacity, 0.16, 0.02, 0.90
            ),
            "copyright_complexity": copyright_complexity,
            "copyright_irregular_enabled": op.parse_bool(
                copyright_irregular_enabled
            ),
            "copyright_prominent_corner_enabled": op.parse_bool(
                copyright_prominent_corner_enabled
            ),
            "fidelity_level": op.clamp_float(fidelity_level, 0.75, 0.0, 1.0),
            "watermark_strength_scale": round(strength_scale, 4),
            "robust_watermark_strength": round(robust_strength, 4),
            "robust_watermark_version": robust_version,
            "robust_watermark_codec": (
                v4_config.codec
                if robust_version == op.robust_version_v4
                else op.robust_codec_v3
                if robust_version == op.robust_version_v3
                else op.robust_codec_v2
                if robust_version == op.robust_version_v2
                else "legacy_robust_64"
            ),
            "robust_auth_code": robust_auth_code.hex()
            if robust_auth_code
            else None,
            "small_crop_trace_enabled": small_crop_enabled,
            "small_crop_trace_strength": small_crop_strength,
            "small_crop_trace_density": small_crop_density,
            "small_crop_trace_version": op.small_trace_version
            if small_crop_enabled
            else None,
            "dot_matrix_trace_enabled": dot_matrix_enabled,
            "dot_matrix_trace_strength": dot_matrix_strength,
            "dot_matrix_trace_version": op.dot_matrix_version
            if dot_matrix_enabled
            else None,
            "robust_watermark": True,
            # 以下三项对 v4 都是"空"值：v4 不走编码层、只有两个自有层、
            # 也不做逐层打分（它有自己的置信度模型）。
            "watermark_code_version": None
            if robust_version == op.robust_version_v4
            else op.code_watermark_version,
            "watermark_layers": (
                {"dct_authenticated": True, "fft_sync": True}
                if robust_version == op.robust_version_v4
                else op.watermark_layers
            ),
            "layer_scores": (
                {}
                if robust_version == op.robust_version_v4
                else op.layer_scores_for_image(watermarked, trace_id)
            ),
        }
        # 匿名嵌入回落到管理员名下，保证每条证据都有归属人；
        # 管理员账号也查不到时才允许 owner 为空（数据库尚未初始化的极端情况）。
        if owner_user_id is None:
            admin = self.repository.get_user_by_username(self.settings.admin_user)
            owner_user_id = None if admin is None else int(admin["id"])
        self.repository.add_record(record, owner_user_id=owner_user_id)
        self.repository.record_watermark_generation()
        self._remember_generated_trace(trace_id)
        return record

    def extract_image(
        self,
        image: Image.Image,
        records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """对一张图跑完整检测流水线，返回命中的证据记录。

        :param records: 候选记录集；传 ``None`` 则现查数据库。调用方（如
            :meth:`extract_upload`）若已经查过，务必传进来复用——**同一次提取
            全程只读一次库**，否则各级检测器会反复查询，成本随记录数放大。
        :raises HTTPException: 所有检测器都没结论时 404。

        本方法自身不含检测逻辑，只做两件事：准备候选集，然后把十来个检测器
        以闭包形式打包交给流水线。闭包的作用是把 ``current_records`` 提前绑定进去，
        让流水线只需按统一签名 ``(image) -> result | None`` 依次调用，
        不必关心每个检测器各自需要什么额外上下文。

        检测器的排列顺序即**优先级**，从"最可靠且最快"到"最兜底"：
        V4 认证水印 → LSB 明文 → 点阵/几何对齐 → 视觉匹配 → 小裁剪 →
        编码层 → 鲁棒水印 → 残差匹配 → 明水印。流水线命中第一个即返回。
        """
        op = self._operations()
        current_records = (
            self.repository.read_records() if records is None else records
        )
        # v4 候选单独预处理一遍（解析出各记录的认证码等），避免在检测循环里重复算。
        v4_candidates = op.v4_candidate_records(current_records)
        return op.watermark_detection_pipeline(
            image,
            records=current_records,
            v4_candidates=v4_candidates,
            detect_v4_watermark=lambda current_image, candidates: (
                op.detect_v4_watermark(
                    current_image, candidates, current_records
                )
            ),
            extract_full_lsb=op.extract_full_lsb,
            extract_block_lsb=op.extract_block_lsb,
            is_registered_original_image=lambda current_image: (
                op.is_registered_original_image(current_image, current_records)
            ),
            should_run_frequency_fallbacks=op.should_run_frequency_fallbacks,
            should_run_visual_match_fallback=lambda current_image: (
                op.should_run_visual_match_fallback(
                    current_image, current_records
                )
            ),
            detect_dot_matrix_trace=lambda current_image: (
                op.detect_dot_matrix_trace(current_image, current_records)
            ),
            detect_aligned_authenticated_watermark=lambda current_image, **kwargs: (
                op.detect_aligned_authenticated_watermark(
                    current_image, current_records, **kwargs
                )
            ),
            detect_by_visual_match=lambda current_image: (
                op.detect_by_visual_match(current_image, current_records)
            ),
            detect_small_crop_trace=lambda current_image: (
                op.detect_small_crop_trace(current_image, current_records)
            ),
            detect_watermark_code=lambda current_image: (
                op.detect_watermark_code(current_image, current_records)
            ),
            detect_robust_watermark=lambda current_image: (
                op.detect_robust_watermark(current_image, current_records)
            ),
            detect_by_residual_match=lambda current_image: (
                op.detect_by_residual_match(current_image, current_records)
            ),
            detect_visible_copyright=lambda current_image: (
                op.detect_visible_copyright(current_image, current_records)
            ),
            # 无论命中与否都回调计数，看板的"检测成功率"由此累计。
            record_detection_result=self.repository.record_detection_result,
            with_evidence_fields=op.with_evidence_fields,
            now_text=op.now_text,
            mode_label=op.mode_label,
            layer_scores_for_image=op.layer_scores_for_image,
            not_found_error=lambda: HTTPException(
                status_code=404, detail="未检测到可识别的隐式水印"
            ),
            watermark_layers=op.watermark_layers,
            state_value=op.state_value,
        )

    async def extract_upload(self, file: UploadFile) -> dict[str, Any]:
        """从上传文件提取水印。

        比 :meth:`extract_image` 多一条**快速路径**：先拿原始字节算文件指纹去
        查表。如果用户上传的就是当初下载的那份文件（未经任何改动），指纹直接
        命中，连解码图片都省了。这是实际使用中最高频的场景。

        指纹未命中才解码成图像走完整流水线，并把已经查出来的 ``records``
        传下去复用，避免第二次查库。
        """
        op = self._operations()
        content = await file.read()
        records = self.repository.read_records()
        fingerprint_match = op.matched_file_fingerprint(content, records)
        if fingerprint_match:
            # 快速路径命中也要计入检测统计，否则成功率会被低估。
            self.repository.record_detection_result(True)
            return fingerprint_match
        image = op.load_image_from_bytes(content)
        return self.extract_image(image, records=records)

    def extract_url(self, url: str) -> dict[str, Any]:
        """从 URL 下载图片并提取水印。

        没有字节指纹快速路径——远端返回的字节通常已被 CDN 或平台重新编码，
        与存档文件不会逐字节相同，查指纹几乎必然落空，不如直接进流水线。
        """
        return self.extract_image(self._operations().load_image_from_url(url))
