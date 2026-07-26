"""水印嵌入 / 提取的 HTTP 接口层。

本模块只负责三件事，全部水印算法都在 :class:`WatermarkService` 中完成：

1. 声明 ``/api/watermark`` 下的 FastAPI 路由与表单字段；
2. 把请求原样转交给注入的水印服务（路由保持"薄委托"，不写业务分支）；
3. 对服务返回的内部结果做**字段白名单裁剪**，并把内部存储路径换成带签名、
   会过期的临时访问地址后再吐给前端。

第 3 点是安全边界：服务层返回的字典里含有磁盘路径、完整证据 UUID 等不应外泄的
字段，必须经过 :func:`_public_embed_response` / :func:`_public_extract_response`
过滤，绝不能直接 ``return`` 服务结果。
"""

from typing import Any

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.config import (
    DEFAULT_ROBUST_WATERMARK_STRENGTH,
    DEFAULT_ROBUST_WATERMARK_VERSION,
)
from trace_app.dependencies import get_optional_current_user, get_watermark_service
from trace_app.media import sign_media_url
from trace_app.watermark.service import WatermarkService

router = APIRouter(prefix="/api/watermark", tags=["watermark"])


# 嵌入接口允许返回给前端的字段白名单。
# 只保留证据链展示所需的元信息：证据 UUID 拆成首尾两段（避免完整 UUID 被拼接盗用）、
# 水印模式、置信度与鲁棒水印版本号；磁盘路径类字段一律不在此列。
EMBED_PUBLIC_FIELDS = (
    "id",
    "user_id",
    "trace_id",
    "evidence_uuid_head",
    "evidence_uuid_tail",
    "mode",
    "mode_label",
    "created_at",
    "status",
    "confidence",
    "robust_watermark_version",
)
# 提取接口的字段白名单。
# 比嵌入多两项：``extracted_at``（本次提取时间）与 ``phash_match``
# （感知哈希是否命中，用于区分"解出水印"与"仅靠指纹兜底匹配"两种结论）。
EXTRACT_PUBLIC_FIELDS = (
    "user_id",
    "trace_id",
    "evidence_uuid_head",
    "evidence_uuid_tail",
    "mode",
    "mode_label",
    "created_at",
    "extracted_at",
    "status",
    "confidence",
    "phash_match",
    "robust_watermark_version",
)


def _signed_access(request: Request, media_url: Any) -> str | None:
    """把内部媒体路径换成带 HMAC 签名、限时有效的访问地址。

    :param request: 当前请求，用于取出应用级的签名密钥与有效期配置。
    :param media_url: 服务层给出的内部地址，形如 ``/uploads/xxx.png``。
    :return: 可直接交给浏览器的签名地址；若入参不是受管的 ``/uploads/`` 路径
        则返回 ``None``，调用方据此跳过该字段。

    这里刻意做了两道防线：类型必须是字符串，前缀必须是 ``/uploads/``。
    任何其他值（``None``、绝对磁盘路径、外链）都不签名，从而杜绝把签名能力
    误用到任意路径上、变成"签名即可读取任意文件"的漏洞。
    """
    if not isinstance(media_url, str) or not media_url.startswith("/uploads/"):
        return None
    return sign_media_url(
        media_url,
        request.app.state.media_signing_key,
        ttl_seconds=request.app.state.media_url_ttl_seconds,
    )


def _public_response(
    payload: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    """按字段白名单裁剪服务层返回值。

    采用"白名单"而非"黑名单"：服务层将来新增任何内部字段，默认都不会外泄，
    必须显式加进 ``fields`` 才会出现在响应里。缺失的字段直接跳过，不补 ``None``。
    """
    return {key: payload[key] for key in fields if key in payload}


def _public_embed_response(
    request: Request, payload: dict[str, Any]
) -> dict[str, Any]:
    """组装嵌入接口的对外响应：白名单字段 + 两个签名下载地址。

    ``original_url``（原图）和 ``download_url``（已嵌入水印的成品图）都是内部
    路径，分别改名为 ``*_access_url`` 后以签名形式返回；签名失败（非受管路径）
    时该字段整个省略，前端据"字段是否存在"决定要不要显示下载按钮。
    """
    result = _public_response(payload, EMBED_PUBLIC_FIELDS)
    for source, target in (
        ("original_url", "original_access_url"),
        ("download_url", "download_access_url"),
    ):
        access_url = _signed_access(request, payload.get(source))
        if access_url:
            result[target] = access_url
    return result


def _public_extract_response(
    request: Request, payload: dict[str, Any]
) -> dict[str, Any]:
    """组装提取接口的对外响应。

    提取成功时服务层会给出 ``matched_file_url``——即命中的那张原始存档图，
    同样换成签名地址后以 ``matched_file_access_url`` 返回，供前端做比对展示。
    """
    result = _public_response(payload, EXTRACT_PUBLIC_FIELDS)
    access_url = _signed_access(request, payload.get("matched_file_url"))
    if access_url:
        result["matched_file_access_url"] = access_url
    return result


@router.post("/embed")
async def embed_watermark(
    request: Request,
    file: UploadFile = File(...),
    # 溯源标识：本次嵌入要绑定的业务用户标识，会写进 LSB 载荷与数据库记录。
    user_id: str = Form(...),
    # 水印模式（dct / lsb 等），由服务层的 normalize_mode 归一化后再使用。
    mode: str = Form("dct"),
    # ---- 可见版权水印（明水印）参数 ----
    copyright_enabled: str = Form("false"),
    copyright_text: str = Form("© QQ:757675150"),
    # 不透明度，服务层会夹紧到 [0.02, 0.90]，默认 0.16 保证肉眼弱可见但可取证。
    copyright_opacity: str = Form("0.16"),
    # 明水印铺排复杂度：影响文字块的数量与旋转角度分布。
    copyright_complexity: str = Form("medium"),
    # 是否使用不规则排布（抗"按固定网格擦除"攻击）。
    copyright_irregular_enabled: str = Form("true"),
    # 是否额外在角落打一处高对比度版权块，用于人工肉眼快速确认。
    copyright_prominent_corner_enabled: str = Form("false"),
    # ---- 隐水印强度参数 ----
    # 保真度：越高越偏向画质，服务层用 fidelity_to_strength 反推为嵌入强度。
    fidelity_level: str = Form("0.75"),
    robust_watermark_strength: str = Form(DEFAULT_ROBUST_WATERMARK_STRENGTH),
    # 鲁棒水印版本（v2/v3/v4）。选 v4 时走 DCT 认证码字 + FFT 同步导频的新链路，
    # 并会自动关闭小裁剪追踪与点阵追踪层。
    robust_watermark_version: str = Form(DEFAULT_ROBUST_WATERMARK_VERSION),
    # ---- 小裁剪追踪层（仅非 v4 链路生效）----
    # 注意默认值是空串而非 "false"：空串表示"未指定"，交由服务端全局开关决定；
    # 显式传 "false" 才是用户主动关闭。二者语义不同，不要改成布尔默认值。
    small_crop_trace_enabled: str = Form(""),
    small_crop_trace_strength: str = Form("1.0"),
    small_crop_trace_density: str = Form("high"),
    # ---- 点阵追踪层（仅非 v4 链路生效）----
    dot_matrix_trace_enabled: str = Form("false"),
    dot_matrix_trace_strength: str = Form("0.85"),
    # 登录态可选：未登录也允许嵌入，此时记录归属会回落到管理员账号。
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
    service: WatermarkService = Depends(get_watermark_service),
) -> dict[str, Any]:
    """嵌入水印：上传原图，返回带水印成品图的证据信息与限时下载地址。

    所有表单字段都以字符串接收、由服务层统一解析和夹紧，好处是前端传空值或非法值
    不会在 FastAPI 校验阶段直接 422，而是回落到各自的安全默认值。

    归属判定：``current_user`` 为 ``None``（匿名调用）时传 ``owner_user_id=None``，
    服务层会把记录挂到管理员名下，保证任何一次嵌入都有明确责任人。
    """
    result = await service.embed(
        file=file,
        owner_user_id=None if current_user is None else current_user.id,
        user_id=user_id,
        mode=mode,
        copyright_enabled=copyright_enabled,
        copyright_text=copyright_text,
        copyright_opacity=copyright_opacity,
        copyright_complexity=copyright_complexity,
        copyright_irregular_enabled=copyright_irregular_enabled,
        copyright_prominent_corner_enabled=copyright_prominent_corner_enabled,
        fidelity_level=fidelity_level,
        robust_watermark_strength=robust_watermark_strength,
        robust_watermark_version=robust_watermark_version,
        small_crop_trace_enabled=small_crop_trace_enabled,
        small_crop_trace_strength=small_crop_trace_strength,
        small_crop_trace_density=small_crop_trace_density,
        dot_matrix_trace_enabled=dot_matrix_trace_enabled,
        dot_matrix_trace_strength=dot_matrix_trace_strength,
    )
    return _public_embed_response(request, result)


@router.post("/extract")
async def extract_watermark(
    request: Request,
    file: UploadFile = File(...),
    service: WatermarkService = Depends(get_watermark_service),
) -> dict[str, Any]:
    """提取水印：上传一张可疑图片，反查它属于哪条溯源记录。

    服务层会先用文件指纹做一次快速命中（同一份字节直接查表），未命中再进入
    多层检测流水线（V4 → LSB → 频域 → 视觉匹配 …）。整条链路都没结论时抛 404。
    """
    return _public_extract_response(request, await service.extract_upload(file))


@router.post("/extract-url")
def extract_watermark_url(
    request: Request,
    url: str = Form(...),
    service: WatermarkService = Depends(get_watermark_service),
) -> dict[str, Any]:
    """按 URL 提取水印：由服务端下载图片后走与上传提取相同的检测流水线。

    与 ``/extract`` 的差别仅在图片来源；因为没有 ``await`` 的上传流，这里是同步路由。
    """
    return _public_extract_response(request, service.extract_url(url))
