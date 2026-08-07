"""图片资产管理的 HTTP 接口层。

对应 ``/api/images``：列表查询、原图/缩略图的受控下载、删除。

安全模型是本模块的重点——图片文件**不通过静态目录直出**，而是：

1. 列表接口只返回签名过的 ``*_access_url``，不暴露磁盘路径；
2. 真正取文件的 :func:`get_image_media` 校验 URL 上的 ``expire_time``+``signature``，
   校验不过一律 403；
3. 响应头带 ``no-store``，避免代理或浏览器把受控图片缓存下来绕过过期时间。
"""

from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from trace_app.auth.schemas import AuthenticatedUser
from trace_app.dependencies import get_current_user, get_management_service
from trace_app.management.service import ManagementService
from trace_app.media import sign_expiring_url, verify_media_signature

router = APIRouter(prefix="/api/images", tags=["images"])

# 列表接口允许外泄的字段白名单。
# ``time``/``conf`` 是给旧版前端的别名字段，与 ``created_at``/``confidence`` 同义，
# 为兼容存量页面一并保留。文件路径、哈希、水印参数等一律不在白名单内。
PUBLIC_IMAGE_FIELDS = (
    "id",
    "name",
    "size",
    "user_id",
    "trace_id",
    "mode",
    "mode_label",
    "created_at",
    "time",
    "status",
    "confidence",
    "conf",
)


def _public_image(
    record: dict[str, Any], *, key: bytes, ttl_seconds: int
) -> dict[str, Any]:
    """把一条内部图片记录裁剪成可以下发给前端的形状。

    :param record: 服务层返回的完整记录（含磁盘路径等敏感字段）。
    :param key: 媒体访问签名密钥（应用启动时生成，见 ``app.state``）。
    :param ttl_seconds: 签名地址有效期。
    :return: 白名单字段 + 最多两个限时访问地址组成的字典。

    这里签的是**接口路径** ``/api/images/{id}/download``，而不是 ``/uploads/...``
    静态路径——所以下载请求必然回到 :func:`get_image_media`，能继续走归属校验。
    ``quote(..., safe="")`` 对 id 做全量转义，防止 id 里的 ``/`` 撑破路径结构、
    导致签名内容与实际请求路径不一致。
    """
    result = {key: record[key] for key in PUBLIC_IMAGE_FIELDS if key in record}
    image_id = quote(str(record["id"]), safe="")
    # 有原图才给下载链接；缩略图同理。字段缺失时整项省略，前端据此隐藏入口。
    if record.get("download_url"):
        result["download_access_url"] = sign_expiring_url(
            f"/api/images/{image_id}/download",
            key,
            ttl_seconds=ttl_seconds,
        )
    if record.get("thumbnail_url"):
        result["thumbnail_access_url"] = sign_expiring_url(
            f"/api/images/{image_id}/thumbnail",
            key,
            ttl_seconds=ttl_seconds,
        )
    return result


@router.get("")
def list_images(
    request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: ManagementService = Depends(get_management_service),
) -> dict[str, Any]:
    """列出当前登录用户可见的图片，并附带统计概览。

    可见范围由服务层按角色裁定（管理员看全量、普通用户只看自己的），
    本路由不重复做权限判断，只负责把每条记录转成对外形状。
    """
    service_result = service.list_images(current_user)
    return {
        "items": [
            _public_image(
                record,
                key=request.app.state.media_signing_key,
                ttl_seconds=request.app.state.media_url_ttl_seconds,
            )
            for record in service_result["items"]
        ],
        "stats": service_result["stats"],
    }


@router.get("/{image_id}/{variant}", response_class=FileResponse)
def get_image_media(
    request: Request,
    image_id: str,
    variant: str,
    expire_time: str | None = None,
    signature: str | None = None,
    service: ManagementService = Depends(get_management_service),
) -> FileResponse:
    """按签名地址下发图片文件（``variant`` 为 ``download`` 或 ``thumbnail``）。

    本接口**不依赖登录态**，凭据就是 URL 上的 ``expire_time`` + ``signature``，
    这样前端 ``<img>`` 标签、下载器等无法携带鉴权头的场景也能取图。

    校验用的 ``access_path`` 必须按与签发时完全一致的规则重新拼装（同样的
    ``quote(safe="")``），否则签名比对会因编码差异误判。校验失败直接 403，
    不透露"图片是否存在"，避免通过响应码枚举 id。
    """
    access_path = f"/api/images/{quote(image_id, safe='')}/{variant}"
    if not verify_media_signature(
        access_path,
        expires=expire_time,
        signature=signature,
        key=request.app.state.media_signing_key,
    ):
        raise HTTPException(status_code=403, detail="图片访问链接无效或已过期")
    # 由服务层把 (id, variant) 解析成真实磁盘路径，并在其中做越界与存在性校验。
    path = service.get_image_media_path(image_id, variant)
    # private + no-store：签名会过期，绝不能让中间代理或浏览器留下可复用的副本。
    return FileResponse(path, headers={"Cache-Control": "private, no-store"})


@router.delete("/{image_id}")
def delete_image(
    image_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: ManagementService = Depends(get_management_service),
) -> dict[str, bool]:
    """删除一条图片记录及其关联文件。

    与列表接口不同，删除必须是登录态（``get_current_user`` 而非 optional），
    归属校验同样下沉到服务层：非本人且非管理员会在那里被拒。
    """
    return service.delete_image(image_id, current_user)
