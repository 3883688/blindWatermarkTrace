"""图片读入：上传文件、字节流、站内媒体路径、远程 URL。

系统里所有"把外部数据变成一个 PIL Image"的入口都收在这里，另外附带一个缩略图落盘
函数（同属 I/O 边界，放一处便于统一处理格式问题）。出口侧的编码与落盘在
``trace_app/imaging/output.py``。

**两类调用方，需求正好相反**：

* **嵌入链路**（:func:`load_upload_image` / :func:`load_image_from_bytes`）拿到的图
  **不做任何格式转换**，原样返回。因为上层要读 ``image.format`` 判断源图是不是
  JPEG，据此决定成品的输出格式与色度采样率；PIL 的 ``convert()`` 返回的新对象
  ``format`` 是 ``None``，转一下这条信息就没了。模式转换留给各水印层按需自理。
* **提取链路**（:func:`load_image_from_url`）反过来，读进来立刻 ``convert("RGB")``。
  这时源格式已无意义，检测器只要求所有输入是统一的三通道布局。

**远程抓取是本模块的安全边界。** :func:`fetch_remote_image_bytes` 承担 SSRF 防护：
用户能提交任意 URL，若不加约束，服务端就成了打内网的跳板。防线有五道——协议白名单、
解析后的 IP 必须全部是公网地址、禁用系统代理、重定向逐跳重新校验，外加体积与
content-type 限制。
"""

import ipaddress
from io import BytesIO
from pathlib import Path
import socket
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib import request as urllib_request

from fastapi import HTTPException, UploadFile
from PIL import Image

from trace_app.media import media_path_from_url, resolve_media_path


REMOTE_IMAGE_MAX_BYTES = 20 * 1024 * 1024
# 最多跟 4 跳。图床普遍有一两跳（CDN 调度、鉴权重写），4 次足够覆盖正常情况；
# 再多就更可能是重定向环，或是有意构造的绕过尝试。
REMOTE_IMAGE_MAX_REDIRECTS = 4
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class _NoRedirectHandler(urllib_request.HTTPRedirectHandler):
    """禁止 urllib 自动跟随重定向的 handler。

    ``redirect_request`` 返回 ``None`` 会让 urllib 放弃跳转，把 3xx 直接当作
    :class:`HTTPError` 抛出——这正是我们要的：把控制权拿回来，对每一跳的新地址都重跑
    一遍 IP 校验。若放任 urllib 自动跳转，攻击者只需让一个公网域名 302 到
    ``127.0.0.1``，首跳的校验就形同虚设了。
    """

    def redirect_request(self, *_args: Any, **_kwargs: Any):
        """返回 ``None`` 表示"不构造重定向请求"，从而阻断自动跳转。

        这是 urllib 约定的钩子：返回 ``None`` 时 urllib 不再跟随该跳转，
        而是把 3xx 响应当作 :class:`HTTPError` 抛出，控制权回到调用方。
        参数全部忽略，用 ``*_args`` / ``**_kwargs`` 吞掉即可——
        我们不关心目标地址是什么，一律不跟。
        """
        return None


def _default_open_url(request: urllib_request.Request, *, timeout: float):
    """用一个刻意"什么都不额外做"的 opener 发起请求。

    ``ProxyHandler({})`` 传空字典表示**不使用任何代理**，显式覆盖掉环境变量里的
    ``http_proxy`` / ``https_proxy``。这不只是性能考虑：走代理时真正的连接由代理发起，
    前面按解析结果做的公网 IP 校验就被整个绕过了。

    抽成独立函数还有一层用意——:func:`fetch_remote_image_bytes` 可以整体替换掉它，
    让测试完全不碰网络。
    """
    opener = urllib_request.build_opener(
        urllib_request.ProxyHandler({}),
        _NoRedirectHandler(),
    )
    return opener.open(request, timeout=timeout)


def _validate_public_http_url(
    url: str,
    *,
    resolve_host: Callable[..., Any],
) -> None:
    """校验 URL 指向的是公网 http(s) 地址，不合格直接抛 400。

    :param resolve_host: DNS 解析函数，签名同 :func:`socket.getaddrinfo`，可注入以便
        测试离线运行。
    :raises HTTPException: 400。分三种文案——协议/主机不合法、解析失败、指向内网。

    **为什么拒绝带 userinfo 的 URL**：``http://evil.com@127.0.0.1/`` 这类写法在不同
    解析器眼里的"真实主机"可能不同，是经典的绕过手法，直接不收。

    **为什么校验的是解析后的 IP 而不是主机名**：黑名单主机名（localhost、127.0.0.1）
    拦不住一个 A 记录指向内网的公网域名。这里把主机名解析成 IP 集合，要求**每一个**
    结果都是公网地址——只要有一条记录落在内网就整体拒绝，不给多结果轮询留缝隙。
    ``is_global`` 一次性覆盖了 loopback、私有段、link-local、保留段等所有非公网范围。

    ``str(item[4][0]).split("%", 1)[0]`` 是在剥 IPv6 的 scope id（形如
    ``fe80::1%eth0``），带着它 :func:`ipaddress.ip_address` 会解析失败。

    .. note::
       本校验与后续真正建连之间存在 TOCTOU 窗口：``urlopen`` 会自己再解析一次 DNS，
       恶意域名可以在两次解析之间把记录改到内网地址（DNS rebinding）。彻底堵住需要
       把校验通过的 IP 直接绑到连接上。当前实现未处理，此处仅如实记录，未作改动。
    """
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise HTTPException(status_code=400, detail="仅支持公网 http(s) 图片链接")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = resolve_host(
            parsed.hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="无法解析图片链接地址") from exc
    if not addresses:
        raise HTTPException(status_code=400, detail="无法解析图片链接地址")
    try:
        resolved = {
            ipaddress.ip_address(str(item[4][0]).split("%", 1)[0])
            for item in addresses
        }
    except (IndexError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="无法解析图片链接地址") from exc
    if any(not address.is_global for address in resolved):
        raise HTTPException(status_code=400, detail="不允许访问内网或本机地址")


def fetch_remote_image_bytes(
    url: str,
    *,
    resolve_host: Callable[..., Any] | None = None,
    open_url: Callable[..., Any] | None = None,
) -> tuple[bytes, str]:
    """抓取远程图片的原始字节，全程带 SSRF 与体积防护。

    :param resolve_host: DNS 解析函数，默认 :func:`socket.getaddrinfo`。
    :param open_url: 发起请求的函数，默认 :func:`_default_open_url`。两者都可注入，
        测试据此完全离线运行。
    :return: ``(图片字节, content-type 原文)``。
    :raises HTTPException: 一律 400，文案区分"链接不合法"、"读不到"、"重定向无效"、
        "超过 20MB"、"内容不是图片"几种情况。

    **重定向手动跟。** 每一跳都回到循环开头重跑 :func:`_validate_public_http_url`，
    这是防护能成立的关键——只校验用户提交的那个初始 URL 毫无意义。3xx 会从两个地方
    冒出来：``_NoRedirectHandler`` 让 urllib 把它抛成 :class:`HTTPError`（常规路径），
    而注入的 ``open_url`` 或别的 handler 也可能把 3xx 当作正常响应返回，所以两条路径
    都要判一次 ``REDIRECT_STATUSES``。相对 Location 用 :func:`urljoin` 拼回绝对地址，
    否则下一轮的 scheme/host 校验会直接失败。

    **体积做两层检查**：先信一次 ``content-length`` 抢先拒绝（省得白读一遍），再实际
    只读 ``MAX + 1`` 字节——多读的那 1 个字节就是超限判据。既不依赖对方声明的长度是否
    诚实，也不会把一个超大响应整个吃进内存。

    **content-type 只在对方给了的时候才查**，且只认 ``image/`` 前缀。不少图床返回
    ``application/octet-stream`` 甚至干脆不给，卡死会误伤正常链接；真正的把关在下游
    解码那一步。
    """
    resolver = resolve_host or socket.getaddrinfo
    opener = open_url or _default_open_url
    current_url = str(url or "").strip()

    for redirect_count in range(REMOTE_IMAGE_MAX_REDIRECTS + 1):
        _validate_public_http_url(current_url, resolve_host=resolver)
        request = urllib_request.Request(
            current_url,
            headers={"User-Agent": "WatermarkSystem/1.0"},
        )
        try:
            response = opener(request, timeout=10)
        except HTTPError as exc:
            if exc.code not in REDIRECT_STATUSES:
                raise HTTPException(status_code=400, detail="无法读取图片链接") from exc
            location = exc.headers.get("location", "")
            # HTTPError 本身也是个未读完的响应对象，跳下一轮前显式关掉，
            # 否则连接会一直攒着不释放。
            exc.close()
            if not location or redirect_count >= REMOTE_IMAGE_MAX_REDIRECTS:
                raise HTTPException(status_code=400, detail="图片链接重定向无效")
            current_url = urljoin(current_url, location)
            continue
        except HTTPException:
            # 校验函数抛出的 400 要原样放行：落进下面的兜底分支会把具体文案
            # （"不允许访问内网或本机地址"等）覆盖成笼统的"无法读取图片链接"。
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail="无法读取图片链接") from exc

        with response:
            status = getattr(response, "status", None)
            if status in REDIRECT_STATUSES:
                location = response.headers.get("location", "")
                if not location or redirect_count >= REMOTE_IMAGE_MAX_REDIRECTS:
                    raise HTTPException(status_code=400, detail="图片链接重定向无效")
                current_url = urljoin(current_url, location)
                continue
            if status is not None and not 200 <= int(status) < 300:
                raise HTTPException(status_code=400, detail="无法读取图片链接")

            content_type = str(response.headers.get("content-type", ""))
            content_length = response.headers.get("content-length")
            try:
                if content_length is not None and int(content_length) > REMOTE_IMAGE_MAX_BYTES:
                    raise HTTPException(status_code=400, detail="图片链接文件超过 20MB")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="图片链接响应无效") from exc
            # 多读 1 字节：读满 MAX+1 就说明对方内容超限，与它声明的长度无关。
            data = response.read(REMOTE_IMAGE_MAX_BYTES + 1)

        if len(data) > REMOTE_IMAGE_MAX_BYTES:
            raise HTTPException(status_code=400, detail="图片链接文件超过 20MB")
        if content_type and not content_type.lower().split(";", 1)[0].strip().startswith(
            "image/"
        ):
            raise HTTPException(status_code=400, detail="链接内容不是图片")
        return data, content_type

    # 循环内要么 return，要么在跳数用尽时抛错，正常走不到这里；纯兜底，
    # 避免将来改动循环条件后函数意外返回 None。
    raise HTTPException(status_code=400, detail="图片链接重定向无效")


async def load_upload_image(
    file: UploadFile,
    *,
    load_image_from_bytes_fn: Callable[[bytes], Image.Image] | None = None,
) -> Image.Image:
    """读 FastAPI 的上传文件并解码成图像。嵌入链路的入口。

    :param load_image_from_bytes_fn: 可注入的解码实现（测试用）。

    **这里刻意不做任何转换**：不转模式、不改尺寸，也不按 EXIF 摆正方向。原因是上层
    embed 要靠 ``image.format`` 判断源图是不是 JPEG，进而决定成品用什么格式、什么色度
    采样率落盘；一旦调了 ``convert()``，PIL 返回的新对象 ``format`` 就是 ``None``，
    这条信息就丢了。

    .. note::
       EXIF 的 Orientation 标签不被解析，图像按存储时的原始朝向进入水印链路。嵌入端与
       提取端都不摆正，所以坐标系是自洽的；但如果用户把图在别处按 EXIF 旋转后重新
       保存再来验证，几何就对不上了，只能指望 v4 的 FFT 同步导频去兜。当前行为如此，
       本次未作改动。
    """
    content = await file.read()
    loader = load_image_from_bytes_fn or load_image_from_bytes
    return loader(content)


def load_image_from_bytes(content: bytes) -> Image.Image:
    """从字节串解码图像，任何失败都转成 400。

    ``Image.open`` 是惰性的，只读文件头就返回；必须紧跟 ``load()`` 把像素真正解出来，
    否则"这串字节其实是坏图"要等到很久以后某次访问像素时才暴露，那时早已脱离这个
    还能给出友好错误的上下文。

    捕获裸 ``Exception`` 是刻意的：PIL 面对畸形输入能抛出的类型五花八门
    （``UnidentifiedImageError``、``OSError``、``struct.error``、超大图的
    ``DecompressionBombError`` 等），但对调用方来说它们是同一件事——这不是一张能用的图。
    """
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="上传文件不是有效图片") from exc
    return image


def save_thumbnail(image: Image.Image, path: Path, scale: float = 0.20) -> None:
    """生成等比缩略图并存成 PNG。

    :param scale: 缩放比例，默认 0.20。

    先 ``convert("RGB")`` 丢掉 alpha 与调色板：缩略图只用于界面展示，统一成三通道最
    省事。``max(1, ...)`` 保证极小图缩放后不会出现 0 像素的边——PIL 遇到 0 会直接报错。

    用 LANCZOS 而不是更快的插值：缩略图在列表页会被反复看，锯齿很显眼，而这点开销放在
    整条嵌入链路里可以忽略。格式固定 PNG 让上层能直接拼出 ``-thumb.png`` 的文件名，
    不必先问格式是什么。
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    thumb_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    thumbnail = rgb.resize(thumb_size, Image.Resampling.LANCZOS)
    thumbnail.save(path, format="PNG", optimize=True)


def load_image_from_url(url: str, upload_dir: Path) -> Image.Image:
    """按 URL 取图，同时兼容站内 ``/uploads/`` 路径与外部公网链接。提取链路的入口。

    :param upload_dir: 站内媒体根目录，用于解析 ``/uploads/`` 开头的路径。
    :raises HTTPException: 400（链接为空、不合法、内容不是有效图片）或 404（站内文件
        不存在）。

    站内分支必须走 :func:`~trace_app.media.resolve_media_path`：它把路径夹在
    ``upload_dir`` 之内，并限定只能落在几个白名单子目录里，防目录穿越。绝不能图省事
    直接 ``upload_dir / url``。

    两个分支都以 ``convert("RGB")`` 收尾。这与嵌入侧的 :func:`load_upload_image`
    形成对照：提取时源格式已经无关紧要，检测器只要求所有输入是一致的三通道布局。
    """
    text = str(url or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="请输入图片链接")
    if text.startswith("/uploads/"):
        path = resolve_media_path(upload_dir, media_path_from_url(text))
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="图片链接不存在")
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="图片链接不是有效图片") from exc
    # content-type 在抓取阶段已经校验过前缀，这里不再需要，直接丢弃。
    data, _content_type = fetch_remote_image_bytes(text)
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="图片链接不是有效图片") from exc
