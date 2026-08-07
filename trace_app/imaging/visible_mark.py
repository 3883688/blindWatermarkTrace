"""可见版权水印（明水印）：铺排渲染，以及基于亮度分布的兜底识别。

**在嵌入链路中的位置。** 明水印是整条流水线里**最先执行**的一层
（见 :class:`trace_app.watermark.service.WatermarkService` 的 embed）。
它做的是像素级的可见改动，必须先落到图上、成为后续所有隐水印层的载体基底；
反过来先嵌隐水印再叠明水印，会把已经写进像素的隐水印信号覆盖掉一部分。

**两种铺排方式，通过 ``irregular`` 开关切换：**

* 规则铺排 :func:`draw_text_pattern` —— 统一字号、统一间距、整体旋转一个角度，
  版面整齐，但存在严格的周期性。
* 不规则铺排 :func:`draw_irregular_text_pattern`（默认）—— 每一处的字体、字号、
  颜色、透明度、旋转角、落点全部独立随机。目的是抵抗"按固定网格擦除"的批量
  去水印工具：网格一旦不存在，攻击者就无法用一个模板同时抹掉所有实例，
  只能逐处手工修补。

**角落版权块** :func:`draw_prominent_corner_label` 走的是相反的路线：黑底黄字、
高对比、不追求隐蔽，供人工肉眼一眼确认归属，用于需要明确宣示版权的场合。

**不透明度。** 调用方（服务层）会把 ``opacity`` 夹在 ``[0.02, 0.90]``，默认 0.16。
下限保证水印不会被调到完全看不见而形同虚设，上限保证图片还能正常观看。
本模块内部再在这个基础上做随机浮动。

**关于识别。** :func:`detect_visible_copyright` 并没有做 OCR，只是统计亮部占比
来判断"这张图看起来像被叠过明亮的文字图层"。它是检测流水线里证据最弱的一环，
排在所有隐水印检测之后，置信度固定压在 68。
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def detect_visible_copyright(
    image: Image.Image,
    *,
    records: list[dict[str, Any]],
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
    now_text: Callable[[], str],
) -> dict[str, Any] | None:
    """靠亮部占比猜测"图上是否叠过明水印"，作为最后一档兜底证据。

    :param records: 全部溯源记录，用于筛出开启了明水印的那些。
    :param with_evidence_fields: 给结果补齐通用证据字段的回调。
    :param now_text: 取当前时间字符串的回调。
    :return: 判定命中时返回证据字典，否则 ``None``。

    **为什么不做 OCR。** 明水印的文字被旋转、缩放、半透明叠加在任意底图上，
    OCR 的识别率并不比这个启发式高多少，却要引入一整套识别依赖。因此这个
    轻量实现只做一件事：转灰度看直方图，亮度 ≥ 205 的像素占比超过 5%
    就认为"存在大面积的明亮叠加层"。205 这个门限对应白色文字在多数底图上
    的实际亮度，5% 则是铺满全图的文字所能占到的最小面积量级。

    命中后返回的 ``trace_id`` 不是真正的溯源号，而是版权文案的 SHA-1 前 12 位
    加 ``VISIBLE-`` 前缀拼出来的标识——本层根本没有读出任何编码数据，
    只能标记"这是哪一份版权文案"。置信度固定 68，明确低于所有隐水印层。

    .. note::
        这是一个**只看整体亮度、不看内容**的启发式，存在两类固有偏差，
        当前实现均未处理，此处按现状描述：

        1. 亮部占比高的普通图片（雪景、白底商品图、浅色截图）会被误判，
           从而返回 ``copyright_records[0]``——**第一条**开启了明水印的记录，
           而不是真正匹配的那条。
        2. 反之，深色文案或低不透明度的水印不会抬高亮部占比，会被漏判。

        实际部署上靠两点约束风险：该检测由
        ``visible_watermark_detection_enabled`` 开关控制、可整体关闭，
        且排在检测链的最末，只有前面所有层都落空时才会执行。
    """
    copyright_records = [item for item in records if item.get("copyright_enabled") and item.get("copyright_text")]
    if not copyright_records:
        return None

    # The visible copyright layer is human-readable but not OCR-backed in this lightweight version.
    # If hidden extraction fails and the image contains strong watermark-like bright overlays,
    # return the configured copyright source as a lower-confidence fallback.
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    # max(1, ...) 兜住零像素图，避免除零
    total = max(1, sum(histogram))
    # 205 是"接近白"的经验门限；占比 5% 对应文字铺满全图时的最小面积量级
    bright_ratio = sum(histogram[205:]) / total
    if bright_ratio < 0.05:
        return None

    record = copyright_records[0]
    text = str(record.get("copyright_text", "")).strip()
    # 文案里带着那串 QQ 号时归一成统一写法，否则退回去掉 © 的原文
    user_id = "QQ:757675150" if "757675150" in text else text.replace("©", "").strip()
    # 没有真实溯源号可用，只能拿文案摘要造一个稳定标识：同一份文案永远得到同一个 ID
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12].upper()
    return with_evidence_fields({
        "id": record.get("id"),
        "trace_id": f"VISIBLE-{digest}",
        "user_id": user_id or record.get("user_id") or "VISIBLE-WATERMARK",
        "mode": "visible",
        "mode_label": "可见版权水印",
        "created_at": record.get("created_at"),
        "confidence": 68,
        "phash_match": False,
        "status": "检测到可见版权水印",
        "extracted_at": now_text(),
    }, record)


def load_font(size: int) -> ImageFont.ImageFont:
    """按优先级取一个可用字体，全都取不到就退回 PIL 内置位图字体。

    顺序是 arial（英文）→ simhei（中文黑体）→ msyh（微软雅黑）：先试体积小、
    渲染快的西文字体，中文文案再靠后两个兜底。

    兜底的 ``load_default()`` 只有一个很小的固定字号，且不含中文字形，
    此时中文水印会渲染成方块——但这总好过直接抛异常让整次嵌入失败。
    """
    for name in ("arial.ttf", "simhei.ttf", "msyh.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_random_font(
    size: int,
    rng: np.random.Generator,
    *,
    load_font_fn: Callable[[int], ImageFont.ImageFont] | None = None,
) -> ImageFont.ImageFont:
    """随机挑一个可用字体，供不规则铺排使用。

    :param rng: 由调用方持有的随机源，保证同一次渲染里的随机决策同源。
    :param load_font_fn: 全部候选都加载失败时的兜底加载器（便于测试注入）。

    **为什么要随机换字体。** 字形是模板匹配的抓手：整幅图的水印若都用同一个
    字体同一个字号，攻击者只需标定一次就能批量定位并抹除。混用衬线/无衬线、
    常规/粗体、等宽/比例字体后，每一处的笔画形态都不同，模板失效。

    候选路径同时包含 ``%WINDIR%\\Fonts`` 下的绝对路径和裸文件名两种写法：
    前者命中 Windows，后者交给 PIL 自己按系统字体目录查找（Linux 容器里
    装了同名字体时可命中）。先 shuffle 再逐个试，取第一个加载成功的。
    """
    font_names = [
        "arial.ttf",
        "arialbd.ttf",
        "simhei.ttf",
        "msyh.ttc",
        "msyhbd.ttc",
        "simsun.ttc",
        "simkai.ttf",
        "consola.ttf",
        "verdana.ttf",
        "tahoma.ttf",
        "times.ttf",
    ]
    font_paths = []
    windows_font_dir = Path(os.getenv("WINDIR", "C:\\Windows")) / "Fonts"
    for name in font_names:
        # 每个字体压两条路径：系统字体目录的绝对路径，以及交给 PIL 自行搜索的裸名
        font_paths.append(windows_font_dir / name)
        font_paths.append(Path(name))
    rng.shuffle(font_paths)
    for path in font_paths:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    fallback = load_font_fn or load_font
    return fallback(size)


def draw_text_pattern(
    layer: Image.Image,
    text: str,
    angle: int,
    gap: int,
    opacity: int,
    *,
    load_font_fn: Callable[[int], ImageFont.ImageFont] | None = None,
) -> None:
    """规则铺排：把文案按固定间距平铺满整层，再整体旋转一个角度。

    :param layer: 待绘制的 RGBA 透明图层，**原地修改**。
    :param angle: 整体旋转角度（度）。斜排比正排难裁剪掉，也不容易与画面里
        本身的水平/垂直结构重合。
    :param gap: 相邻两处水印之间的像素间距，越小越密。
    :param opacity: 0~255 的 alpha 值，由上层的不透明度换算而来。

    **为什么要开一张两倍大的临时画布。** 旋转会把原本靠边的内容甩出画面，
    留下空白的三角。做法是在 ``2W × 2H`` 的画布上铺排（铺排范围还从
    ``-W`` 起、到 ``2W`` 止，四周都有余量），旋转后再从正中裁回原尺寸，
    这样四个角一定被填满。

    字号取短边的 1/18 并保底 18 像素：按比例走保证大图小图的观感一致，
    保底值防止缩略图上的水印小到无法辨认。

    ``text_width`` / ``text_height`` 的下限 80×24 是给步长兜底——文案很短
    （比如只有一个字符）时，步长会退化到只剩 gap，导致铺排过密、画面糊掉。
    """
    width, height = layer.size
    # 两倍画布：旋转后从正中裁回原尺寸，保证四角不留空白
    tile = Image.new("RGBA", (width * 2, height * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    font_loader = load_font_fn or load_font
    font = font_loader(max(18, min(width, height) // 18))
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except UnicodeEncodeError:
        # 兜底字体（load_default）不含 © 字形，换成可渲染的等义写法
        text = text.replace("©", "Copyright")
        bbox = draw.textbbox((0, 0), text, font=font)
    text_width = max(80, bbox[2] - bbox[0])
    text_height = max(24, bbox[3] - bbox[1])
    step_x = text_width + gap
    step_y = text_height + gap
    # 起点取负、终点超出画布，让铺排在旋转裁切后仍能盖住边缘
    for y in range(-height, height * 2, step_y):
        for x in range(-width, width * 2, step_x):
            draw.text((x, y), text, fill=(255, 255, 255, opacity), font=font)
    # expand=False：绕画布中心转，尺寸不变；BICUBIC 让斜向笔画不产生锯齿
    rotated = tile.rotate(angle, expand=False, resample=Image.Resampling.BICUBIC)
    layer.alpha_composite(rotated.crop((width // 2, height // 2, width // 2 + width, height // 2 + height)))


def draw_irregular_text_pattern(
    layer: Image.Image,
    text: str,
    opacity: int,
    complexity: str,
    *,
    load_random_font_fn: Callable | None = None,
) -> None:
    """不规则铺排：把文案以随机的字体/字号/颜色/角度/位置散布满整层。

    :param layer: 待绘制的 RGBA 透明图层，**原地修改**。
    :param opacity: 0~255 的基准 alpha，每一处会在此基础上随机浮动。
    :param complexity: 密度档位，接受 ``low/medium/high/extreme`` 或
        对应的中文 ``低/中/高/极``；识别不了的值一律按 ``medium`` 处理。

    **这是默认铺排方式，存在的理由就是抗擦除。** 规则铺排的每一处水印都可以
    由一个 ``(步长, 角度, 字体)`` 三元组推算出来，攻击者标定一次即可写脚本
    批量修补。这里把每一处的所有属性都独立随机化后，不存在可推算的规律，
    只能逐处手工处理，成本呈数量级上升。

    **两轮绘制。** 第一轮是主水印（数十到数百处，字号显眼）；第二轮是"微字"
    （数量约为主水印的 1.8 倍，字号减半、alpha 更低），铺在缝隙里。微字单独
    看几乎注意不到，但会让"抹掉所有水印"的工作量再翻几倍——攻击者往往只清掉
    看得见的主水印，微字留下来就足以证明来源。

    **随机源用 ``os.urandom`` 播种，即每次调用结果都不同。** 明水印不需要
    可复现（它不承载可解码的数据），反而是"同一张图重复加水印两次得到不同
    排布"更有利：攻击者拿不到两份同源样本做差分来定位水印位置。

    .. note::
        主循环里的 ``UnicodeEncodeError`` 兜底把 ``©`` 换成 ``Copyright``，
        改的是 ``safe_text``；而微字层用的 ``micro_text`` 是从入参 ``text``
        重新算的，拿不到这份替换。因此当随机挑中的字体不含 ``©`` 字形时，
        微字层的 ``ImageDraw.text`` 仍可能抛出未捕获的 ``UnicodeEncodeError``。
        实际未观察到该异常（候选字体清单里的字体都带 ``©``），
        此处按现状描述，未作改动。
    """
    random_font = load_random_font_fn or load_random_font
    width, height = layer.size
    # 每次调用都重新播种：明水印不需要可复现，随机反而能阻断差分定位
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), "big"))
    # 基准字号取短边的 1/20，保底 16 像素；后面每处再在 0.70~1.35 倍间浮动
    base_size = max(16, min(width, height) // 20)
    density = {
        "low": 0.55,
        "medium": 0.90,
        "high": 1.25,
        "extreme": 1.75,
        "低": 0.55,
        "中": 0.90,
        "高": 1.25,
        "极": 1.75,
    }.get(complexity, 0.90)
    # 数量随面积线性增长，除以 13 万（约 360×360）把密度归一到"每这么大一块几处"，
    # 保证不同尺寸的图观感一致；保底 10 处，防止小图上只落一两个水印被轻易裁掉
    count = max(10, int((width * height / 130_000) * density))
    # 四种接近白但略带色偏的颜色（暖白、淡黄、淡蓝、淡紫）。
    # 不用纯白：纯白在白底上会完全消失，且单一颜色便于攻击者按色值抠图。
    colors = [
        (255, 255, 255),
        (255, 248, 196),
        (210, 245, 255),
        (235, 235, 255),
    ]
    safe_text = text
    for index in range(count):
        size = int(base_size * float(rng.uniform(0.70, 1.35)))
        font = random_font(size, rng)
        # 18% 的概率画成"压缩版"：去掉空格、字号再砍到 0.45~0.65。
        # 这些小号实例混在正常水印中间，容易被攻击者忽略而留存下来。
        if rng.random() < 0.18:
            draw_text = safe_text.replace(" ", "")
            size = max(10, int(size * float(rng.uniform(0.45, 0.65))))
            font = random_font(size, rng)
        else:
            draw_text = safe_text
        try:
            # 只为量文字尺寸，开一张 1×1 的临时图即可，不必分配真实画布
            bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), draw_text, font=font)
        except UnicodeEncodeError:
            # 随机挑中的字体可能不含 © 字形；换掉后同时更新 safe_text，
            # 让后续循环不再重复触发
            safe_text = safe_text.replace("©", "Copyright")
            draw_text = draw_text.replace("©", "Copyright")
            bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), draw_text, font=font)
        text_width = max(1, bbox[2] - bbox[0])
        text_height = max(1, bbox[3] - bbox[1])
        # 每处水印先画在自己的小画布上再旋转。四周留 24 像素（上下左右各 12）：
        # 旋转会让文字的对角线方向变长，没有留白就会被切掉笔画尖端
        patch = Image.new("RGBA", (text_width + 24, text_height + 24), (0, 0, 0, 0))
        patch_draw = ImageDraw.Draw(patch)
        color = colors[int(rng.integers(0, len(colors)))]
        # alpha 在基准值的 0.45~1.25 倍间浮动，再夹到 [8, 220]：
        # 下限保证不会淡到彻底看不见，上限留一点透明度、不至于完全糊住画面
        alpha = max(8, min(220, int(opacity * float(rng.uniform(0.45, 1.25)))))
        patch_draw.text((12, 12), draw_text, fill=(*color, alpha), font=font)
        # 主角度范围 ±38°：够斜以避开画面本身的水平/垂直结构，又不至于难以辨读
        angle = float(rng.uniform(-38, 38))
        # 25% 的概率再叠加 ±58°，把这一处甩到接近垂直的方向，
        # 使整体角度分布出现两个明显不同的簇，进一步破坏规律性
        if rng.random() < 0.25:
            angle += float(rng.choice(np.array([-58, 58], dtype=np.int16)))
        # expand=True：让画布跟着旋转结果放大，避免转出去的部分被裁掉
        rotated = patch.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        # 落点允许**越出画布**（左上可为负、右下可超界），让边缘出现半截水印。
        # 全部落在画内的话，四周会留出一圈干净边框，裁掉边框即可减轻水印
        x = int(rng.integers(-rotated.width // 3, max(1, width - rotated.width * 2 // 3)))
        y = int(rng.integers(-rotated.height // 3, max(1, height - rotated.height * 2 // 3)))
        layer.alpha_composite(rotated, (x, y))

    # 第二轮：微字层。数量约为主水印的 1.8 倍，字号减半、alpha 只有基准的
    # 0.18~0.45，几乎注意不到。作用是在主水印被清除后仍留有取证痕迹
    micro_count = max(18, int(count * 1.8))
    micro_font = random_font(max(9, base_size // 2), rng)
    micro_text = text.replace(" ", "")
    for _ in range(micro_count):
        # 减去 24 / 12 是给字宽字高留出余量，避免文字大半被画布边缘截断
        x = int(rng.integers(0, max(1, width - 24)))
        y = int(rng.integers(0, max(1, height - 12)))
        alpha = max(5, int(opacity * float(rng.uniform(0.18, 0.45))))
        ImageDraw.Draw(layer).text((x, y), micro_text, fill=(255, 255, 255, alpha), font=micro_font)


def draw_prominent_corner_label(
    image: Image.Image,
    text: str,
    *,
    load_font_fn: Callable[[int], ImageFont.ImageFont] | None = None,
) -> Image.Image:
    """在右下角画一个高对比度的版权标牌，供人工肉眼快速确认归属。

    :param text: 版权文案；为空则退回内置默认文案。
    :return: 新的 RGB 图像（不修改入参）。

    与铺排水印的取向完全相反：这里**不追求隐蔽**。黑色半透明圆角底板
    （alpha 205，压住底图但仍透出一点纹理）+ 亮黄色文字（255,212,0）
    + 黑色描边。黄配黑是明度差最大的一组搭配，无论底图是亮是暗、
    是繁是简都能看清；描边则保证文字边缘不会与底图同色而糊在一起。

    **位置选右下角**，是图片信息密度通常最低、且社交平台裁剪时最少动到的
    区域（多数平台裁的是顶部或做居中方裁）。

    **自适应缩放。** 字号先取短边的 1/14（保底 22 像素），若渲染出来超过
    图宽的 72%，就以 2 像素为步长往下调，直到放得下或触到 16 像素下限。
    留 28% 余量是为了标牌不至于横贯整幅画面。下限 16 像素则是可读性底线——
    宁可标牌超宽也不能小到看不清，毕竟本层的全部意义就是"人能看见"。

    绘制文字时的 ``- bbox[1]`` 是在抵消字体的上方留白（ascent 起始偏移）：
    ``textbbox`` 返回的 top 通常不为 0，不减掉的话文字会在底板里偏下。
    """
    font_loader = load_font_fn or load_font
    base = image.convert("RGBA")
    draw = ImageDraw.Draw(base)
    safe_text = text.strip() or "© QQ:757675150"
    # 字号取短边的 1/14，明显大于铺排水印——这一层就是要让人一眼看到
    font_size = max(22, min(base.size) // 14)
    font = font_loader(font_size)
    try:
        # 量尺寸时带上 stroke_width，否则描边会撑破底板
        bbox = draw.textbbox((0, 0), safe_text, font=font, stroke_width=max(2, font_size // 18))
    except UnicodeEncodeError:
        safe_text = safe_text.replace("©", "Copyright")
        bbox = draw.textbbox((0, 0), safe_text, font=font, stroke_width=max(2, font_size // 18))
    # 标牌最宽占到图宽的 72%，剩下的余量避免它横贯整幅画面
    max_text_width = max(120, int(base.width * 0.72))
    # 逐步缩字号直到放得下；16 像素是可读性下限，到此为止不再缩
    while bbox[2] - bbox[0] > max_text_width and font_size > 16:
        font_size -= 2
        font = font_loader(font_size)
        bbox = draw.textbbox((0, 0), safe_text, font=font, stroke_width=max(2, font_size // 18))

    # 描边宽度随字号走（约 1/18），保证大小字号的观感一致
    stroke_width = max(2, font_size // 18)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    padding_x = max(12, font_size // 3)
    padding_y = max(8, font_size // 4)
    # 距图片边缘的留白，同样按短边比例自适应
    margin = max(14, min(base.size) // 40)
    # 以右下角为锚点反推左上角；max(margin, ...) 保证超宽文案也不会顶出画面
    right = base.width - margin
    bottom = base.height - margin
    left = max(margin, right - text_width - padding_x * 2)
    top = max(margin, bottom - text_height - padding_y * 2)
    radius = max(5, font_size // 6)
    # 黑色底板 alpha 205：压得住底图，又透出一点原有纹理，不像贴纸那样突兀
    draw.rounded_rectangle((left, top, right, bottom), radius=radius, fill=(0, 0, 0, 205))
    draw.text(
        # 减 bbox[1] 抵消字体顶部留白，让文字在底板里真正垂直居中
        (left + padding_x, top + padding_y - bbox[1]),
        safe_text,
        font=font,
        # 亮黄配黑底黑描边：明度差最大的组合，任何底图上都清晰
        fill=(255, 212, 0, 255),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 255),
    )
    return base.convert("RGB")


def apply_visible_copyright(
    image: Image.Image,
    enabled: bool,
    text: str,
    opacity: float,
    complexity: str,
    irregular: bool = True,
    prominent_corner: bool = False,
    *,
    draw_irregular_text_pattern_fn: Callable | None = None,
    draw_text_pattern_fn: Callable | None = None,
    draw_prominent_corner_label_fn: Callable | None = None,
) -> Image.Image:
    """明水印的总入口：铺排文字层，按需再叠角落版权块。

    :param enabled: 关闭时原样返回（仍会统一转成 RGB，保证下游拿到的格式一致）。
    :param opacity: 0~1 的不透明度，调用方已夹在 ``[0.02, 0.90]``。
    :param complexity: 密度档位，接受 ``low/medium/high/extreme`` 或 ``低/中/高/极``。
    :param irregular: 走不规则铺排（默认）还是规则铺排。
    :param prominent_corner: 是否额外画右下角的高对比度版权块。
    :return: 处理后的 RGB 图像。

    **先在独立的透明图层上画，最后一次性 alpha_composite 合成。** 直接往原图上
    逐处绘制的话，重叠处的半透明文字会一层层累加，越叠越不透明，出现斑块；
    在单独图层上画则重叠处仍受同一次合成控制，浓淡均匀。

    **无条件转 RGB 输出。** 后续的隐水印层（LSB、频域、点阵）都按三通道处理，
    这里统一格式，免得带 alpha 的 PNG 走到下游炸掉。

    ``settings`` 里的 ``(角度, 间距)`` 只在规则铺排时生效，档位越高列表越长：
    低档只铺一遍且间距很大（220），高档用两个相反角度交叉、极档用三个角度、
    间距压到 75。多角度交叉的目的是让水印形成网状，单向裁剪或单向修补都难以
    完全去除。

    .. note::
        ``settings`` 在 ``irregular`` 为真时也会被求值，但那条分支不会用到它。
        这是一次可以省掉的字典构造，开销可以忽略，此处未作改动。
    """
    draw_irregular = draw_irregular_text_pattern_fn or draw_irregular_text_pattern
    draw_pattern = draw_text_pattern_fn or draw_text_pattern
    draw_corner = draw_prominent_corner_label_fn or draw_prominent_corner_label
    if not enabled:
        return image.convert("RGB")

    base = image.convert("RGBA")
    # 单独一层透明画布：所有文字先画在这里，避免重叠处的半透明反复累加成斑块
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    text = text.strip() or "© QQ:757675150"
    alpha = int(255 * opacity)
    # 规则铺排的档位表：每个元组是一遍铺排的 (旋转角度, 间距)。
    # 档位越高，遍数越多、间距越小；多个角度交叉可形成难以单向去除的网状覆盖
    settings = {
        "low": [(-24, 220)],
        "medium": [(-24, 110)],
        "high": [(-24, 105), (24, 105)],
        "extreme": [(-32, 75), (0, 75), (32, 75)],
        "低": [(-24, 220)],
        "中": [(-24, 110)],
        "高": [(-24, 105), (24, 105)],
        "极": [(-32, 75), (0, 75), (32, 75)],
    }.get(complexity, [(-24, 110)])
    if irregular:
        draw_irregular(layer, text, alpha, complexity)
    else:
        # 逐档铺排，同一张图层上叠加多个角度
        for angle, gap in settings:
            draw_pattern(layer, text, angle, gap, alpha)
    result = Image.alpha_composite(base, layer).convert("RGB")
    # 角落版权块画在合成之后：它是不透明的实心标牌，不参与铺排层的透明度计算
    if prominent_corner:
        result = draw_corner(result, text)
    return result
