"""传统链路的视觉配准与候选匹配。

本模块服务于**非 V4 的旧检测链路**：把可疑图与库中已登记的成品图做 ORB 特征
匹配，用 RANSAC 估出单应矩阵，再把可疑图"摆正"到原图坐标系，交给
:func:`trace_app.watermark.robust.detect_aligned_authenticated_watermark`
去解码鲁棒水印。此外还承担两件配套工作：候选粗排（决定先验证谁）和
残差相关性打分（在没有可解码码字时给出一个相似度证据）。

**与** :mod:`watermark_v4.features` **的分工。** 两者主题相同（ORB + RANSAC），
但服务对象不同：

* :mod:`watermark_v4.features` 是 V4 的配准层，围绕带校验的
  :class:`~watermark_v4.features.FeatureIndex` 数据类构建，索引里存关键点坐标、
  描述子和缩略图，能做"受约束匹配"（旋转缩放已由 FFT 导频给出，只求平移），
  质量闸门也更严（几何合理性、条件数、重投影误差）。
* 本模块是历史实现，只存**裸描述子**（见 :mod:`candidate_feature_index`），
  匹配时重新在灰度图上现提关键点。没有索引数据类，也没有受约束匹配，
  阈值以经验值直接写在代码里。

两者互不调用，唯一的交集是 :func:`save_record_feature_index_v4`——嵌入水印时
本模块顺带把 V4 索引也写一份，供 V4 链路日后使用。

**关键产物：alignment 字典**（:func:`align_query_to_record` 的返回值）。
字段含义：``image`` 配准后的 RGB 数组、``valid_mask`` 有效像素掩码（配准后
落在画布内的位置为 True）、``target_scale`` 分析尺度相对原图的缩放比
（下游据此把原图坐标换算到分析坐标）、``inliers``/``ratio`` 匹配质量、
``coverage`` 有效像素占比、``homography`` 查询图到目标图的 3×3 矩阵。

**为什么阈值分两档。** 全图级判断（:func:`detect_by_visual_match`）要求
inliers ≥ 80、ratio ≥ 0.80，这是"基本就是同一张图"的标准；而配准类判断
（:func:`align_query_to_record`、:func:`record_visual_consistency`）放宽到
18 / 0.32，因为局部截图天然只有一小块能匹配上，内点数和占比都会大幅下降，
用严档会把截图场景全部挡在门外。
"""

import re
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from candidate_feature_index import (
    descriptor_match_score,
    extract_feature_descriptors,
    load_feature_descriptors,
    save_feature_descriptors,
)
from trace_app.config import (
    FEATURE_MATCH_MIN_GOOD,
    FEATURE_RECENT_BACKFILL,
    FEATURE_RECENT_RESERVE,
    ROBUST_CHANNEL,
    WATERMARK_LAYERS,
)
from watermark_v4.features import (
    extract_feature_index as extract_v4_feature_index,
    save_feature_index as save_v4_feature_index,
)

# 关掉 OpenCV 的内部多线程。本服务是多请求并发的 Web 后端，进程级并行已经存在，
# 再让每次 ORB/warp 调用铺满所有核心只会造成线程抢占、整体吞吐反而下降。
cv2.setNumThreads(1)


def save_record_feature_index(
    image: Image.Image,
    record_id: str,
    data_dir: Path,
    *,
    extract_feature_descriptors_fn: Callable | None = None,
    save_feature_descriptors_fn: Callable | None = None,
) -> str:
    """为一条记录提取并落盘传统 ORB 描述子索引。

    :param record_id: 记录 ID，会被清洗后用作文件名。
    :param data_dir: 数据根目录，索引写在其下的 ``feature_index/``。
    :return: 相对 ``data_dir`` 的 POSIX 风格路径，供写回记录的
        ``feature_index_path`` 字段。
    :raises ValueError: 记录 ID 清洗后为空。

    两个 ``*_fn`` 参数是给测试注入替身用的，生产调用不传。

    **为什么要清洗 ID。** 这个值直接拼成文件名，若原样使用，含 ``/`` 或 ``..``
    的 ID 就能把文件写到数据目录之外。这里用白名单（字母数字、下划线、连字符）
    而非黑名单过滤——白名单不会因为漏想到某种编码变体而被绕过。
    清洗后为空说明 ID 完全不可用，宁可报错也不写到一个意外的路径上。
    """
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", str(record_id))
    if not safe_id:
        raise ValueError("feature index record id is invalid")
    relative = Path("feature_index") / f"{safe_id}.npz"
    extractor = extract_feature_descriptors_fn or extract_feature_descriptors
    saver = save_feature_descriptors_fn or save_feature_descriptors
    descriptors = extractor(image)
    saver(data_dir / relative, descriptors)
    # 返回 POSIX 形式而非平台原生分隔符：这个字符串要存进记录（数据库/JSON），
    # 在 Windows 上生成、在 Linux 上读取是常态，统一成正斜杠才不会跨平台失效。
    return relative.as_posix()


def save_record_feature_index_v4(
    image: Image.Image,
    record_id: str,
    data_dir: Path,
    *,
    extract_v4_feature_index_fn: Callable | None = None,
    save_v4_feature_index_fn: Callable | None = None,
) -> str:
    """为一条记录额外落盘一份 **V4 格式**的特征索引。

    :return: 相对 ``data_dir`` 的 POSIX 路径（位于 ``feature_index_v4/``）。
    :raises ValueError: 记录 ID 清洗后为空。

    与 :func:`save_record_feature_index` 并列调用，两份索引各写各的目录：
    传统链路读 ``feature_index/``（裸描述子），V4 链路读 ``feature_index_v4/``
    （带关键点坐标与缩略图的 :class:`~watermark_v4.features.FeatureIndex`）。
    嵌入时一次性都生成，是为了让同一张图无论走哪条检测链路都有索引可用，
    不必事后回填。

    .. note::
       本模块只写不读 ``feature_index_v4/``——:func:`record_feature_index_path`
       解析的始终是传统索引目录。V4 索引的读取在 V4 检测链路里，属正常分工。
    """
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", str(record_id))
    if not safe_id:
        raise ValueError("feature index record id is invalid")
    relative = Path("feature_index_v4") / f"{safe_id}.npz"
    extractor = extract_v4_feature_index_fn or extract_v4_feature_index
    saver = save_v4_feature_index_fn or save_v4_feature_index
    index = extractor(image)
    saver(data_dir / relative, index)
    return relative.as_posix()


def record_feature_index_path(record: dict[str, Any], data_dir: Path) -> Path | None:
    """解析一条记录对应的特征索引文件路径。

    :return: 索引的绝对路径；路径不可信或无法推断时返回 ``None``。

    优先用记录里存的 ``feature_index_path``，缺失时按记录 ID 回退到约定位置
    ``feature_index/<id>.npz``——早期记录没有这个字段，回退能让它们继续可用。

    **路径必须当作不可信输入处理。** 它来自持久化数据，一旦被写脏（越权接口、
    历史数据迁移出错），拼进 ``data_dir`` 就会变成任意文件读取。两道拦截：
    绝对路径直接拒绝（否则 ``data_dir / "/etc/passwd"`` 会丢弃 ``data_dir``），
    含 ``..`` 段的也拒绝（防逐级上跳）。
    """
    raw = str(record.get("feature_index_path") or "").strip()
    if raw:
        # 先把反斜杠统一成正斜杠：Windows 上写入的记录可能带 ``\``，
        # 在 Linux 上 Path 不会把它当分隔符，``..`` 检查就会漏掉
        # ``..\..\etc`` 这类形式。
        relative = Path(raw.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            return None
        return data_dir / relative
    record_id = re.sub(r"[^A-Za-z0-9_-]", "", str(record.get("id") or ""))
    if not record_id:
        return None
    return data_dir / "feature_index" / f"{record_id}.npz"


def rank_aligned_candidates(
    image: Image.Image,
    records: list[dict[str, Any]],
    *,
    upload_dir: Path,
    data_dir: Path,
    generated_trace_ids: list[str],
    feature_match_min_good: int | None = None,
    feature_recent_backfill: int | None = None,
    feature_recent_reserve: int | None = None,
    record_feature_index_path_fn: Callable[[dict[str, Any]], Path | None] | None = None,
    save_record_feature_index_fn: Callable[[Image.Image, str], str] | None = None,
    extract_feature_descriptors_fn: Callable | None = None,
    load_feature_descriptors_fn: Callable | None = None,
    descriptor_match_score_fn: Callable | None = None,
) -> list[dict[str, Any]]:
    """对候选记录做粗排，把最可能命中的排在前面。

    :param records: 待排序的候选记录（调用方已筛掉无 trace_id / 无鲁棒水印的）。
    :param generated_trace_ids: 本会话刚生成的 trace_id，享最高回填优先级。
    :param feature_match_min_good: 进入"特征命中档"的最少匹配数，
        默认取 :data:`~trace_app.config.FEATURE_MATCH_MIN_GOOD`（12）。
    :return: 重排后的记录列表，**不截断**——截断由调用方按
        ``candidate_limit`` 决定。

    下游 :func:`align_query_to_record` 每个候选都要跑一遍 ORB + RANSAC + 两次
    透视变换，代价很高，所以必须先用便宜的手段把顺序排对，让预算花在前几名上。

    **三段式排序**，按证据强度从强到弱拼接：

    1. **特征命中档**——描述子匹配数 ≥ ``min_good``，按匹配数、匹配质量降序。
       这是最可信的信号，直接看到了共同的视觉特征。
    2. **近期保留档**——没到匹配阈值，但属于最近生成的记录，按生成顺序排。
       刚嵌完就来验证的图往往被平台重压缩过，特征匹配分数可能上不去，
       但它恰恰是最可能的答案，得给它留名额。
    3. **宽高比档**——其余记录按宽高比与查询图的接近程度排。
       这是纯兜底信号：比例差得远的基本不可能是同一张图。
    """
    min_good = FEATURE_MATCH_MIN_GOOD if feature_match_min_good is None else feature_match_min_good
    recent_backfill = (
        FEATURE_RECENT_BACKFILL if feature_recent_backfill is None else feature_recent_backfill
    )
    recent_reserve = (
        FEATURE_RECENT_RESERVE if feature_recent_reserve is None else feature_recent_reserve
    )
    index_path_for = record_feature_index_path_fn or (
        lambda record: record_feature_index_path(record, data_dir)
    )
    save_index = save_record_feature_index_fn or (
        lambda target, record_id: save_record_feature_index(target, record_id, data_dir)
    )
    extract_descriptors = extract_feature_descriptors_fn or extract_feature_descriptors
    load_descriptors = load_feature_descriptors_fn or load_feature_descriptors
    score_descriptors = descriptor_match_score_fn or descriptor_match_score
    # max(1, ...) 防除零：PIL 理论上不会给出 0 高度，但这里的代价是一次比较
    query_ratio = image.width / max(1, image.height)
    # 先取本会话刚生成的 trace_id，不足 recent_backfill 个再从 records 顺序补齐。
    # records 由上游按时间倒序给出，所以顺着取就是"最近的若干条"。
    recent_trace_ids = list(generated_trace_ids[:recent_backfill])
    for record in records:
        trace_id = record.get("trace_id")
        if len(recent_trace_ids) >= recent_backfill:
            break
        if trace_id and record.get("created_at") and trace_id not in recent_trace_ids:
            recent_trace_ids.append(trace_id)
    # 保留席位只给最前面 recent_reserve（2）个；名次即字典里的序号，
    # 后面第二段排序直接拿它当排序键。
    recent_order = {
        trace_id: index
        for index, trace_id in enumerate(recent_trace_ids[:recent_reserve])
    }
    # 回填范围比保留席位宽（4 > 2）：多建几份索引成本不高，
    # 而索引一旦缺失，这条记录在特征档就永远是 0 分。
    backfill_trace_ids = set(recent_trace_ids)

    # ===== 第一步：为近期记录按需补建缺失的特征索引 =====
    # 只对近期记录做，不全量重建——全量会把一次检测请求拖成分钟级。
    for record in records:
        if record.get("trace_id") not in backfill_trace_ids:
            continue
        path = index_path_for(record)
        if path and path.exists():
            continue
        url = record.get("download_url")
        record_id = record.get("id")
        # 只接受 /uploads/ 前缀的相对 URL。外链或绝对路径一律跳过，
        # 否则拼进 upload_dir 就成了任意文件读取。
        if not record_id or not url or not url.startswith("/uploads/"):
            continue
        image_path = upload_dir / url.replace("/uploads/", "")
        try:
            with Image.open(image_path) as target:
                save_index(target.convert("RGB"), str(record_id))
        except (OSError, ValueError):
            # 补建索引只是优化，失败了这条记录退化成"无索引"照常参与后面的排序，
            # 不该让整个检测请求失败。
            continue

    # ===== 第二步：用描述子匹配数把候选分成"命中档"和"其余" =====
    query_descriptors = extract_descriptors(image)
    feature_ranked = []
    remaining = []
    for record in records:
        path = index_path_for(record)
        # 无索引的记录用空描述子数组顶上，让它走同一条打分路径得 0 分，
        # 不必在下面再分支处理。(0, 32) 是 ORB 描述子的固定形状。
        descriptors = (
            load_descriptors(path)
            if path is not None and path.exists()
            else np.empty((0, 32), dtype=np.uint8)
        )
        match_count, match_quality = score_descriptors(query_descriptors, descriptors)
        if match_count >= min_good:
            # 用 ``**record`` 浅拷贝再挂上打分字段，避免把临时字段写回调用方
            # 持有的记录字典（那份可能是缓存，会被后续请求复用）。
            # 下划线前缀标明这些字段只在排序期间存在，不属于记录本体。
            feature_ranked.append({
                **record,
                "_feature_match_count": match_count,
                "_feature_match_quality": match_quality,
            })
        else:
            remaining.append(record)

    # 主键匹配数、次键匹配质量，都取负号实现降序（Python 的 sort 只升序，
    # 对数值取负是保持稳定排序的标准写法，比 reverse=True 更灵活——
    # reverse=True 会把所有键一起反转）。
    feature_ranked.sort(
        key=lambda record: (
            -int(record.get("_feature_match_count", 0)),
            -float(record.get("_feature_match_quality", 0.0)),
        )
    )

    def ratio_distance(record: dict[str, Any]) -> float:
        """记录与查询图的宽高比差值，越小越可能是同一张图。

        :return: 比值之差的绝对值；尺寸无从得知时返回 ``inf``，
            这样它会被排到最后而不是插到中间。

        优先用记录里存的宽高字段——纯数值比较，不碰磁盘。只有字段缺失或
        损坏时才退回去开图读尺寸，那一步会有 I/O 开销，能省则省。
        """
        recorded_width = record.get("image_width")
        recorded_height = record.get("image_height")
        if recorded_width and recorded_height:
            try:
                target_ratio = float(recorded_width) / max(1.0, float(recorded_height))
                return abs(target_ratio - query_ratio)
            except (TypeError, ValueError):
                # 字段里存了非数值（历史脏数据），转不动就走下面的读图分支
                pass
        url = record.get("download_url")
        if not url or not url.startswith("/uploads/"):
            return float("inf")
        path = upload_dir / url.replace("/uploads/", "")
        try:
            # Image.open 是惰性的，只读文件头就能拿到 size，不会解码整幅像素
            with Image.open(path) as target:
                target_ratio = target.width / max(1, target.height)
        except Exception:
            # 排序函数里抛异常会让整次检测崩掉，这里宽口径兜住：
            # 文件缺失、格式不支持、权限不足都只意味着"这条排最后"。
            return float("inf")
        return abs(target_ratio - query_ratio)

    # ===== 第三步：拼接三段 =====
    feature_trace_ids = {record.get("trace_id") for record in feature_ranked}
    recent_ranked = sorted(
        [
            record
            for record in remaining
            if record.get("trace_id") in recent_order
            and record.get("trace_id") not in feature_trace_ids
        ],
        key=lambda record: recent_order[record.get("trace_id")],
    )[:recent_reserve]
    # 用 id() 而非 trace_id 去重：同一个 trace_id 可能对应多条记录
    # （同一批次的多张图），必须精确排除**已入选的那几个对象**，
    # 否则会把同 trace_id 的兄弟记录一并从第三段里误删。
    reserved_ids = {id(record) for record in recent_ranked}
    aspect_ranked = sorted(
        [record for record in remaining if id(record) not in reserved_ids],
        key=ratio_distance,
    )
    return feature_ranked + recent_ranked + aspect_ranked


def image_to_cv_gray(image: Image.Image, max_side: int = 1200):
    """PIL 图转成 OpenCV 灰度数组，并把长边限制在 ``max_side`` 内。

    :return: uint8 灰度 ndarray。

    ORB 只看亮度，先转灰度能省掉三分之二的内存与计算。限长边是因为
    特征检测耗时随像素数线性增长，而 ORB 本身有尺度不变性——1200 边长
    足以支撑匹配打分，再大只是白烧 CPU。

    .. note::
       这里只做打分（内点数、内点占比），坐标不会被换算回原图尺度，
       所以缩放比例无需保留。需要真实坐标的场景走
       :func:`align_query_to_record`，它自己管理尺度。
    """
    rgb = image.convert("RGB")
    arr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
    height, width = arr.shape[:2]
    # min(1.0, ...) 保证只缩不放：放大不会带来新细节，只会增加计算量
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        # INTER_AREA（区域平均）是缩小图像的正确选择：它对被丢弃的像素做平均，
        # 天然抗混叠。用 INTER_LINEAR 缩小会产生摩尔纹，凭空造出假角点。
        arr = cv2.resize(arr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return arr


def record_visual_consistency(
    image: Image.Image,
    record: dict[str, Any],
    upload_dir: Path,
    *,
    image_to_cv_gray_fn: Callable | None = None,
    feature_match_score_fn: Callable | None = None,
    robust_residual_score_fn: Callable | None = None,
) -> tuple[bool, int, float, float]:
    """判断可疑图与某条记录在视觉上是否一致。

    :return: ``(是否一致, inliers, ratio, residual_score)``。前置条件不满足时
        统一返回 ``(False, 0, 0.0, 0.0)``——不区分"文件缺失"与"确实不像"，
        对调用方来说都是"这条走不通"。

    结合两种互补的证据：几何一致性（ORB + RANSAC 的内点数与占比）和
    水印残差相关性（把成品图与原图的差值当模板，看可疑图里有没有同样的扰动）。
    单看几何只能证明"是同一张图"，加上残差才能佐证"带着这条记录的水印"。

    **两档判定，取或。** 一张图只要满足其一即算一致：

    * 标准档 ``inliers ≥ 18 且 ratio ≥ 0.32 且 residual ≥ 0.08``——
      通用情形，几何证据一般但残差能兜住。
    * 强视觉小截图档 ``inliers ≥ 30 且 ratio ≥ 0.65 且 residual ≥ 0.06``——
      几何证据非常硬（内点多、占比高）时，允许残差门槛降一档。
      小截图上参与残差计算的像素本来就少，相关系数天然偏低，
      用统一门槛会把这类真命中全部误杀。
    """
    to_gray = image_to_cv_gray_fn or image_to_cv_gray
    match_score = feature_match_score_fn or feature_match_score
    residual_score_for = robust_residual_score_fn or robust_residual_score
    url = record.get("download_url")
    original_url = record.get("original_url")
    if not url or not original_url or not url.startswith("/uploads/") or not original_url.startswith("/uploads/"):
        return False, 0, 0.0, 0.0
    path = upload_dir / url.replace("/uploads/", "")
    original_path = upload_dir / original_url.replace("/uploads/", "")
    # 残差计算必须同时拿到成品图和原图（要用两者之差当水印模板），
    # 缺一不可，所以两个文件都得在
    if not path.exists() or not original_path.exists():
        return False, 0, 0.0, 0.0
    try:
        query = to_gray(image)
        target = to_gray(Image.open(path))
    except Exception:
        # 损坏的图片、超大图触发的 DecompressionBombError、格式不支持——
        # 一律降级为"不一致"，不能让单条脏记录拖垮整轮候选比对
        return False, 0, 0.0, 0.0
    inliers, ratio = match_score(query, target)
    # 显式传 18 / 0.32，覆盖 robust_residual_score 默认的 80 / 0.80。
    # 默认值是给全图级判断用的；这里面对的是局部截图，用严档会直接得 0 分，
    # 后面两条判定都指望这个残差值，一刀切会让整个函数永远返回 False。
    residual_score = residual_score_for(image, original_path, path, min_inliers=18, min_ratio=0.32)
    standard_match = inliers >= 18 and ratio >= 0.32 and residual_score >= 0.08
    strong_visual_small_crop_match = inliers >= 30 and ratio >= 0.65 and residual_score >= 0.06
    return (standard_match or strong_visual_small_crop_match), inliers, ratio, residual_score


def residual_candidate_evidence(
    image: Image.Image,
    *,
    records: list[dict[str, Any]],
    record_visual_consistency_fn: Callable[
        [Image.Image, dict[str, Any]], tuple[bool, int, float, float]
    ],
) -> dict[str, Any] | None:
    """在所有候选中挑出残差证据最强的一条，作为**辅助线索**输出。

    :return: 含候选 ID 与三项指标的字典；没有够强的候选则 ``None``。

    注意这是"线索"不是"结论"。残差相关性只能说明可疑图里存在与某条记录
    相似的扰动，不能证明水印确实存在——真正的归属判定由能解出码字的
    检测器负责（见 :func:`detect_by_residual_match` 的说明）。
    本函数的产物挂在检测证据里，供人工复核时参考。
    """
    # 只有带鲁棒水印的记录才存在"成品图 − 原图"的残差可比，其余没有可比对象
    records = [record for record in records if record.get("robust_watermark")]
    if not records:
        return None

    best_record = None
    best_inliers = 0
    best_ratio = 0.0
    best_residual = 0.0
    for record in records:
        consistent, inliers, ratio, residual_score = record_visual_consistency_fn(image, record)
        if not consistent:
            continue
        # 三级比较：残差为主键，同分再看内点数，再同分才看内点占比。
        # 残差排第一是因为它直接反映"水印图案对上了"，
        # 而内点数只反映"图像内容像"——后者对同一批相似素材区分度不够。
        if residual_score > best_residual or (
            residual_score == best_residual and (inliers > best_inliers or (inliers == best_inliers and ratio > best_ratio))
        ):
            best_record = record
            best_inliers = inliers
            best_ratio = ratio
            best_residual = residual_score

    # 0.12 比 record_visual_consistency 的入围线（0.08 / 0.06）高一截：
    # 那里是"值得看一眼"的门槛，这里是"值得写进证据"的门槛。
    # 弱线索写进报告只会误导复核的人，宁可什么都不给。
    if not best_record or best_residual < 0.12:
        return None

    return {
        "candidate_id": best_record.get("id"),
        "candidate_trace_id": best_record.get("trace_id"),
        "visual_inliers": best_inliers,
        "visual_ratio": round(best_ratio, 3),
        "residual_score": round(best_residual, 4),
    }


def detect_by_residual_match(image: Image.Image) -> dict[str, Any] | None:
    """残差匹配检测器——**已停用，恒返回** ``None``。

    :return: 始终 ``None``。

    保留这个空壳是为了兼容检测流水线的接口：
    :func:`trace_app.watermark.detection.extract_watermark_from_image` 把它作为
    回退链的一环注入，直接删掉会牵动一整串签名和测试。

    停用的原因见下面的原注释——视觉与残差相似度只能给候选排序，
    无法证明可疑图里真的嵌了水印。相似度高完全可能只是同一素材的不同副本，
    据此判定归属会造成误报。最终归属必须由能解出码字的检测器给出。
    """
    # Visual and residual similarity can rank candidates but cannot prove that the
    # query contains a watermark. Code-backed detectors perform final attribution.
    return None


def feature_match_score(query_gray, target_gray) -> tuple[int, float]:
    """对两幅灰度图做 ORB 匹配，返回几何一致性打分。

    :param query_gray: 可疑图的灰度数组。
    :param target_gray: 库中成品图的灰度数组。
    :return: ``(inliers, ratio)``——RANSAC 内点数与内点占比。

    只要打分、不要矩阵时用这个；需要矩阵走
    :func:`feature_match_homography`（两者流程完全一致，只是返回值不同）。

    **为什么用内点占比而不只看内点数。** 内点数受图片纹理丰富程度影响很大，
    一张细节密集的照片轻易能有几百内点，而纯色海报可能只有几十个。
    占比反映的是"通过比率检验的匹配里，有多大比例服从同一个几何变换"，
    跨图片可比性好得多。两个指标都返回，由调用方组合判断。

    .. note::
       ``good`` 不足 10 个时返回的是 ``(len(good), 0.0)``——第一个分量此刻是
       匹配对数而非内点数，语义与正常路径不一致。因为调用方一律用
       ``inliers >= 18`` 这类条件判断，而这里的值必然小于 10，实际不会造成
       误判。当前行为保持原样未改。
    """
    # nfeatures=3000 是精度与耗时的折中；fastThreshold=7 远低于 OpenCV 默认的
    # 20，让低对比度图片（截图、浅色海报）也能检出足够角点，
    # 随之混入的弱角点由后面的比率检验和 RANSAC 滤掉。
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    q_keypoints, q_descriptors = orb.detectAndCompute(query_gray, None)
    t_keypoints, t_descriptors = orb.detectAndCompute(target_gray, None)
    # 12 个关键点是能谈几何的最低要求。低于此值，即使全是内点，
    # RANSAC 也无法把真实变换与偶然巧合区分开。
    if q_descriptors is None or t_descriptors is None or len(q_keypoints) < 12 or len(t_keypoints) < 12:
        return 0, 0.0

    # ORB 是二进制描述子，必须用汉明距离；欧氏距离在此毫无意义
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(q_descriptors, t_descriptors, k=2)
    good = []
    for pair in raw_matches:
        # 边缘情况：目标描述子过少时 knnMatch 可能只返回 1 个结果，
        # 凑不出次近邻就没法做比率检验，跳过
        if len(pair) != 2:
            continue
        first, second = pair
        # Lowe 比率检验：最近邻要明显优于次近邻才算数。若两个候选一样像，
        # 说明这是重复纹理（砖墙、窗格、文字），对应关系无法确定，
        # 留着只会污染 RANSAC。
        # 0.78 比 Lowe 原论文的 0.75 略松——这里的输入常是被重压缩过的图，
        # 描述子有噪声，太严会把真匹配也筛掉。
        if first.distance < 0.78 * second.distance:
            good.append(first)

    # RANSAC 求单应至少要 4 点，但 4 点必然"完美拟合"、区分不出内外点。
    # 10 是留出冗余的经验下限。
    if len(good) < 10:
        return len(good), 0.0

    # findHomography 要求 (N, 1, 2) 的形状，这是 OpenCV 点集的约定布局
    q_points = np.float32([q_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    t_points = np.float32([t_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # 只要 mask（内点标记），矩阵本身丢弃。5.0 像素是重投影误差阈值：
    # 放得下 JPEG 压缩与缩放带来的坐标抖动，又不至于把明显错配算成内点。
    _, mask = cv2.findHomography(q_points, t_points, cv2.RANSAC, 5.0)
    if mask is None:
        return len(good), 0.0
    # mask 是 (N, 1) 的 0/1 数组，ravel 后求和即内点数
    inliers = int(mask.ravel().sum())
    ratio = inliers / max(1, len(good))
    return inliers, ratio


def feature_match_homography(query_gray, target_gray):
    """ORB 匹配并求出**目标图 → 查询图**的单应矩阵。

    :return: ``(target_to_query, inliers, ratio)``。任一环节失败时矩阵为
        ``None``，此时 inliers/ratio 可能仍带有已算出的部分信息。

    与 :func:`feature_match_score` 前半段完全相同（同样的 ORB 参数、同样的
    0.78 比率检验、同样的 5.0 像素 RANSAC 阈值），差别只在这里额外返回矩阵。

    **注意方向。** ``findHomography(q_points, t_points)`` 求出的是
    查询→目标，本函数把它**求逆后**返回，即 ``target_to_query``。
    这个方向正好是 :func:`robust_residual_score` 需要的——它要把成品图和原图
    一起搬到查询图的坐标系里做逐像素比对，而查询图的画幅是固定参照。

    求逆失败（矩阵奇异）时返回 ``None`` 矩阵但保留 inliers/ratio：
    几何解不出来，可打分依然有效，调用方还能拿它做相似度判断。
    """
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    q_keypoints, q_descriptors = orb.detectAndCompute(query_gray, None)
    t_keypoints, t_descriptors = orb.detectAndCompute(target_gray, None)
    if q_descriptors is None or t_descriptors is None or len(q_keypoints) < 12 or len(t_keypoints) < 12:
        return None, 0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(q_descriptors, t_descriptors, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        # 同 feature_match_score 的 Lowe 比率检验，0.78 略松于原论文的 0.75
        if first.distance < 0.78 * second.distance:
            good.append(first)

    if len(good) < 10:
        return None, len(good), 0.0

    q_points = np.float32([q_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    t_points = np.float32([t_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    query_to_target, mask = cv2.findHomography(q_points, t_points, cv2.RANSAC, 5.0)
    if mask is None or query_to_target is None:
        return None, len(good), 0.0
    inliers = int(mask.ravel().sum())
    ratio = inliers / max(1, len(good))
    try:
        # 求逆得到反方向的矩阵。退化点分布（例如所有内点共线）会让矩阵奇异，
        # numpy 抛 LinAlgError，此时几何无解但打分仍可用。
        target_to_query = np.linalg.inv(query_to_target)
    except np.linalg.LinAlgError:
        return None, inliers, ratio
    return target_to_query, inliers, ratio


def align_query_to_record(
    image: Image.Image,
    record: dict[str, Any],
    upload_dir: Path,
    *,
    resize_for_residual_fn: Callable | None = None,
    feature_match_homography_fn: Callable | None = None,
) -> dict[str, Any] | None:
    """把可疑图配准到某条记录的成品图坐标系，产出 alignment 字典。

    :return: 配准成功返回 alignment 字典，任一环节不达标返回 ``None``。

    这是传统链路的**核心几何步骤**。下游
    :func:`~trace_app.watermark.robust.detect_aligned_authenticated_watermark`
    拿到 ``image`` 后按 DCT 分块逐块解码鲁棒水印——前提是分块网格与嵌入时
    严格对齐，所以这里的几何精度直接决定能不能解出码字。

    返回字典的字段：

    ``image``
        配准后的 RGB uint8 数组，画幅与目标图（分析尺度下）一致。
    ``valid_mask``
        布尔数组，标记哪些像素真的来自查询图。查询图只是原图的一部分时，
        画布上会有大片空白，下游必须跳过这些位置，否则会把黑边当成信号解码。
    ``inliers`` / ``ratio``
        匹配质量，原样透传，写进检测证据。
    ``coverage``
        有效像素占比，即 ``valid_mask`` 的均值。
    ``target_scale``
        分析尺度宽度 ÷ 原图宽度。下游按原图坐标枚举 DCT 分块，
        再乘这个比例换算到配准后的坐标。
    ``target_size``
        分析尺度下的 ``(宽, 高)``。
    ``homography``
        查询图 → 目标图的 3×3 矩阵。

    **几道闸门及其理由**：

    * ``inliers < 18 或 ratio < 0.32``——几何证据不足，算出的矩阵不可信。
      比 :func:`detect_by_visual_match` 的 80 / 0.80 宽松得多，因为局部截图
      能匹配上的区域本就有限。
    * 矩阵非有限、或行列式绝对值 < 1e-9——退化解（把整幅图压成一条线之类），
      数学上成立、物理上荒谬，warp 出来会是一团噪声。
    * ``coverage < 0.05``——重叠区不到 5%，剩下的像素撑不起统计上有意义的解码。
    """
    resize_image = resize_for_residual_fn or resize_for_residual
    match_homography = feature_match_homography_fn or feature_match_homography
    url = record.get("download_url")
    if not url or not url.startswith("/uploads/"):
        return None
    target_path = upload_dir / url.replace("/uploads/", "")
    if not target_path.exists():
        return None
    try:
        # 双方都缩到同一个长边上限（1200）。查询图和目标图必须在**同一分析尺度**
        # 下匹配，否则 target_scale 的换算关系就不成立了。
        query_image = resize_image(image)
        # 保留缩放前的 original_target：末尾要用它的原始宽度算 target_scale
        original_target = Image.open(target_path).convert("RGB")
        target_image = resize_image(original_target)
    except Exception:
        return None

    query = np.asarray(query_image, dtype=np.uint8)
    target = np.asarray(target_image, dtype=np.uint8)
    query_gray = cv2.cvtColor(query, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    target_to_query, inliers, ratio = match_homography(query_gray, target_gray)
    if target_to_query is None or inliers < 18 or ratio < 0.32:
        return None
    try:
        # 再求一次逆，换回查询 → 目标的方向：warpPerspective 的矩阵语义是
        # 源 → 目的，而这里要把查询图搬到目标画布上。
        #
        # .. note::
        #    feature_match_homography 内部已经把 findHomography 的原始结果
        #    （查询 → 目标）求过一次逆，这里再求回来，等于 inv(inv(H))。
        #    多一次求逆会引入少量浮点误差，也多一个失败分支。当前行为未改动。
        query_to_target = np.linalg.inv(target_to_query)
    except np.linalg.LinAlgError:
        return None
    # 退化性检查。RANSAC 在点分布病态时会给出把平面压扁、或元素爆到 inf 的矩阵，
    # 它们的重投影误差可能很小，但 warp 出来毫无意义。
    # 行列式接近 0 正是"平面被压成线"的判据。
    if not np.isfinite(query_to_target).all() or abs(float(np.linalg.det(query_to_target))) < 1e-9:
        return None

    # OpenCV 的 shape 是 (高, 宽)，而 warpPerspective 的 dsize 是 (宽, 高)——
    # 顺序相反，是这一带最常见的低级错误来源
    target_height, target_width = target.shape[:2]
    # 图像用 INTER_CUBIC（双三次）：水印藏在 DCT 中频系数里，重采样会衰减它，
    # 三次插值比双线性保留更多高频，能提高解码成功率。
    # 画布外补纯黑，配合下面的 valid_mask 一起标出无效区。
    aligned = cv2.warpPerspective(
        query,
        query_to_target,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    # 用一张全 255 的图走同一个变换，落点非零处即"有真实像素"。
    # 这比"检查 aligned 是否为黑"可靠——原图里本来就可能有纯黑区域。
    #
    # 插值必须用 INTER_NEAREST。掩码是二值的，用 CUBIC/LINEAR 会在边界产生
    # 灰度过渡（甚至因 cubic 过冲而出现负值），阈值化后边界会外扩或内缩几个像素，
    # 把插值污染过的边缘像素误标成有效。
    valid_mask = cv2.warpPerspective(
        np.ones(query.shape[:2], dtype=np.uint8) * 255,
        query_to_target,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    coverage = float(valid_mask.mean())
    # 重叠不足 5% 时，剩下的像素太少，解码只是在噪声上做统计。
    #
    # .. note::
    #    ``coverage > 1.0`` 这一支永远为假——布尔数组的均值必落在 [0, 1]。
    #    属冗余防御，无害，未改动。
    if coverage < 0.05 or coverage > 1.0:
        return None
    # 分析尺度相对原图的缩放比。下游按**原图**坐标枚举 DCT 分块位置，
    # 乘上它才能定位到 aligned 数组里的对应区域。
    target_scale = target_width / max(1, original_target.width)
    return {
        "image": aligned,
        "valid_mask": valid_mask,
        "inliers": inliers,
        "ratio": round(ratio, 4),
        "coverage": round(coverage, 4),
        "target_scale": target_scale,
        "target_size": (target_width, target_height),
        "homography": query_to_target,
    }


def resize_for_residual(image: Image.Image, max_side: int = 1200) -> Image.Image:
    """转 RGB 并把长边限制在 ``max_side`` 内，供配准与残差计算共用。

    :return: RGB 模式的 PIL 图；原图未超限时是原尺寸的 RGB 副本。

    和 :func:`image_to_cv_gray` 用同一个 1200 的上限，但保留彩色——残差要在
    单个颜色通道（:data:`~trace_app.config.ROBUST_CHANNEL`，即蓝通道）上算，
    转灰度就把通道信息抹掉了。

    这里用 BICUBIC 而非 :func:`image_to_cv_gray` 的 INTER_AREA：残差比对关心的
    是水印那点微弱的高频扰动，区域平均会把它抹平，而双三次能保住更多细节。
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    # 只缩不放，理由同 image_to_cv_gray
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        rgb = rgb.resize((int(width * scale), int(height * scale)), Image.Resampling.BICUBIC)
    return rgb


def robust_residual_score(
    query_image: Image.Image,
    original_path: Path,
    watermarked_path: Path,
    min_inliers: int = 80,
    min_ratio: float = 0.80,
    *,
    robust_channel: int | None = None,
    resize_for_residual_fn: Callable | None = None,
    feature_match_homography_fn: Callable | None = None,
) -> float:
    """用"水印残差模板"衡量可疑图带某条记录水印的可能性。

    :param original_path: 该记录**未加水印**的原图。
    :param watermarked_path: 该记录的成品图（原图 + 水印）。
    :param min_inliers: 几何匹配的内点数下限，默认 80（全图级严档）。
    :param min_ratio: 内点占比下限，默认 0.80。
    :param robust_channel: 在哪个颜色通道上算，默认取
        :data:`~trace_app.config.ROBUST_CHANNEL`（2，即蓝通道）。
    :return: −1.0 ~ 1.0 的相关系数；几何不达标或数据退化时返回 0.0。

    **核心思路。** ``成品图 − 原图`` 就是水印本身的像素扰动，可以当作已知模板；
    ``可疑图 − 原图`` 是可疑图上实际观察到的扰动。两者做归一化相关，
    高相关说明可疑图身上带着同一个水印图案。

    这么做比"直接比可疑图和成品图有多像"强得多：后者主要被图片内容主导，
    同一素材的不同副本也会很像；减去原图后内容被抵消掉，
    剩下的差异才真正跟水印相关。

    **为什么用蓝通道。** 人眼对蓝色的亮度变化最不敏感，水印嵌在这里视觉影响最小，
    所以嵌入端选它，检测端自然也在这个通道上找。

    **为什么先几何对齐。** 逐像素相减要求两幅图像素级对应，差一个像素残差就全乱。
    所以先用 :func:`feature_match_homography` 求出矩阵，把成品图和原图都
    warp 到查询图的坐标系里再相减。

    默认 80 / 0.80 是"基本就是同一张图"的严档；局部截图场景由调用方
    （:func:`record_visual_consistency`）显式传 18 / 0.32 放宽。
    """
    channel = ROBUST_CHANNEL if robust_channel is None else robust_channel
    resize_image = resize_for_residual_fn or resize_for_residual
    match_homography = feature_match_homography_fn or feature_match_homography
    # 三幅图都过同一个 resize_for_residual，保证落在同一分析尺度。
    # 用 float32 是因为后面要做减法（会出负值）和范数计算，uint8 会溢出截断。
    query = np.array(resize_image(query_image), dtype=np.float32)
    watermarked = np.array(resize_image(Image.open(watermarked_path)), dtype=np.float32)
    original = np.array(resize_image(Image.open(original_path)), dtype=np.float32)
    query_gray = cv2.cvtColor(query.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(watermarked.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # 这里拿到的是**目标 → 查询**方向的矩阵，正好能把成品图/原图搬进查询图画幅
    homography, inliers, ratio = match_homography(query_gray, target_gray)
    if homography is None or inliers < min_inliers or ratio < min_ratio:
        return 0.0

    query_height, query_width = query.shape[:2]
    # 成品图与原图必须用**同一个矩阵**变换，否则两者之间就会错位，
    # 相减得到的"水印模板"是错的
    warped_watermarked = cv2.warpPerspective(watermarked, homography, (query_width, query_height))
    warped_original = cv2.warpPerspective(original, homography, (query_width, query_height))
    valid = cv2.warpPerspective(
        np.ones(watermarked.shape[:2], dtype=np.uint8) * 255,
        homography,
        (query_width, query_height),
    ) > 0
    # 覆盖率要求 30%，远高于 align_query_to_record 的 5%。这里算的是相关系数，
    # 样本量太少时它会剧烈波动——几百个像素凑巧对上就能给出很高的相关值，
    # 造成误报。30% 保证统计量稳定。
    if int(valid.sum()) < query_width * query_height * 0.30:
        return 0.0

    # expected：已知的水印扰动模板（成品 − 原图）
    # observed：可疑图上实际观察到的扰动（可疑 − 原图）
    # 两者都只取重叠区（[valid] 花式索引会摊平成一维向量）
    expected = (warped_watermarked[:, :, channel] - warped_original[:, :, channel])[valid]
    observed = (query[:, :, channel] - warped_original[:, :, channel])[valid]
    # 减去均值再算相关：去掉直流分量。可疑图经历的整体调亮/调暗会给 observed
    # 加一个常数偏置，不扣掉的话这个偏置会虚高相关系数。
    expected = expected - expected.mean()
    observed = observed - observed.mean()
    expected_norm = float(np.linalg.norm(expected))
    observed_norm = float(np.linalg.norm(observed))
    # 范数近零意味着该区域根本没有扰动（例如原图和成品图在这块完全相同，
    # 或可疑图这块被涂平了），除下去会得到 inf/nan
    if expected_norm < 1e-6 or observed_norm < 1e-6:
        return 0.0
    # 归一化相关（余弦相似度）。除以两个范数使结果落在 −1~1，
    # 与扰动的绝对强度无关——这点很关键，因为压缩会整体削弱残差幅度，
    # 但只要图案形状还在，相关系数就不会掉太多。
    return float(np.dot(expected, observed) / (expected_norm * observed_norm))


def detect_by_visual_match(
    image: Image.Image,
    *,
    records: list[dict[str, Any]],
    upload_dir: Path,
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
    now_text: Callable[[], str],
    watermark_layers: list[str] | None = None,
    image_to_cv_gray_fn: Callable | None = None,
    feature_match_score_fn: Callable | None = None,
    robust_residual_score_fn: Callable | None = None,
) -> dict[str, Any] | None:
    """视觉匹配检测器：靠"这就是同一张图"来判定归属。

    :param with_evidence_fields: 注入的回调，给结果补上通用证据字段。
    :param now_text: 注入的当前时间格式化函数，便于测试固定时间。
    :return: 命中返回检测结果字典，否则 ``None``。

    检测流水线里的**次级回退**：排在能直接读出数据的 LSB、点阵、对齐认证解码
    之后。它拿不到码字，只能说"可疑图与这条记录的成品图几何上高度吻合，
    且残差也对得上"，因此置信度上限压在 96 而非 99。

    **双阈值 80 / 0.80。** 比 :func:`align_query_to_record` 的 18 / 0.32 严得多，
    因为那里只是"值不值得试着解码"（解码失败自然会被否掉，误判无代价），
    而这里的结论**直接就是归属判定**，误判会把水印算到别人头上。

    残差检查只在几何已经达标时才做——残差计算要开两个文件、跑一次配准，
    很贵，几何都不过关就没必要花这个钱。
    """
    to_gray = image_to_cv_gray_fn or image_to_cv_gray
    match_score = feature_match_score_fn or feature_match_score
    residual_score_for = robust_residual_score_fn or robust_residual_score
    active_watermark_layers = WATERMARK_LAYERS if watermark_layers is None else watermark_layers
    if not records:
        return None

    # 查询图只转一次灰度，循环里复用
    query = to_gray(image)
    best_record = None
    best_inliers = 0
    best_ratio = 0.0
    for record in records:
        if not record.get("robust_watermark"):
            continue
        url = record.get("download_url")
        original_url = record.get("original_url")
        if not url or not original_url or not url.startswith("/uploads/") or not original_url.startswith("/uploads/"):
            continue
        path = upload_dir / url.replace("/uploads/", "")
        original_path = upload_dir / original_url.replace("/uploads/", "")
        if not path.exists() or not original_path.exists():
            continue
        try:
            target = to_gray(Image.open(path))
        except Exception:
            continue
        inliers, ratio = match_score(query, target)
        if inliers >= 80 and ratio >= 0.80:
            # 几何过关才值得算残差（用默认的 80 / 0.80 严档）。
            # 0.18 是这里的残差下限，比 residual_candidate_evidence 的 0.12 高——
            # 那边只是辅助线索，这边要直接下结论。
            residual_score = residual_score_for(image, original_path, path)
            if residual_score < 0.18:
                continue
        else:
            # 几何不过关就不花钱算残差，记 0 分让它继续参与下面的比较
            residual_score = 0.0
        # 排序键只看几何（内点数为主、占比为次），残差不参与择优——
        # 残差在上面已经当作一票否决用过了。
        #
        # .. note::
        #    这里的择优发生在最终阈值判定之前：某条记录若内点极多但占比不足
        #    0.80，仍会占住 best_record，把后面真正达标的记录挤掉，
        #    最终因 ``best_ratio < 0.80`` 整体返回 None（漏检而非误报）。
        #    当前行为如实保留，未改动。
        if inliers > best_inliers or (inliers == best_inliers and ratio > best_ratio):
            # 浅拷贝后挂上临时字段，不污染调用方持有的记录字典
            best_record = {**record, "_residual_score": residual_score}
            best_inliers = inliers
            best_ratio = ratio

    if not best_record or best_inliers < 80 or best_ratio < 0.80:
        return None

    # 置信度映射：基线 75，残差每 0.01 加 0.25 分，上限压在 96。
    # 不给到 99 是因为这条路径没有码字背书，理论上仍可能是巧合，
    # 要给密码学认证的 V4 路径留出区分度。
    confidence = min(96, max(75, int(75 + best_record.get("_residual_score", 0) * 25)))
    return with_evidence_fields({
        "id": best_record.get("id"),
        "trace_id": best_record.get("trace_id"),
        "user_id": best_record.get("user_id"),
        "mode": "robust_dct",
        "mode_label": "30% 局部截图匹配",
        "created_at": best_record.get("created_at"),
        "confidence": confidence,
        "phash_match": True,
        "status": "局部截图命中",
        "extracted_at": now_text(),
        "match_inliers": best_inliers,
        "match_ratio": round(best_ratio, 3),
        "watermark_layers": best_record.get("watermark_layers", active_watermark_layers),
        # 三层填同一个残差分：这条路径没有分层解码，拿不到各层独立的分数。
        # 字段仍要给全，前端的证据面板按固定结构渲染。
        "layer_scores": {
            "dct": round(float(best_record.get("_residual_score", 0.0)), 4),
            "dwt": round(float(best_record.get("_residual_score", 0.0)), 4),
            "fft": round(float(best_record.get("_residual_score", 0.0)), 4),
        },
    }, best_record)


def is_registered_original_image(
    image: Image.Image,
    *,
    records: list[dict[str, Any]],
    upload_dir: Path,
) -> bool:
    """判断上传的图**就是库里某条记录的未加水印原图**。

    :return: 命中任一原图返回 ``True``。

    检测流水线开头就要问这个问题。原图确实在库里，但它身上**没有水印**，
    所以正确结论是"未检出"而非"命中"。若不先挡掉，后面基于相似度的检测器
    （:func:`detect_by_visual_match`、残差类）会因为它与成品图极度相似而误报，
    把一张干净的原图判成带水印的泄露件。

    判据故意定得极严——两条同时满足才算：

    * 尺寸完全相同（不同尺寸直接跳过，连比都不比）；
    * 平均绝对差 ≤ 0.05 **且**最大绝对差 ≤ 1。

    最大差 ≤ 1 意味着全图不存在任何一个像素偏离超过 1 个色阶，只有无损保存或
    重新编码的舍入误差能满足。只看均值不够：水印是稀疏的局部扰动，
    大片相同区域会把均值稀释得很低，加上最大差这一条才挡得住"整体像、
    局部被改过"的情况。
    """
    # 用 int16 而非 uint8：下面要做减法，uint8 会回绕（3 − 5 变成 254），
    # 差值全错
    query = np.array(image.convert("RGB"), dtype=np.int16)
    query_height, query_width = query.shape[:2]
    for record in records:
        original_url = record.get("original_url")
        if not original_url or not original_url.startswith("/uploads/"):
            continue
        original_path = upload_dir / original_url.replace("/uploads/", "")
        if not original_path.exists():
            continue
        try:
            with Image.open(original_path) as original:
                # 尺寸不同就不可能是同一张原图，尽早跳过省下一次全图解码。
                # 注意 PIL 的 size 是 (宽, 高)，与 numpy shape 的 (高, 宽) 相反。
                if original.size != (query_width, query_height):
                    continue
                original_arr = np.array(original.convert("RGB"), dtype=np.int16)
        except Exception:
            # 文件损坏或格式不支持：当作"不是这一张"，继续看下一条
            continue
        diff = np.abs(query - original_arr)
        if float(diff.mean()) <= 0.05 and int(diff.max()) <= 1:
            return True
    return False
