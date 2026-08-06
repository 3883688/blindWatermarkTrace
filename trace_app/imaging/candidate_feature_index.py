"""候选记录的 ORB 特征索引：提取、落盘、加载与匹配打分。

这是水印检测的前置环节。检测一张来路不明的图时，不可能拿全库记录逐一做水印
解码——那太慢。于是先用视觉特征把库里的记录粗排一遍，取最像的若干条作为候选，
再交给水印层（``trace_app.watermark.ecc`` / ``trace_app.watermark.auth`` /
``watermark_v4``）
做定向验证。本模块负责的就是"粗排"这一步的数据结构与打分。

**为什么用 ORB 而不是感知哈希？** 目标场景是截图、裁剪、缩放后的图片，
整图级的哈希在裁掉一半内容后就完全失配了；ORB 是局部特征，只要还剩下一部分
原图内容就能匹配上。ORB 的描述子还是二进制的，比对用汉明距离，比 SIFT/SURF
那类浮点描述子快一个数量级，也没有专利顾虑。

登记图片时把描述子存成 npz，检测时直接加载回来与查询图比对，省去每次重新
解码原图、重跑特征提取的开销。索引文件属于**可重建的缓存**，缺失或损坏时
本模块一律静默降级为"没有特征"，绝不让整次检测失败。

本模块的完整导入路径是 ``trace_app.imaging.candidate_feature_index``，被
``trace_app.imaging.feature_matching`` 与
``trace_app.watermark.default_operations`` 使用。
"""

from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image


# 提特征前先把图缩到最长边 640。ORB 自带尺度金字塔，更高的分辨率对匹配质量
# 帮助有限却会线性推高耗时；固定上限还让不同尺寸的图落在同一尺度上，
# 使描述子之间更可比。
FEATURE_INDEX_MAX_SIDE = 640
# 每张图最多保留 768 个描述子，直接决定索引文件大小与比对耗时。
FEATURE_INDEX_MAX_DESCRIPTORS = 768
# ORB 描述子固定 256 位 = 32 字节，这是 OpenCV 的格式常量，不可改。
FEATURE_DESCRIPTOR_BYTES = 32


def extract_feature_descriptors(image: Image.Image) -> np.ndarray:
    """从图片中提取 ORB 描述子矩阵。

    :return: 形状 ``(n, 32)`` 的 ``uint8`` 矩阵；无可用特征时返回 ``(0, 32)``
        的空矩阵。

    始终返回二维矩阵而不是 ``None``，调用方就不必到处判空——空矩阵参与后续
    比对会自然得到 0 分。

    ``fastThreshold=7`` 明显低于 OpenCV 默认的 20：目标图片里常有大片平坦区域
    （纯色背景、渐变），阈值高了可能一个角点都检不出来。调低后宁可多收一些弱
    角点，再靠 ``nfeatures`` 上限和匹配阶段的比值检验把噪声筛掉。
    """
    rgb = image.convert("RGB")
    # 只缩不放：原图本来就小于 640 时 scale == 1.0，保持原样。
    # 放大不会凭空产生新特征，只会白白增加耗时。
    scale = min(1.0, FEATURE_INDEX_MAX_SIDE / max(rgb.size))
    if scale < 1.0:
        rgb = rgb.resize(
            # max(1, ...) 兜住极端长条图，避免短边被缩成 0 像素
            (max(1, int(round(rgb.width * scale))), max(1, int(round(rgb.height * scale)))),
            Image.Resampling.BICUBIC,
        )
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(
        nfeatures=FEATURE_INDEX_MAX_DESCRIPTORS,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=7,
    )
    _, descriptors = orb.detectAndCompute(gray, None)
    # 纯色图、空白图上 ORB 检不出角点，此时返回 None——统一折成空矩阵
    if descriptors is None or descriptors.size == 0:
        return np.empty((0, FEATURE_DESCRIPTOR_BYTES), dtype=np.uint8)
    # nfeatures 只是 ORB 在金字塔各层间的分配依据，实际数量可能略微超出，
    # 这里再截一刀兜底。ascontiguousarray 保证内存连续——BFMatcher 有此要求。
    return np.ascontiguousarray(
        descriptors[:FEATURE_INDEX_MAX_DESCRIPTORS],
        dtype=np.uint8,
    )


def save_feature_descriptors(path: Path, descriptors: np.ndarray) -> None:
    """把描述子矩阵压缩存盘，父目录不存在时自动创建。

    :raises ValueError: 矩阵形状不是 ``(n, 32)``。

    用 ``savez_compressed`` 而非裸 ``.npy``：描述子是二进制位模式，压缩率可观，
    而索引文件数量随记录数线性增长，磁盘占用值得省。存盘前强制校验形状，
    免得把脏数据写进索引、到检测时才炸。

    .. note::
        ``np.savez_compressed`` 在文件名不以 ``.npz`` 结尾时会自动补上后缀，
        此时实际落盘路径与传入的 ``path`` 不一致，
        :func:`load_feature_descriptors` 按原 ``path`` 就读不到。
        现有调用方传的都是 ``.npz`` 结尾的路径，未触发此问题，未作改动。
    """
    normalized = np.asarray(descriptors, dtype=np.uint8)
    if normalized.ndim != 2 or normalized.shape[1] != FEATURE_DESCRIPTOR_BYTES:
        raise ValueError("ORB descriptors must have shape (n, 32)")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, descriptors=normalized)


def load_feature_descriptors(path: Path) -> np.ndarray:
    """从 npz 读回描述子矩阵，读不出来就退化为空矩阵。

    :return: 形状 ``(n, 32)`` 的 ``uint8`` 矩阵；文件缺失、损坏或内容不合规时
        返回 ``(0, 32)`` 空矩阵。

    索引文件是可重建的缓存，读不出来最多让这条记录在粗排里得 0 分、落到后面的
    兜底排序里去，不该让整次检测失败——所以这里选择静默降级而非抛异常。
    形状校验放在异常捕获之外，是因为"文件能读但内容不对"同样要降级处理。

    ``allow_pickle=False`` 是必须的：索引文件位于数据目录，若被换成含恶意
    pickle 的文件，反序列化时会直接执行任意代码。

    ``zipfile.BadZipFile`` 必须单独列出：npz 本质是 zip 容器，中央目录被截断
    或损坏（写盘中断、磁盘写满）时 ``np.load`` 抛的就是它，而它直接继承
    ``Exception``、**不是** ``OSError`` 的子类，只捕获 ``OSError`` 会漏掉。
    """
    try:
        with np.load(path, allow_pickle=False) as payload:
            descriptors = np.asarray(payload["descriptors"], dtype=np.uint8)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return np.empty((0, FEATURE_DESCRIPTOR_BYTES), dtype=np.uint8)
    if descriptors.ndim != 2 or descriptors.shape[1] != FEATURE_DESCRIPTOR_BYTES:
        return np.empty((0, FEATURE_DESCRIPTOR_BYTES), dtype=np.uint8)
    return np.ascontiguousarray(descriptors, dtype=np.uint8)


def descriptor_match_score(
    query_descriptors: np.ndarray,
    target_descriptors: np.ndarray,
) -> tuple[int, float]:
    """比对两组描述子，返回 ``(可信匹配数, 匹配质量)``。

    :param query_descriptors: 待检测图片的描述子。
    :param target_descriptors: 某条候选记录的描述子。
    :return: ``(count, quality)``——``count`` 是通过比值检验的匹配对数量，
        ``quality`` 是该数量相对描述子较少一方总数的占比（保留 6 位小数）。

    **为什么用比值检验（Lowe's ratio test）而不是绝对距离阈值？**
    描述子之间的绝对汉明距离会随图片纹理丰富程度大幅漂移，定不出一个通用阈值。
    比值检验看的是"最佳匹配比次佳匹配好多少"：只有当最近邻明显甩开次近邻
    （距离小于其 0.75 倍）时才认可这对匹配，这个判据对图片内容的依赖小得多。

    **为什么数量和占比都返回？** 数量反映证据的绝对多寡，占比反映匹配的相对
    充分程度。小图描述子少，绝对数量天然吃亏，占比能把它拉回来；大图描述子多，
    占比又会自动稀释掉零星的偶然匹配。调用方先按数量卡门槛、再按占比排序。

    两侧都要求至少 2 个描述子——``knnMatch(k=2)`` 得有次佳匹配才谈得上比值检验。
    任何形状不合规或数量不足的情形一律返回 ``(0, 0.0)``，与"完全不匹配"同义，
    调用方无需区分。
    """
    query = np.asarray(query_descriptors, dtype=np.uint8)
    target = np.asarray(target_descriptors, dtype=np.uint8)
    if (
        query.ndim != 2
        or target.ndim != 2
        or query.shape[1:] != (FEATURE_DESCRIPTOR_BYTES,)
        or target.shape[1:] != (FEATURE_DESCRIPTOR_BYTES,)
        or len(query) < 2
        or len(target) < 2
    ):
        return 0, 0.0
    # ORB 描述子是二进制串，汉明距离是它的天然度量；描述子量级不大，
    # 暴力匹配（BFMatcher）比建 FLANN 索引更划算，且结果精确无近似。
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(query, target, k=2)
    good = [
        first
        for pair in pairs
        # 边缘情形下 knnMatch 可能只返回 1 个近邻，那就没法做比值检验
        if len(pair) == 2
        # 借单元素列表在推导式里完成解包，好让两个近邻同时参与下面的条件
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]
    count = len(good)
    # 分母取较小一侧的描述子总数：匹配对数不可能超过它，占比因此落在 [0, 1]。
    # max(1, ...) 只是防除零的形式保险，前面已经保证两侧至少各有 2 个。
    quality = count / max(1, min(len(query), len(target)))
    # 定到 6 位小数：分数要落库、要跨进程比较排序，截断掉浮点尾数才稳定
    return count, round(float(quality), 6)
