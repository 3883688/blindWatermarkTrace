"""ORB 特征索引：几何配准与候选粗排。

V4 检测要解码 DCT 码字，前提是把可疑图**摆回原图的坐标系**。本模块提供这一
能力：嵌入时为每张成品图存一份特征索引，检测时提取可疑图的特征与之匹配，
用 RANSAC 拟合出单应矩阵。

**为什么用 ORB。** 二进制描述子（32 字节）体积小、汉明距离比对快，
适合"一次检测要跟成百上千条记录比"的场景；且它对旋转和尺度变化有内建的
不变性，正好覆盖常见的图片改动。

**两种匹配模式**：

* :func:`match_feature_indexes` —— 无约束，直接 RANSAC 拟合。先试相似变换
  （只有旋转+缩放+平移，4 自由度），失败再退到完整单应（8 自由度）。
  自由度越少越不容易过拟合，所以优先试受限模型。
* :func:`match_feature_indexes_constrained` —— 已知旋转/缩放（由 FFT 导频给出），
  只需求平移。仅 3 个匹配点即可工作，是低纹理图片的救命通道。

**索引还兼作粗排依据。** :func:`rank_feature_candidates` 用少量描述子加
32×32 缩略图做快速打分，把最可能的候选排前面，避免对每个候选都做完整匹配。
"""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import V4Config


# 索引格式版本。加载时严格比对，版本不符直接判为不可用——
# 旧格式的描述子与新代码的匹配逻辑可能不兼容，宁可重建索引。
FEATURE_INDEX_SCHEMA_VERSION = 4
# 提特征前把图缩到的最大边长。缩小可大幅提速，且 ORB 本身有尺度不变性，
# 精度损失有限；关键点坐标事后会按比例还原回原图尺寸。
FEATURE_INDEX_MAX_SIDE = 640
# 索引里最多保存的描述子数，直接决定索引文件大小
FEATURE_INDEX_MAX_DESCRIPTORS = 3072
# 粗排时只用前 256 个描述子，够区分优劣即可，不必全量比对
FEATURE_COARSE_DESCRIPTORS = 256
# 让 ORB 多检出一些（4096），再由空间均衡策略挑出 3072 个，
# 这样有挑选余地，不会被密集堆在某一处的关键点占满配额
FEATURE_DETECTION_MAX_DESCRIPTORS = 4096
# ORB 描述子固定 32 字节（256 位）
FEATURE_DESCRIPTOR_BYTES = 32
# 附带的灰度缩略图边长，用于粗排时的整体外观比对
FEATURE_THUMBNAIL_SIZE = 32
# 索引文件大小上限，超过视为异常文件拒绝加载（防御畸形/恶意文件）
FEATURE_INDEX_MAX_FILE_BYTES = 8 * 1024 * 1024
# 空间均衡选点的网格划分（8×8 = 64 个格子）
FEATURE_SELECTION_GRID = 8


@dataclass(frozen=True, slots=True)
class FeatureIndex:
    """一张图片的特征索引：关键点、描述子与缩略图。

    ``frozen=True`` 只能防止**重新绑定字段**，挡不住对 NumPy 数组内容的原地
    修改。因此 :meth:`__post_init__` 末尾把三个数组都换成只读副本，
    用 ``object.__setattr__`` 绕过冻结限制完成替换——这是冻结数据类持有
    可变数组时的标准做法。
    """

    schema_version: int
    # 生成索引时的 OpenCV 版本。ORB 实现细节在版本间可能有差异，
    # 记下来便于排查"换了版本后匹配率下降"这类问题。
    opencv_version: str
    # 原图尺寸（非分析尺寸），关键点坐标以此为参照系
    image_width: int
    image_height: int
    # 提特征时用的缩放比例，坐标已按它还原，此处仅作记录
    analysis_scale: float
    # (n, 2) float32，关键点在**原图**坐标系中的位置
    keypoints: np.ndarray
    # (n, 32) uint8，与关键点一一对应的 ORB 描述子
    descriptors: np.ndarray
    # (32, 32) uint8 灰度缩略图，用于粗排
    thumbnail: np.ndarray

    def __post_init__(self) -> None:
        """全面校验形状与取值范围，并把数组替换为只读副本。

        关键点必须落在原图边界内——越界说明坐标换算出了错，
        后续用它做配准会得到完全错误的矩阵。
        """
        if type(self.schema_version) is not int:
            raise TypeError("feature schema version must be an integer")
        if self.schema_version != FEATURE_INDEX_SCHEMA_VERSION:
            raise ValueError("feature schema version is incompatible")
        if type(self.opencv_version) is not str or not self.opencv_version:
            raise TypeError("OpenCV version must be a nonempty string")
        if type(self.image_width) is not int or type(self.image_height) is not int:
            raise TypeError("feature image dimensions must be integers")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("feature image dimensions must be positive")
        if type(self.analysis_scale) is not float or not np.isfinite(self.analysis_scale):
            raise TypeError("feature analysis scale must be a finite float")
        if not 0.0 < self.analysis_scale <= 1.0:
            raise ValueError("feature analysis scale must be between zero and one")

        keypoints = np.asarray(self.keypoints)
        descriptors = np.asarray(self.descriptors)
        thumbnail = np.asarray(self.thumbnail)
        if keypoints.dtype != np.float32 or keypoints.ndim != 2 or keypoints.shape[1:] != (2,):
            raise ValueError("feature keypoints must have shape (n, 2) float32")
        if (
            descriptors.dtype != np.uint8
            or descriptors.ndim != 2
            or descriptors.shape[1:] != (FEATURE_DESCRIPTOR_BYTES,)
        ):
            raise ValueError("feature descriptors must have shape (n, 32) uint8")
        if len(keypoints) != len(descriptors) or len(keypoints) > FEATURE_INDEX_MAX_DESCRIPTORS:
            raise ValueError("feature keypoints and descriptors have incompatible counts")
        if thumbnail.dtype != np.uint8 or thumbnail.shape != (
            FEATURE_THUMBNAIL_SIZE,
            FEATURE_THUMBNAIL_SIZE,
        ):
            raise ValueError("feature thumbnail must have shape (32, 32) uint8")
        if not np.isfinite(keypoints).all():
            raise ValueError("feature keypoints must be finite")
        if len(keypoints) and (
            np.any(keypoints[:, 0] < 0.0)
            or np.any(keypoints[:, 0] >= self.image_width)
            or np.any(keypoints[:, 1] < 0.0)
            or np.any(keypoints[:, 1] >= self.image_height)
        ):
            raise ValueError("feature keypoints must be within original image bounds")

        # 冻结数据类不能直接赋值，用 object.__setattr__ 绕过；
        # 换成只读副本后，外部即使拿到引用也改不动索引内容。
        object.__setattr__(self, "keypoints", _readonly_copy(keypoints, np.float32))
        object.__setattr__(self, "descriptors", _readonly_copy(descriptors, np.uint8))
        object.__setattr__(self, "thumbnail", _readonly_copy(thumbnail, np.uint8))


@dataclass(frozen=True, slots=True)
class FeatureMatch:
    """一次成功的特征匹配：单应矩阵 + 质量指标。"""

    # 3×3 单应矩阵，把查询图坐标映射到目标（原图）坐标
    query_to_target: np.ndarray
    # 通过比率检验的匹配对总数
    good_matches: int
    # 其中被 RANSAC 认定为内点的数量
    inliers: int
    # 内点占比，衡量匹配的一致性
    inlier_ratio: float
    # 内点的中位重投影误差（像素），越小说明几何拟合越准
    median_reprojection_error: float

    def __post_init__(self) -> None:
        """校验矩阵形状与各质量指标；内点数不得超过匹配总数。"""
        matrix = np.asarray(self.query_to_target)
        if matrix.shape != (3, 3) or matrix.dtype.kind not in "f":
            raise ValueError("feature homography must be a 3x3 floating matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("feature homography must be finite")
        for name, value in (("good matches", self.good_matches), ("inliers", self.inliers)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.inliers > self.good_matches:
            raise ValueError("feature inliers cannot exceed good matches")
        for name, value in (
            ("inlier ratio", self.inlier_ratio),
            ("reprojection error", self.median_reprojection_error),
        ):
            if type(value) is not float or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        if self.inlier_ratio > 1.0:
            raise ValueError("inlier ratio must not exceed one")
        object.__setattr__(self, "query_to_target", _readonly_copy(matrix, np.float64))


@dataclass(frozen=True, slots=True)
class RankedFeatureCandidate:
    """粗排后的候选项及其打分依据。"""

    record_id: str
    index: FeatureIndex
    # 粗排匹配到的描述子对数（主排序键）
    match_count: int
    # 匹配数除以两侧描述子数的较小值，抵消"描述子多的天然占优"
    match_quality: float
    # 缩略图平均绝对差（归一化到 0~1），越小越像
    thumbnail_distance: float
    # 是否为"近期记录保留席位"入选，而非凭分数挤进来
    reserved: bool


def extract_feature_index(image: Image.Image) -> FeatureIndex:
    """为一张图片提取 ORB 特征索引。

    :return: 关键点坐标已换算回**原图**尺度的 :class:`FeatureIndex`。

    先缩图再提特征：ORB 的耗时随像素数增长，而它本身具备尺度不变性，
    在 640 边长上提取足以支撑配准。关键点坐标最后除以缩放比还原。

    ``fastThreshold=7`` 比 OpenCV 默认值（20）低不少，意在低对比度图片上
    也能检出足够多的角点；随之而来的低质量点由后面的空间均衡策略筛掉。
    """
    if type(image) is not Image.Image:
        raise TypeError("feature image must be an exact PIL Image")
    rgb = image.convert("RGB")
    # min(1.0, ...) 保证只缩小不放大：放大不会凭空产生细节，纯属浪费算力
    scale = min(1.0, FEATURE_INDEX_MAX_SIDE / max(rgb.size))
    if scale < 1.0:
        rgb = rgb.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            Image.Resampling.BICUBIC,
        )
    grayscale = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(
        nfeatures=FEATURE_DETECTION_MAX_DESCRIPTORS,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=7,
    )
    cv_keypoints, cv_descriptors = orb.detectAndCompute(grayscale, None)
    # 纯色或极低纹理的图片可能一个关键点都检不出。这不是错误，
    # 返回空索引即可——后续匹配会因描述子不足而自然失败，
    # 检测流水线会转向其他手段。
    if cv_descriptors is None or not cv_keypoints:
        keypoints = np.empty((0, 2), dtype=np.float32)
        descriptors = np.empty((0, FEATURE_DESCRIPTOR_BYTES), dtype=np.uint8)
    else:
        selected = _spatially_balanced_keypoint_indices(
            cv_keypoints,
            grayscale.shape[1],
            grayscale.shape[0],
        )
        count = min(len(selected), len(cv_descriptors), FEATURE_INDEX_MAX_DESCRIPTORS)
        selected = selected[:count]
        keypoints = np.asarray(
            [cv_keypoints[index].pt for index in selected],
            dtype=np.float32,
        )
        # 除以缩放比，把坐标从分析尺度换算回原图尺度
        keypoints /= np.float32(scale)
        # 换算的舍入误差可能让坐标恰好等于宽/高，超出 [0, W) 的合法区间，
        # 触发 FeatureIndex 的边界校验。这里夹一下，退回区间内。
        keypoints[:, 0] = np.minimum(keypoints[:, 0], image.width - 1e-4)
        keypoints[:, 1] = np.minimum(keypoints[:, 1], image.height - 1e-4)
        # 转成 C 连续内存：OpenCV 的匹配器要求连续布局，
        # 花式索引的结果通常不连续。
        descriptors = np.ascontiguousarray(cv_descriptors[selected], dtype=np.uint8)
    # 缩略图用 INTER_AREA（区域平均），能保留整体明暗结构，
    # 适合做"整体外观有多像"的粗略判断。
    thumbnail = cv2.resize(
        grayscale,
        (FEATURE_THUMBNAIL_SIZE, FEATURE_THUMBNAIL_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return FeatureIndex(
        schema_version=FEATURE_INDEX_SCHEMA_VERSION,
        opencv_version=cv2.__version__,
        image_width=image.width,
        image_height=image.height,
        analysis_scale=float(scale),
        keypoints=keypoints,
        descriptors=descriptors,
        thumbnail=np.asarray(thumbnail, dtype=np.uint8),
    )


def save_feature_index(path: Path, index: FeatureIndex) -> None:
    """把特征索引写成压缩 npz 文件。

    用 ``savez_compressed`` 而非 pickle：npz 只存数组，不含可执行内容，
    加载时可以关掉 pickle（见 :func:`load_feature_index` 的 ``allow_pickle=False``），
    从根本上杜绝"加载索引文件即执行任意代码"的风险。
    """
    if not isinstance(path, Path):
        raise TypeError("feature index path must be a Path")
    if type(index) is not FeatureIndex:
        raise TypeError("feature index must be an exact FeatureIndex")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(index.schema_version, dtype=np.int64),
        opencv_version=np.asarray(index.opencv_version),
        image_width=np.asarray(index.image_width, dtype=np.int64),
        image_height=np.asarray(index.image_height, dtype=np.int64),
        analysis_scale=np.asarray(index.analysis_scale, dtype=np.float64),
        keypoints=index.keypoints,
        descriptors=index.descriptors,
        thumbnail=index.thumbnail,
    )


def load_feature_index(path: Path) -> FeatureIndex | None:
    """加载特征索引；任何异常一律返回 ``None``。

    索引是**缓存性质**的数据，缺失或损坏都不该让整个检测流程崩掉——
    大不了这条候选走不通，流水线还有别的检测手段。因此这里把
    OSError/KeyError/TypeError/ValueError/EOFError 全部吞掉。

    两道安全措施：文件大小上限（防超大文件耗尽内存）、
    ``allow_pickle=False``（防反序列化执行任意代码）。
    """
    if not isinstance(path, Path):
        raise TypeError("feature index path must be a Path")
    try:
        if not path.is_file() or path.stat().st_size > FEATURE_INDEX_MAX_FILE_BYTES:
            return None
        with np.load(path, allow_pickle=False) as payload:
            return FeatureIndex(
                schema_version=int(payload["schema_version"].item()),
                opencv_version=str(payload["opencv_version"].item()),
                image_width=int(payload["image_width"].item()),
                image_height=int(payload["image_height"].item()),
                analysis_scale=float(payload["analysis_scale"].item()),
                keypoints=np.asarray(payload["keypoints"]),
                descriptors=np.asarray(payload["descriptors"]),
                thumbnail=np.asarray(payload["thumbnail"]),
            )
    except (OSError, KeyError, TypeError, ValueError, EOFError):
        # 文件不存在、字段缺失、版本不符、内容损坏——一律当作"没有索引"。
        return None


def match_feature_indexes(
    query: FeatureIndex,
    target: FeatureIndex,
) -> FeatureMatch | None:
    """无约束特征匹配，返回把查询图映射到目标图的单应矩阵。

    :return: 匹配成功返回 :class:`FeatureMatch`，否则 ``None``。

    **两级模型退化**。先用 ``estimateAffinePartial2D`` 拟合相似变换
    （旋转+等比缩放+平移，4 个自由度），失败再退到完整单应（8 个自由度）。

    这个顺序很重要：真实场景中的图片改动绝大多数是相似变换（截图、缩放、
    转发压缩），自由度少的模型不容易被离群点带偏。反过来先拟合单应，
    在退化配置下容易"完美拟合"出一个几何上荒谬的矩阵。

    18 是匹配数下限。低于此值，RANSAC 的样本量不足以可靠区分内点与外点。
    """
    _validate_feature_pair(query, target)
    good = _good_descriptor_matches(query.descriptors, target.descriptors)
    if len(good) < 18:
        return None
    query_points = np.asarray(
        [query.keypoints[item.queryIdx] for item in good],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    target_points = np.asarray(
        [target.keypoints[item.trainIdx] for item in good],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    # 第一级：相似变换。maxIters/confidence 给得很高，
    # 因为这一步是首选路径，值得多花些迭代把它拟合准。
    affine, affine_mask = cv2.estimateAffinePartial2D(
        query_points.reshape(-1, 2),
        target_points.reshape(-1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=10000,
        confidence=0.999,
        refineIters=20,
    )
    if affine is not None and affine_mask is not None:
        # 2×3 仿射矩阵补一行 [0,0,1] 升成 3×3，与单应矩阵统一形状，
        # 下游就不必区分两种模型了。
        affine_matrix = np.vstack(
            (np.asarray(affine, dtype=np.float64), np.asarray([0.0, 0.0, 1.0]))
        )
        affine_match = _feature_match_from_geometry(
            affine_matrix,
            affine_mask,
            good,
            query_points,
            target_points,
            query,
            target,
        )
        if affine_match is not None:
            return affine_match

    # 第二级：完整单应。相似变换拟合不出来（存在透视形变，如翻拍）时的退路。
    homography, mask = cv2.findHomography(
        query_points,
        target_points,
        cv2.RANSAC,
        5.0,
    )
    return _feature_match_from_geometry(
        homography,
        mask,
        good,
        query_points,
        target_points,
        query,
        target,
    )


def match_feature_indexes_constrained(
    query: FeatureIndex,
    target: FeatureIndex,
    *,
    rotation_degrees: float,
    scale: float,
    tile_size: int,
    tile_offset: tuple[int, int] | None = None,
) -> FeatureMatch | None:
    """受约束的特征匹配：旋转与缩放已由 FFT 导频给出，这里只求平移。

    :param rotation_degrees: 导频估出的旋转角。
    :param scale: 导频估出的缩放比。
    :param tile_size: DCT 分块尺寸，用于把平移量吸附到网格上。
    :param tile_offset: 导频估出的网格起始偏移；提供时启用吸附。
    :return: 成功返回 :class:`FeatureMatch`，否则 ``None``。

    **为什么需要它。** 无约束匹配要 18 个点起步，低纹理图片（大片纯色、
    简单图形）根本凑不出来。但如果旋转和缩放已知，未知量就只剩平移的两个
    分量——**3 个点**就够了，而且不需要 RANSAC 迭代。

    **求解方法：平移量投票。** 已知线性部分后，每一对匹配点都能独立给出一个
    平移量估计。正确匹配给出的值会聚成一簇，错误匹配则四散分布。
    于是找出邻居最多的那个点作为种子（相当于一次简易的众数搜索），
    再用簇内中位数得到稳健估计。

    **网格吸附。** 有 ``tile_offset`` 时，把平移量吸附到最近的
    ``tile_offset + k×tile_size``。DCT 解码要求分块严格对齐，
    几个像素的偏差就会让整块解不出——而导频给出的网格相位往往比
    特征点回归出的平移量更准。
    """
    _validate_feature_pair(query, target)
    if type(rotation_degrees) not in (int, float) or not np.isfinite(rotation_degrees):
        raise TypeError("constrained rotation must be finite")
    if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("constrained scale must be finite and positive")
    if type(tile_size) is not int or tile_size <= 0:
        raise ValueError("constrained tile size must be a positive integer")
    if tile_offset is not None and (
        type(tile_offset) is not tuple
        or len(tile_offset) != 2
        or any(type(value) is not int for value in tile_offset)
    ):
        raise ValueError("tile offset must be an integer pair")

    good = _good_descriptor_matches(query.descriptors, target.descriptors)
    # 只需 3 个点即可（未知量只剩平移的两个分量），远低于无约束匹配的 18 个
    if len(good) < 3:
        return None
    radians = math.radians(float(rotation_degrees))
    # 取倒数：sync 给出的是"图被放大了 s 倍"，而这里要构造的是
    # 把它映射**回**原图的变换，故用 1/s。
    inverse_scale = 1.0 / float(scale)
    linear = inverse_scale * np.asarray(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ],
        dtype=np.float64,
    )
    query_points = np.asarray(
        [query.keypoints[item.queryIdx] for item in good],
        dtype=np.float64,
    )
    target_points = np.asarray(
        [target.keypoints[item.trainIdx] for item in good],
        dtype=np.float64,
    )
    # 每对匹配点各自推出一个平移量估计：t = 目标点 - 线性变换(查询点)
    translations = target_points - query_points @ linear.T
    # 投票找众数：选出 6 像素邻域内同伴最多的那个估计作为种子
    seed_index = max(
        range(len(translations)),
        key=lambda index: int(
            np.count_nonzero(
                np.linalg.norm(translations - translations[index], axis=1) <= 6.0
            )
        ),
    )
    seed = translations[seed_index]
    # 两轮收敛：先以种子为中心按 6 像素圈定初始内点、取中位数，
    # 再以这个更准的中心按 5 像素重新圈定。逐步收紧能剔掉边缘的离群点。
    inlier_mask = np.linalg.norm(translations - seed, axis=1) <= 6.0
    translation = np.median(translations[inlier_mask], axis=0)
    inlier_mask = np.linalg.norm(translations - translation, axis=1) <= 5.0
    inliers = int(np.count_nonzero(inlier_mask))
    if inliers < 3:
        return None
    translation = np.median(translations[inlier_mask], axis=0)
    # 网格吸附：把平移量归到最近的 offset + k×tile_size。
    # 导频给出的网格相位比特征点回归更精确，此处以它为准。
    if tile_offset is not None:
        translation = np.asarray(
            [
                offset + round((value - offset) / tile_size) * tile_size
                for value, offset in zip(translation, tile_offset)
            ],
            dtype=np.float64,
        )
    # 组装成标准 3×3 齐次矩阵：左上 2×2 是线性部分，第三列是平移
    matrix = np.asarray(
        [
            [linear[0, 0], linear[0, 1], translation[0]],
            [linear[1, 0], linear[1, 1], translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not _plausible_homography(matrix, query, target):
        return None
    # 用最终矩阵（可能已被网格吸附改动过）重算一遍内点与误差：
    # 吸附会移动平移量，内点集合随之变化，必须以最终结果为准重新统计。
    projected = query_points @ linear.T + translation
    errors = np.linalg.norm(projected - target_points, axis=1)
    inlier_mask = errors <= 5.0
    inliers = int(np.count_nonzero(inlier_mask))
    if inliers < 3:
        return None
    median_error = float(np.median(errors[inlier_mask]))
    return FeatureMatch(
        query_to_target=matrix,
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=float(inliers / len(good)),
        median_reprojection_error=median_error,
    )


def _feature_match_from_geometry(
    geometry: np.ndarray | None,
    mask: np.ndarray | None,
    good: list[cv2.DMatch],
    query_points: np.ndarray,
    target_points: np.ndarray,
    query: FeatureIndex,
    target: FeatureIndex,
) -> FeatureMatch | None:
    """把 RANSAC 的输出包装成 :class:`FeatureMatch`，并施加三道质量闸。

    :return: 三道闸全过才返回结果，否则 ``None``。

    仿射与单应两条路径共用本函数，保证质量标准完全一致。
    """
    if geometry is None or mask is None:
        return None
    matrix = np.asarray(geometry, dtype=np.float64)
    # 闸门 1：几何合理性——矩阵不退化，且映射后的图框落在合理范围内
    if not _plausible_homography(matrix, query, target):
        return None
    inlier_mask = np.asarray(mask).ravel()[: len(good)].astype(bool)
    inliers = int(np.count_nonzero(inlier_mask))
    ratio = inliers / len(good)
    # 闸门 2：内点数量与占比。两个条件是"或"的关系：
    #   内点数 < 18            —— 样本太少，直接否决；
    #   占比 < 32% 且内点 < 256 —— 一致性差。但内点绝对数很大（≥256）时
    #                             即便占比低也接受：这是重复纹理图片的典型
    #                             特征（大量误匹配摊薄了占比），
    #                             几百个几何一致的内点已足以证明匹配成立。
    if inliers < 18 or (ratio < 0.32 and inliers < 256):
        return None
    # 闸门 3：中位重投影误差不超过 5 像素。
    # 用中位数而非均值，避免个别离群内点抬高整体误差。
    projected = cv2.perspectiveTransform(query_points, matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - target_points.reshape(-1, 2), axis=1)
    inlier_errors = errors[inlier_mask]
    median_error = float(np.median(inlier_errors)) if inlier_errors.size else float("inf")
    if not np.isfinite(median_error) or median_error > 5.0:
        return None
    # 齐次归一化：单应矩阵按比例缩放不改变几何意义，
    # 统一除以 matrix[2,2] 得到规范形式，便于比较与调试。
    matrix /= matrix[2, 2]
    return FeatureMatch(
        query_to_target=matrix,
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=float(ratio),
        median_reprojection_error=median_error,
    )


def rank_feature_candidates(
    query: FeatureIndex,
    candidates: tuple[tuple[str, FeatureIndex], ...],
    *,
    recent_record_ids: tuple[str, ...] = (),
    config: V4Config,
) -> tuple[RankedFeatureCandidate, ...]:
    """对候选做轻量粗排，返回值得深入验证的前几个。

    :param recent_record_ids: 近期生成的记录，享有"保留席位"。
    :return: 最多 ``config.candidate_limit`` 个候选，按可能性降序。

    完整匹配（RANSAC + 多次试解）很贵，不能对每个候选都做。这里用
    **前 256 个描述子**的匹配数快速打分，只把最可能的几个交给完整验证。

    **保留席位机制。** 排序名额先按分数发前 2 个，剩下的名额留给
    ``recent_record_ids``。原因是刚嵌完就来验证的图往往被压缩或改动过，
    特征匹配分数可能不高、挤不进前列，但它恰恰是最可能的答案。
    """
    if type(query) is not FeatureIndex:
        raise TypeError("query feature index must be an exact FeatureIndex")
    if type(candidates) is not tuple:
        raise TypeError("feature candidates must be a tuple")
    if type(recent_record_ids) is not tuple or any(
        type(value) is not str or not value for value in recent_record_ids
    ):
        raise ValueError("recent record IDs must be nonempty strings in a tuple")
    if type(config) is not V4Config:
        raise TypeError("config must be an exact V4Config")

    ranked = []
    seen_ids: set[str] = set()
    for record_id, index in candidates:
        if type(record_id) is not str or not record_id or record_id in seen_ids:
            raise ValueError("candidate record IDs must be unique nonempty strings")
        if type(index) is not FeatureIndex:
            raise TypeError("candidate indexes must be exact FeatureIndex instances")
        seen_ids.add(record_id)
        good = _good_descriptor_matches(
            query.descriptors[:FEATURE_COARSE_DESCRIPTORS],
            index.descriptors,
        )
        match_count = len(good)
        # 除以两侧描述子数的**较小值**：分母取小的一方，
        # 才能抵消"某张图特征特别多所以天然容易匹配上"的偏差。
        # max(1, ...) 防止空索引导致除零。
        match_quality = match_count / max(
            1,
            min(len(query.descriptors), len(index.descriptors)),
        )
        # 缩略图平均绝对差，归一化到 0~1，作为整体外观相似度的辅助信号
        thumbnail_distance = float(
            np.mean(
                np.abs(
                    query.thumbnail.astype(np.float32)
                    - index.thumbnail.astype(np.float32)
                )
            )
            / 255.0
        )
        ranked.append(
            RankedFeatureCandidate(
                record_id=record_id,
                index=index,
                match_count=match_count,
                match_quality=float(match_quality),
                thumbnail_distance=thumbnail_distance,
                reserved=False,
            )
        )
    # 四级排序键：匹配数降序 → 匹配质量降序 → 缩略图距离升序 → ID 升序。
    # 最后一级用 ID 保证**结果完全确定**：同分候选的顺序在任何机器、
    # 任何次运行上都一致，检测结果因此可复现。
    ranked.sort(
        key=lambda item: (
            -item.match_count,
            -item.match_quality,
            item.thumbnail_distance,
            item.record_id,
        )
    )
    # 先按分数取前 2 名（candidate_limit 更小时以它为准）
    selected = ranked[: min(2, config.candidate_limit)]
    selected_ids = {item.record_id for item in selected}
    by_id = {item.record_id: item for item in ranked}
    # 剩余名额留给近期记录；循环体末尾无条件 break，故最多补 1 个
    for record_id in recent_record_ids:
        if len(selected) >= config.candidate_limit:
            break
        if record_id in selected_ids or record_id not in by_id:
            continue
        item = by_id[record_id]
        # 重建一份并标记 reserved=True，表明它是靠保留席位入选、
        # 而非凭分数挤进来的，便于事后分析检测命中的来源分布。
        selected.append(
            RankedFeatureCandidate(
                record_id=item.record_id,
                index=item.index,
                match_count=item.match_count,
                match_quality=item.match_quality,
                thumbnail_distance=item.thumbnail_distance,
                reserved=True,
            )
        )
        selected_ids.add(record_id)
        break
    return tuple(selected)


def _good_descriptor_matches(
    query_descriptors: np.ndarray,
    target_descriptors: np.ndarray,
) -> list[cv2.DMatch]:
    """描述子匹配 + Lowe 比率检验，返回可靠的匹配对。

    **Lowe 比率检验**：对每个查询描述子取最近的两个目标描述子，
    只有当最近邻明显优于次近邻（距离 < 0.75 倍）时才接受。

    道理在于——如果一个特征在目标图里有两个同样像的对象，
    那它多半是重复纹理（窗格、砖墙、文字），无法确定对应关系，
    留着只会污染 RANSAC。0.75 是 Lowe 原论文的推荐值。

    描述子少于 2 个时无法做比率检验（凑不出次近邻），直接返回空。
    """
    if len(query_descriptors) < 2 or len(target_descriptors) < 2:
        return []
    # ORB 是二进制描述子，必须用汉明距离；欧氏距离在此毫无意义。
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(query_descriptors, target_descriptors, k=2)
    return [
        first
        for pair in pairs
        # 边缘情况：目标描述子过少时 knnMatch 可能只返回 1 个结果
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]


def _validate_feature_pair(query: FeatureIndex, target: FeatureIndex) -> None:
    """两侧都必须是 :class:`FeatureIndex` 本身，不接受子类。"""
    if type(query) is not FeatureIndex or type(target) is not FeatureIndex:
        raise TypeError("feature matching requires exact FeatureIndex instances")


def _plausible_homography(
    matrix: np.ndarray,
    query: FeatureIndex,
    target: FeatureIndex,
) -> bool:
    """判断单应矩阵在几何上是否说得通，挡住"数学成立但物理荒谬"的解。

    RANSAC 在点分布退化时会拟合出把整幅图压成一条线、翻到画布外、
    或放大几百倍的矩阵。这些矩阵重投影误差可能很小，却毫无实际意义。
    四道检查按代价从低到高排列，早失败早返回：

    1. **形状与有限性**——基本合法性；
    2. **行列式**——接近零意味着退化（平面被压扁）；
    3. **条件数**——大于 1e8 说明矩阵病态，微小扰动会导致结果剧变；
    4. **投影面积与位置**——把查询图四角映射过去，
       面积须在目标图的 2%~400% 之间，且不能整体飞出画布太远。

    2%~400% 这个区间对应"从小截图到大幅放大"的合理范围；
    位置检查允许溢出到画布外一个身位，因为部分重叠是正常的。
    """
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return False
    if abs(float(matrix[2, 2])) < 1e-9:
        return False
    normalized = matrix / matrix[2, 2]
    determinant = float(np.linalg.det(normalized))
    if not np.isfinite(determinant) or abs(determinant) < 1e-8:
        return False
    if float(np.linalg.cond(normalized)) > 1e8:
        return False
    # 把查询图的四个角投影到目标坐标系，检查落点是否合理
    corners = np.asarray(
        [
            [[0.0, 0.0]],
            [[float(query.image_width), 0.0]],
            [[float(query.image_width), float(query.image_height)]],
            [[0.0, float(query.image_height)]],
        ],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(corners, normalized).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return False
    area = abs(float(cv2.contourArea(projected.astype(np.float32))))
    target_area = float(target.image_width * target.image_height)
    if not 0.02 * target_area <= area <= 4.0 * target_area:
        return False
    if (
        np.any(projected[:, 0] < -target.image_width)
        or np.any(projected[:, 0] > 2 * target.image_width)
        or np.any(projected[:, 1] < -target.image_height)
        or np.any(projected[:, 1] > 2 * target.image_height)
    ):
        return False
    return True


def _readonly_copy(values: np.ndarray, dtype: np.dtype[object] | type) -> np.ndarray:
    """复制成指定类型的 C 连续只读数组。

    ``order="C"`` 保证连续布局（OpenCV 要求），``writeable = False``
    让数组内容无法被外部改动，配合冻结数据类实现真正的不可变。
    """
    copied = np.array(values, dtype=dtype, copy=True, order="C")
    copied.flags.writeable = False
    return copied


def _spatially_balanced_keypoint_indices(
    keypoints: tuple[cv2.KeyPoint, ...] | list[cv2.KeyPoint],
    width: int,
    height: int,
) -> list[int]:
    """按空间与尺度均衡地挑选关键点，避免全挤在图像的某一块。

    :return: 选中关键点的下标列表，最多 ``FEATURE_INDEX_MAX_DESCRIPTORS`` 个。

    **为什么不能只按响应强度排序取前 N。** ORB 的响应值在高对比度区域
    （文字、边缘密集处）普遍偏高，纯按强度选点会让特征全部堆在图片的一小块。
    一旦这块恰好被裁掉或遮挡，整张索引就失效了；而且点集聚在一处时，
    RANSAC 拟合的几何是"局部最优、全局跑偏"的。

    **三轮选取，配额逐轮放宽**：

    1. 严格模式——每格有数量配额，且每格每个尺度层最多 6 个
       （防止同一格里全是同一尺度的点）；
    2. 放宽模式——只保留格配额，不再限制尺度层；
    3. 兜底模式——配额全部取消，按响应强度补足名额。

    这样既保证了空间覆盖，又不会因为约束太严而白白浪费名额。
    """
    # 只考虑响应最强的前 4 倍名额，再多也轮不上，先截断以省排序开销
    available = min(len(keypoints), FEATURE_INDEX_MAX_DESCRIPTORS * 4)
    # 按响应降序；次级键用下标保证同分时顺序稳定、结果可复现
    response_order = sorted(
        range(available),
        key=lambda index: (-float(keypoints[index].response), index),
    )
    # 每格配额 = 总名额 / 格数（8×8=64），即每格约 48 个
    quota = max(
        1,
        FEATURE_INDEX_MAX_DESCRIPTORS // (FEATURE_SELECTION_GRID**2),
    )
    counts = np.zeros((FEATURE_SELECTION_GRID, FEATURE_SELECTION_GRID), dtype=np.int16)
    # 逐格分尺度层计数（ORB 金字塔 8 层）
    octave_counts = np.zeros(
        (FEATURE_SELECTION_GRID, FEATURE_SELECTION_GRID, 8),
        dtype=np.int16,
    )
    selected: list[int] = []
    selected_set: set[int] = set()
    # 第一轮：格配额 + 尺度层配额，双重约束
    for index in response_order:
        x, y = keypoints[index].pt
        column = min(FEATURE_SELECTION_GRID - 1, int(x * FEATURE_SELECTION_GRID / width))
        row = min(FEATURE_SELECTION_GRID - 1, int(y * FEATURE_SELECTION_GRID / height))
        # octave 夹到 0~7：某些 OpenCV 版本会给出负值或超范围值
        octave = min(7, max(0, int(keypoints[index].octave)))
        if counts[row, column] >= quota or octave_counts[row, column, octave] >= 6:
            continue
        counts[row, column] += 1
        octave_counts[row, column, octave] += 1
        selected.append(index)
        selected_set.add(index)
        if len(selected) >= FEATURE_INDEX_MAX_DESCRIPTORS:
            return selected
    # 第二轮：放弃尺度层约束，只守格配额。
    # 用于补上那些"格子还有空位、但某个尺度层已满"而被第一轮跳过的点。
    for index in response_order:
        if index in selected_set:
            continue
        x, y = keypoints[index].pt
        column = min(FEATURE_SELECTION_GRID - 1, int(x * FEATURE_SELECTION_GRID / width))
        row = min(FEATURE_SELECTION_GRID - 1, int(y * FEATURE_SELECTION_GRID / height))
        if counts[row, column] >= quota:
            continue
        counts[row, column] += 1
        selected.append(index)
        selected_set.add(index)
        if len(selected) >= FEATURE_INDEX_MAX_DESCRIPTORS:
            return selected
    # 第三轮兜底：配额全部放开，按响应强度补足名额。
    # 图片内容集中在少数几格时（如白底居中的商品图），
    # 前两轮会因格配额限制而选不满，这里保证名额不被浪费。
    for index in response_order:
        if index in selected_set:
            continue
        selected.append(index)
        if len(selected) >= FEATURE_INDEX_MAX_DESCRIPTORS:
            break
    return selected


__all__ = (
    "FEATURE_INDEX_SCHEMA_VERSION",
    "FeatureIndex",
    "FeatureMatch",
    "extract_feature_index",
    "load_feature_index",
    "match_feature_indexes",
    "match_feature_indexes_constrained",
    "rank_feature_candidates",
    "save_feature_index",
)
