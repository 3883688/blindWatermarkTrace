"""V4 检测器：把几何对齐、码字解码与候选认证串成完整的提取流水线。

这是 V4 提取侧的总入口，负责回答"这张可疑图属于哪条记录"。

**双路几何对齐。** 要解码 DCT 码字，必须先把可疑图摆回原图的坐标系。
本模块并行准备两条路：

* **ORB 特征匹配**——提取图像关键点，与候选记录存档的特征索引做匹配，
  用 RANSAC 拟合单应矩阵。适应性最强，能处理透视形变，但依赖图像纹理，
  在纯色或重复纹理的图上会失效。
* **FFT 同步导频**——见 :mod:`watermark_v4.sync`，给出旋转/缩放/平移。
  不依赖图像内容，但只能处理相似变换。

先试 ORB，失败再用导频参数去约束一次受限匹配，两路互为补充。
最终结果里的 ``geometry_method`` 字段记录了实际走通的是哪条路。

**唯一性要求。** :func:`detect_v4` 只在**恰好一个**候选通过认证时才返回结果。
零个是没找到；两个及以上说明出现了不该发生的多重匹配，此时宁可判定失败，
也不能随便挑一个——溯源结论指错了人，比查不到更严重。
"""

from dataclasses import dataclass
from time import monotonic

import cv2
import numpy as np
from PIL import Image

from .config import V4Config
from .dct import extract_tile_scores
from .features import (
    FeatureIndex,
    extract_feature_index,
    match_feature_indexes,
    match_feature_indexes_constrained,
    rank_feature_candidates,
)
from .payload import decode_candidate_codeword, phase_for_tile, phase_permutation
from .sync import detect_pilot


@dataclass(frozen=True, slots=True)
class V4Candidate:
    """一条待验证的候选记录：认证标签 + 原图特征索引。

    检测的本质是"在候选名单里找出唯一吻合的那个"，因此每个候选必须同时
    携带**认证标签**（用于码字比对）和**特征索引**（用于几何对齐）。
    """

    record_id: str
    trace_id: str
    # 4 字节 HMAC 认证标签，与嵌入时写进图里的净荷相同
    auth_tag: bytes
    # 原图的 ORB 特征索引，用于把可疑图配准回原始坐标系
    feature_index: FeatureIndex

    def __post_init__(self) -> None:
        """校验标识非空规范、标签长度、特征索引类型。

        标识要求 ``value == value.strip()``：带首尾空白的 ID 会让
        字典查找与日志比对静默失配，此类问题在现场极难定位。
        """
        for name, value in (("record ID", self.record_id), ("trace ID", self.trace_id)):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be a nonempty canonical string")
        if type(self.auth_tag) is not bytes or len(self.auth_tag) != 4:
            raise ValueError("candidate auth tag must contain exactly 4 bytes")
        if type(self.feature_index) is not FeatureIndex:
            raise TypeError("candidate feature index must be an exact FeatureIndex")


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """单个候选通过认证后的证据明细，用于评估这次判定有多可靠。

    这些指标不影响"是否命中"的判定（那由 RS 解码与标签比对决定），
    而是给出**命中质量**：参与的分块越多、相位越全、纠错动用得越少、
    比特错误越少，结论就越硬。取证场景需要能出示这些量化依据。
    """

    record_id: str
    trace_id: str
    # 有多少个分块参与了聚合
    tile_count: int
    # 覆盖了几种相位
    phase_count: int
    # 参与分块中最低的那个有效像素覆盖率
    minimum_coverage: float
    # RS 纠正的符号数、动用的擦除数、观测与理论码字的汉明距离
    corrected_symbols: int
    erasure_count: int
    bit_errors: int
    # 聚合分数的平均绝对值，反映信号整体强度
    mean_abs_score: float

    def __post_init__(self) -> None:
        """校验各指标为非负整数 / 有限非负浮点，覆盖率不超过 1。"""
        for name, value in (("record ID", self.record_id), ("trace ID", self.trace_id)):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty string")
        for name, value in (
            ("tile count", self.tile_count),
            ("phase count", self.phase_count),
            ("corrected symbols", self.corrected_symbols),
            ("erasure count", self.erasure_count),
            ("bit errors", self.bit_errors),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name, value in (
            ("minimum coverage", self.minimum_coverage),
            ("mean absolute score", self.mean_abs_score),
        ):
            if type(value) is not float or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        if self.minimum_coverage > 1.0:
            raise ValueError("minimum coverage must not exceed one")


@dataclass(frozen=True, slots=True)
class V4Detection:
    """V4 检测的最终结论，在 :class:`CandidateEvidence` 之上补充几何与耗时信息。"""

    record_id: str
    trace_id: str
    codec: str
    # 实际走通的对齐路径："fft_orb_ransac"（有导频辅助）或 "orb_ransac"（纯特征）
    geometry_method: str
    # RANSAC 内点数与内点占比，衡量几何对齐的可靠程度
    orb_inliers: int
    orb_ratio: float
    # 参与排序的候选总数，配合 candidate_match_probability 估算误报率
    candidate_count: int
    tile_count: int
    phase_count: int
    corrected_symbols: int
    erasure_count: int
    bit_errors: int
    mean_abs_score: float
    # 同步置信度；未用到导频时为 None
    sync_confidence: float | None
    elapsed_seconds: float

    def __post_init__(self) -> None:
        """校验全部字段，其中 ``geometry_method`` 限定为两个合法取值。"""
        if type(self.record_id) is not str or not self.record_id:
            raise ValueError("detection record ID must be a nonempty string")
        if type(self.trace_id) is not str or not self.trace_id:
            raise ValueError("detection trace ID must be a nonempty string")
        if type(self.codec) is not str or not self.codec:
            raise ValueError("detection codec must be a nonempty string")
        if self.geometry_method not in ("fft_orb_ransac", "orb_ransac"):
            raise ValueError("detection geometry method is invalid")
        for name, value in (
            ("ORB inliers", self.orb_inliers),
            ("candidate count", self.candidate_count),
            ("tile count", self.tile_count),
            ("phase count", self.phase_count),
            ("corrected symbols", self.corrected_symbols),
            ("erasure count", self.erasure_count),
            ("bit errors", self.bit_errors),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name, value in (
            ("ORB ratio", self.orb_ratio),
            ("mean absolute score", self.mean_abs_score),
            ("elapsed seconds", self.elapsed_seconds),
        ):
            if type(value) is not float or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        if self.orb_ratio > 1.0:
            raise ValueError("ORB ratio must not exceed one")
        if self.sync_confidence is not None and (
            type(self.sync_confidence) is not float
            or not np.isfinite(self.sync_confidence)
            or not 0.0 <= self.sync_confidence <= 1.0
        ):
            raise ValueError("sync confidence must be None or a float between zero and one")


def decode_aligned_candidate(
    image: Image.Image,
    query_to_target: np.ndarray,
    candidate: V4Candidate,
    config: V4Config,
    *,
    deadline: float | None = None,
) -> CandidateEvidence | None:
    """按给定单应矩阵把可疑图配准到候选原图坐标系，然后解码验证。

    :param query_to_target: 3×3 单应矩阵，把可疑图坐标映射到候选原图坐标。
    :return: 认证通过返回证据，否则 ``None``。

    五步：

    1. **透视变换**，把可疑图摆回原图尺寸与朝向；
    2. **有效性掩码**，标出哪些像素来自真实内容、哪些是变换后的空白填充；
    3. **逐分块提取**，跳过覆盖率不足的分块，逐块读出软判决分数；
    4. **归一化聚合**，各分块按自身强度归一后取平均；
    5. **RS 解码 + 标签比对**，通过才算认证成功。

    掩码这一步是关键。可疑图往往只是原图的一部分（裁剪、局部截图），
    变换后画布上会有大片空白。若不识别这些区域，空白处读出的"分数"
    实为纯噪声，会把聚合结果污染到解不出码字。
    """
    _validate_image(image)
    matrix = _validated_homography(query_to_target)
    if type(candidate) is not V4Candidate:
        raise TypeError("candidate must be an exact V4Candidate")
    if type(config) is not V4Config:
        raise TypeError("config must be an exact V4Config")
    _validate_deadline(deadline)
    _check_deadline(deadline)

    target_width = candidate.feature_index.image_width
    target_height = candidate.feature_index.image_height
    rgb = np.asarray(image)[..., :3]
    # 变换到候选原图的尺寸。双线性插值在重采样精度与速度间取平衡；
    # 空白区域填 0。
    warped_rgb = cv2.warpPerspective(
        rgb,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # 用一张全白图走**完全相同**的变换，得到有效区域掩码。
    # 这里必须用最近邻插值：双线性会在边缘产生 0~255 的过渡灰阶，
    # 让"有效/无效"的界线变得模糊，覆盖率也就算不准了。
    source_mask = np.full((image.height, image.width), 255, dtype=np.uint8)
    valid_mask = cv2.warpPerspective(
        source_mask,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    _check_deadline(deadline)
    luminance = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2YCrCb)[..., 0]

    logical_batches: list[np.ndarray] = []
    coverages: list[float] = []
    phases: set[int] = set()
    tile_size = config.tile_size
    for tile_y in range(target_height // tile_size):
        for tile_x in range(target_width // tile_size):
            _check_deadline(deadline)
            top = tile_y * tile_size
            left = tile_x * tile_size
            coverage = float(
                np.mean(
                    valid_mask[
                        top : top + tile_size,
                        left : left + tile_size,
                    ]
                    > 0
                )
            )
            # 覆盖率不足的分块直接丢弃：它有相当部分是变换空白，
            # 读出来的分数是噪声，参与聚合只会拉低整体信噪比。
            if coverage < config.minimum_coverage:
                continue
            physical = np.asarray(
                extract_tile_scores(
                    luminance[
                        top : top + tile_size,
                        left : left + tile_size,
                    ],
                    config,
                ),
                dtype=np.float64,
            )
            # 相位由分块坐标决定；用置换表当索引即完成逆置换
            phase = phase_for_tile(tile_x, tile_y)
            logical = physical[np.asarray(phase_permutation(phase), dtype=np.intp)]
            # 用**中位数**绝对值作为该块的信号强度基准。
            # 中位数不受少数极端值影响：某几个比特被局部损伤破坏时，
            # 均值会被拉偏，中位数仍能反映这块的真实强度水平。
            robust_energy = float(np.median(np.abs(logical)))
            # 强度接近零说明这块几乎没有信号（纯色区域等），跳过。
            if not np.isfinite(robust_energy) or robust_energy <= 1e-9:
                continue
            # 除以自身强度做归一化后再聚合。这一步让强块与弱块获得**同等
            # 权重**：否则一个高对比度分块会主导整个平均值，而它未必更可信。
            logical_batches.append(logical / robust_energy)
            coverages.append(coverage)
            phases.add(phase)

    # 样本量不足：分块太少或相位太单一，冗余不够，结论不可信。
    if (
        len(logical_batches) < config.minimum_tiles
        or len(phases) < config.minimum_phases
    ):
        return None
    # 所有分块逐比特平均。因为每块嵌的都是同一个码字，这就是一次
    # 相干累加：信号同向叠加、噪声互相抵消，信噪比随分块数提升。
    aggregate = np.mean(np.stack(logical_batches), axis=0)
    observed = _scores_to_bytes(aggregate)
    # 每字节（8 比特一组）的置信度，取该组中**最弱的那一位**——
    # 一个字节只要有一位读错，整字节就错了，所以短板决定成败。
    # 再用 m/(1+m) 把 [0, ∞) 压缩到 [0, 1)：这是一条单调饱和曲线，
    # 弱信号区分辨率高，强信号区趋近 1，正好符合置信度的语义。
    byte_confidences = tuple(
        float(
            minimum
            / (1.0 + minimum)
        )
        for minimum in (
            np.min(np.abs(aggregate[start : start + 8]))
            for start in range(0, 64, 8)
        )
    )
    decoded = decode_candidate_codeword(
        observed,
        candidate.auth_tag,
        byte_confidences,
    )
    if decoded is None:
        return None
    return CandidateEvidence(
        record_id=candidate.record_id,
        trace_id=candidate.trace_id,
        tile_count=len(logical_batches),
        phase_count=len(phases),
        minimum_coverage=float(min(coverages)),
        corrected_symbols=decoded.corrected_symbols,
        erasure_count=decoded.erasure_count,
        bit_errors=decoded.bit_errors,
        mean_abs_score=float(np.mean(np.abs(aggregate))),
    )


def detect_v4(
    image: Image.Image,
    candidates: tuple[V4Candidate, ...],
    config: V4Config,
    *,
    recent_record_ids: tuple[str, ...] = (),
    deadline: float | None = None,
) -> V4Detection | None:
    """V4 检测总入口：在候选名单中找出唯一吻合的记录。

    :param candidates: 候选记录，``record_id`` 必须互不重复。
    :param recent_record_ids: 近期生成的记录，在候选排序中优先——
        "刚嵌完就来验证"是最高频的使用场景，优先试它们能显著缩短耗时。
    :param deadline: 外部截止时刻；与配置的硬超时取更早者。
    :return: 恰好一个候选通过时返回结论，否则 ``None``。

    对排序靠前的 ``candidate_limit`` 个候选逐一尝试，每个候选最多走两条路：
    先 ORB 特征匹配（并附带九宫格平移微调），失败则用导频参数做受限匹配。

    候选数量硬性设限，因为每多验一个候选，误报概率就线性上升一分
    （见 :func:`watermark_v4.payload.candidate_match_probability`），
    同时耗时也线性增长。
    """
    _validate_image(image)
    if type(candidates) is not tuple or any(
        type(candidate) is not V4Candidate for candidate in candidates
    ):
        raise TypeError("v4 candidates must be a tuple of exact V4Candidate instances")
    if type(config) is not V4Config:
        raise TypeError("config must be an exact V4Config")
    _validate_deadline(deadline)
    started = monotonic()
    # 取外部截止与配置硬超时中更早的那个：外部可以更严，但不能突破硬上限。
    hard_deadline = started + config.hard_timeout_seconds
    effective_deadline = hard_deadline if deadline is None else min(deadline, hard_deadline)
    _check_deadline(effective_deadline)
    if not candidates:
        return None

    # 查询图的特征只提一次，供后面所有候选复用（这是较重的一步）。
    query_index = extract_feature_index(image)
    _check_deadline(effective_deadline)
    # 导频检测同样只做一次，与具体候选无关。失败返回 None，不影响主流程。
    sync = detect_pilot(image, config, deadline=effective_deadline)
    candidate_by_id = {candidate.record_id: candidate for candidate in candidates}
    # ID 重复会让后面的字典查找悄悄丢候选，属于调用方的数据错误，直接报错。
    if len(candidate_by_id) != len(candidates):
        raise ValueError("v4 candidate record IDs must be unique")
    # 先用轻量的特征相似度给候选排序，把最可能的排前面，
    # 再对前 N 个做昂贵的完整验证。
    ranked = rank_feature_candidates(
        query_index,
        tuple(
            (candidate.record_id, candidate.feature_index)
            for candidate in candidates
        ),
        recent_record_ids=recent_record_ids,
        config=config,
    )

    authenticated = []
    for ranked_candidate in ranked[: config.candidate_limit]:
        _check_deadline(effective_deadline)
        candidate = candidate_by_id[ranked_candidate.record_id]
        feature_match = match_feature_indexes(
            query_index,
            candidate.feature_index,
        )
        evidence = None
        matches = [] if feature_match is None else [feature_match]
        # ---- 路径一：ORB 特征匹配 + 平移微调 ----
        # RANSAC 给出的平移量常有亚像素级偏差，而 DCT 网格对错位极其敏感——
        # 差一个像素就可能整块解不出。因此围绕原始矩阵试探九宫格邻域，
        # 任一命中即停。
        if feature_match is not None:
            for matrix in _translation_refinements(feature_match.query_to_target):
                _check_deadline(effective_deadline)
                evidence = decode_aligned_candidate(
                    image,
                    matrix,
                    candidate,
                    config,
                    deadline=effective_deadline,
                )
                if evidence is not None:
                    break
        # ---- 路径二：导频约束下的受限匹配 ----
        # ORB 在低纹理图上会失败或给出错误矩阵；此时把导频估出的
        # 旋转/缩放/偏移作为约束交给匹配器，大幅缩小搜索空间。
        if evidence is None and sync is not None:
            constrained = match_feature_indexes_constrained(
                query_index,
                candidate.feature_index,
                rotation_degrees=sync.rotation_degrees,
                scale=sync.scale,
                tile_size=config.tile_size,
                tile_offset=(sync.offset_x, sync.offset_y)
                if sync.offset_x is not None and sync.offset_y is not None
                else None,
            )
            if constrained is not None:
                matches.append(constrained)
                _check_deadline(effective_deadline)
                evidence = decode_aligned_candidate(
                    image,
                    constrained.query_to_target,
                    candidate,
                    config,
                    deadline=effective_deadline,
                )
        if evidence is not None:
            # matches[-1] 是最终奏效的那次匹配（受限匹配若成功会追加在后）
            authenticated.append((evidence, matches[-1]))
    # 严格要求**恰好一个**候选通过：
    #   0 个 —— 没检出；
    #   ≥2 个 —— 出现多重匹配，认证机制本不应允许这种情况发生，
    #            说明数据或图像存在异常，此时任意挑一个都可能指错人。
    # 两种情形都返回 None，交由上层走其他检测手段。
    if len(authenticated) != 1:
        return None

    evidence, feature_match = authenticated[0]
    return V4Detection(
        record_id=evidence.record_id,
        trace_id=evidence.trace_id,
        codec=config.codec,
        geometry_method="fft_orb_ransac" if sync is not None else "orb_ransac",
        orb_inliers=feature_match.inliers,
        orb_ratio=feature_match.inlier_ratio,
        candidate_count=len(ranked),
        tile_count=evidence.tile_count,
        phase_count=evidence.phase_count,
        corrected_symbols=evidence.corrected_symbols,
        erasure_count=evidence.erasure_count,
        bit_errors=evidence.bit_errors,
        mean_abs_score=evidence.mean_abs_score,
        sync_confidence=None if sync is None else sync.confidence,
        elapsed_seconds=float(monotonic() - started),
    )


def _translation_refinements(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    """围绕给定单应矩阵生成一组平移微调版本，供逐一试解。

    :return: 最多 11 个矩阵，按尝试优先级排列——原始矩阵、
        平移取整版、以及八个方向各偏 1 像素的版本。

    只动矩阵的第三列（平移分量），旋转与缩放部分保持不变。

    取整版排在方向偏移之前：DCT 网格本就按整像素对齐，
    把亚像素平移归到最近整数往往一次就能命中。
    """
    refinements = [matrix]
    rounded = matrix.copy()
    rounded[0, 2] = round(float(rounded[0, 2]))
    rounded[1, 2] = round(float(rounded[1, 2]))
    # 原矩阵平移量本就是整数时，取整版与之相同，无需重复尝试
    if not np.array_equal(rounded, matrix):
        refinements.append(rounded)
    # 八邻域：先四个正交方向（更常见），再四个对角方向
    for offset_x, offset_y in (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ):
        refined = matrix.copy()
        refined[0, 2] += offset_x
        refined[1, 2] += offset_y
        refinements.append(refined)
    return tuple(refinements)


def _scores_to_bytes(scores: np.ndarray) -> bytes:
    """把 64 个软判决分数硬判决成 8 字节码字。

    只看符号：``> 0`` 为 1，否则为 0（强度信息另行用于计算字节置信度）。
    位序高位在前，与 :func:`watermark_v4.payload.bytes_to_bits` 严格对应。

    ``dtype=np.uint16`` 用于求和的中间结果：8 位权重相加最大 255，
    在 ``uint8`` 下正好卡在边界上，用 16 位累加再收窄可确保不溢出。
    """
    bits = (scores > 0.0).astype(np.uint8).reshape(8, 8)
    weights = (1 << np.arange(7, -1, -1, dtype=np.uint8))[None, :]
    return bytes(np.sum(bits * weights, axis=1, dtype=np.uint16).astype(np.uint8))


def _validated_homography(value: np.ndarray) -> np.ndarray:
    """校验并复制单应矩阵。

    行列式接近零意味着矩阵退化（把整个平面压成一条线），
    这种变换不可逆，用它做配准只会得到无意义的结果。

    ``copy=True`` 是必需的：:func:`_translation_refinements` 会就地修改
    矩阵元素，不复制就会污染调用方持有的对象。
    """
    if type(value) is not np.ndarray:
        raise TypeError("query-to-target homography must be a NumPy array")
    if value.shape != (3, 3) or value.dtype.kind not in "f":
        raise ValueError("query-to-target homography must be a 3x3 floating matrix")
    matrix = value.astype(np.float64, copy=True)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-10:
        raise ValueError("query-to-target homography must be finite and nonsingular")
    return matrix


def _validate_image(image: Image.Image) -> None:
    """待检测图片必须是 RGB 或 RGBA 模式的 PIL 图像。"""
    if type(image) is not Image.Image:
        raise TypeError("query image must be an exact PIL Image")
    if image.mode not in ("RGB", "RGBA"):
        raise ValueError("query image mode must be RGB or RGBA")


def _validate_deadline(deadline: float | None) -> None:
    """截止时刻必须是有限数或 ``None``。"""
    if deadline is None:
        return
    if type(deadline) not in (int, float) or not np.isfinite(deadline):
        raise TypeError("deadline must be a finite monotonic timestamp")


def _check_deadline(deadline: float | None) -> None:
    """超时即抛 :class:`TimeoutError`；在每个候选、每次试解前都检查。"""
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("v4 detection deadline expired")


__all__ = (
    "CandidateEvidence",
    "V4Candidate",
    "V4Detection",
    "decode_aligned_candidate",
    "detect_v4",
)
