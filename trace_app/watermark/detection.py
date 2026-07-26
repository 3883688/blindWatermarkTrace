"""水印检测流水线：按优先级依次调用各检测器，命中即停。

本模块是提取侧的**调度中枢**，自身不含任何图像算法——所有检测器都以回调
形式传入（见 :func:`extract_watermark_from_image` 那一长串参数），
这里只负责决定"先试谁、什么条件下才试、命中后如何组装证据"。

**两条互斥的主干**（见 :func:`extract_watermark_from_image`）：

* 存在 V4 候选 → 只走 V4 认证路径，成败都不再回退。
  V4 有密码学认证，它说没有就是没有；再去试那些基于相关性的旧算法，
  只会徒增误报。
* 无 V4 候选 → 走传统多级回退链：LSB → 点阵 → 几何对齐 → 视觉匹配 →
  小裁剪 → 编码层 → 鲁棒水印 → 残差 → 明水印 → 分块 LSB。

顺序原则是"**先快后慢、先确定后模糊**"：能直接读出数据的排前面，
靠相似度推断的排后面。

多处 ``state_value(...)`` 开关允许运维在线关停某些昂贵的回退，
用召回率换响应时间。
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from PIL import Image
from watermark_v4.detector import V4Candidate


# 统一的检测器签名：吃一张图，命中返回证据字典，未命中返回 None
Detector = Callable[[Image.Image], dict[str, Any] | None]


@dataclass(frozen=True, slots=True)
class DetectionPipeline:
    """检测器的顺序组合器：按序调用，第一个非 ``None`` 结果即为结论。

    与 :func:`extract_watermark_from_image` 的区别在于——本类是**无条件**
    顺序执行的简单组合，不含任何开关或前置判断，适合组装固定的子链路。
    """

    detectors: tuple[Detector, ...]

    def detect(self, image: Image.Image) -> dict[str, Any] | None:
        """依次调用各检测器，命中即返回；全部落空返回 ``None``。"""
        for detector in self.detectors:
            result = detector(image)
            if result is not None:
                return result
        return None

    def __call__(self, image: Image.Image) -> dict[str, Any] | None:
        """让实例本身可当作 :data:`Detector` 使用，从而支持嵌套组合。"""
        return self.detect(image)


def v4_candidate_records(
    *,
    records: list[dict[str, Any]],
    data_dir: Path,
    version_v4: int,
    config_factory: Callable[[], Any],
    record_feature_index_path: Callable[[dict[str, Any], Path], Path | None],
    load_feature_index: Callable[[Path], Any | None],
) -> tuple[V4Candidate, ...]:
    """从记录集里筛出可用于 V4 检测的候选。

    :return: 通过全部校验的 :class:`V4Candidate` 元组。

    五道过滤，任一不满足就跳过该记录。用**静默跳过**而非报错，
    是因为记录集里混有旧版本、数据不全或索引已丢失的条目属于常态，
    不该因此中断整次检测：

    1. 版本号必须是 V4；
    2. codec 字符串必须与当前配置**完全一致**——它编码了纠错方案、
       置换规则等全部格式细节，不同 codec 的水印互不兼容；
    3. 记录 ID 与溯源号非空；
    4. 认证码必须是 8 个小写十六进制字符（即 4 字节）；
    5. 特征索引文件存在且能正常加载。
    """
    config = config_factory()
    candidates = []
    for record in records:
        if record.get("robust_watermark_version") != version_v4:
            continue
        if record.get("robust_watermark_codec") != config.codec:
            continue
        record_id = str(record.get("id") or "").strip()
        trace_id = str(record.get("trace_id") or "").strip()
        auth_hex = str(record.get("robust_auth_code") or "").strip()
        # 正则限定小写：写入时统一用 bytes.hex()（输出小写），
        # 大写值说明数据来路不明，一并排除
        if not record_id or not trace_id or not re.fullmatch(r"[0-9a-f]{8}", auth_hex):
            continue
        path = record_feature_index_path(record, data_dir)
        if path is None:
            continue
        # 索引文件缺失或损坏时无法做几何配准，这条候选没法验，跳过
        feature_index = load_feature_index(path)
        if feature_index is None:
            continue
        candidates.append(
            V4Candidate(
                record_id=record_id,
                trace_id=trace_id,
                auth_tag=bytes.fromhex(auth_hex),
                feature_index=feature_index,
            )
        )
    return tuple(candidates)


def detect_v4_watermark(
    image: Image.Image,
    candidates: tuple[V4Candidate, ...] | None = None,
    *,
    records: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]],
    generated_trace_ids: list[str],
    version_v4: int,
    config_factory: Callable[[], Any],
    candidate_records: Callable[[], tuple[V4Candidate, ...]],
    detect: Callable[..., Any],
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
    now_text: Callable[[], str],
) -> dict[str, Any] | None:
    """调用 V4 检测器，并把结果组装成统一格式的证据字典。

    :param candidates: 候选集；传 ``None`` 则调用 ``candidate_records()`` 现取。
    :param records: 记录集，可以是列表或返回列表的可调用对象——
        **延迟求值**很关键：V4 未命中时根本不需要读记录，
        传可调用对象可以省掉这次查库。
    :return: 命中返回证据字典，否则 ``None``。
    """
    available = candidate_records() if candidates is None else candidates
    if not available:
        return None
    # 按"最近生成"的顺序收集 record_id，供检测器做候选优先排序。
    # 外层遍历 generated_trace_ids（已按时间倒序），保证越新的排越前。
    recent_record_ids = tuple(
        candidate.record_id
        for trace_id in generated_trace_ids
        for candidate in available
        if candidate.trace_id == trace_id
    )
    result = detect(
        image.convert("RGB"),
        available,
        config_factory(),
        recent_record_ids=recent_record_ids,
    )
    if result is None:
        return None
    # 只有命中之后才真正去读记录集，未命中时省掉这次查询
    current_records = records() if callable(records) else records
    record_by_id = {
        str(record.get("id")): record
        for record in current_records
        if record.get("robust_watermark_version") == version_v4
    }
    record = record_by_id.get(result.record_id)
    if record is None:
        return None
    return with_evidence_fields({
        "id": result.record_id,
        "trace_id": result.trace_id,
        "user_id": record.get("user_id"),
        "mode": "v4_authenticated_dct",
        "mode_label": "V4 认证水印",
        "created_at": record.get("created_at"),
        # 置信度从 99 起扣，每个比特错误扣 4 分，下限 80。
        # 即便有错误也不低于 80，因为 RS 纠错 + HMAC 认证双重通过本身
        # 就是极强的证据；比特错误只反映信号质量，不动摇结论的成立。
        "confidence": max(80, min(99, 99 - result.bit_errors * 4)),
        "phash_match": False,
        "status": "V4 认证命中",
        "extracted_at": now_text(),
        "watermark_layers": record.get("watermark_layers"),
        "layer_scores": {},
        "code_recovery": {
            "method": result.geometry_method,
            "codec": result.codec,
            "candidate_count": result.candidate_count,
            "authenticated_tiles": result.tile_count,
            "phase_count": result.phase_count,
            "corrected_symbols": result.corrected_symbols,
            "erasure_count": result.erasure_count,
            "bit_errors": result.bit_errors,
            "mean_abs_score": round(result.mean_abs_score, 6),
            "orb_inliers": result.orb_inliers,
            "orb_ratio": round(result.orb_ratio, 6),
            "sync_confidence": result.sync_confidence,
            "elapsed_ms": round(result.elapsed_seconds * 1000.0, 3),
        },
    }, record)


def should_run_frequency_fallbacks(image: Image.Image) -> bool:
    """判断是否值得为这张图跑昂贵的频域回退检测。

    频域检测的耗时随像素数增长，大图上可能要好几秒，因此按规模设闸：

    * ≤ 300 万像素：一律跑；
    * 300 万 ~ 500 万像素：只有**长宽比 ≥ 2.2** 才跑。
      细长图通常是长截图、拼接图这类"局部内容"，
      恰恰最需要频域层来兜底，值得为它多花时间；
    * > 500 万像素：一概不跑。
    """
    width, height = image.size
    pixels = width * height
    if pixels <= 3_000_000:
        return True
    # max(1, ...) 防止零尺寸导致除零
    aspect = max(width, height) / max(1, min(width, height))
    return pixels <= 5_000_000 and aspect >= 2.2


def should_run_visual_match_fallback(
    image: Image.Image, *, records: list[dict[str, Any]]
) -> bool:
    """判断是否值得跑视觉匹配回退。

    两个前提缺一不可：

    1. 记录集里**至少有一条**启用了鲁棒水印——一条都没有的话，
       视觉匹配无从比对，跑了也是白跑；
    2. 图片不小于 40000 像素（约 200×200）。太小的图特征量不足，
       视觉匹配的结果不可信，反而容易误报。
    """
    if not any(record.get("robust_watermark") for record in records):
        return False
    width, height = image.size
    return width * height >= 40_000


def extract_watermark_from_image(
    image: Image.Image,
    *,
    records: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]],
    v4_candidates: tuple[V4Candidate, ...],
    detect_v4_watermark: Callable[..., dict[str, Any] | None],
    extract_full_lsb: Callable[[Image.Image], dict[str, Any] | None],
    extract_block_lsb: Callable[[Image.Image], dict[str, Any] | None],
    is_registered_original_image: Callable[[Image.Image], bool],
    should_run_frequency_fallbacks: Callable[[Image.Image], bool],
    should_run_visual_match_fallback: Callable[[Image.Image], bool],
    detect_dot_matrix_trace: Detector,
    detect_aligned_authenticated_watermark: Callable[..., dict[str, Any] | None],
    detect_by_visual_match: Detector,
    detect_small_crop_trace: Detector,
    detect_watermark_code: Detector,
    detect_robust_watermark: Detector,
    detect_by_residual_match: Detector,
    detect_visible_copyright: Detector,
    record_detection_result: Callable[[bool], None],
    with_evidence_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
    now_text: Callable[[], str],
    mode_label: Callable[[str], str],
    layer_scores_for_image: Callable[[Image.Image, str], Any],
    not_found_error: Callable[[], Exception],
    watermark_layers: Any,
    state_value: Callable[[str], Any],
) -> dict[str, Any]:
    """检测流水线主函数：按优先级尝试各检测器，返回第一个命中的证据。

    :raises Exception: 全部落空时抛出 ``not_found_error()`` 构造的异常。

    参数虽多，但全是注入的回调与开关，本函数只负责编排。
    每条出口（命中或落空）都会调用 ``record_detection_result``，
    保证看板上的检测成功率统计不漏计。

    **两条主干见模块文档。** 传统链路的回退顺序按"证据强度 × 代价"排：
    能直接读出数据的（LSB、点阵）在前，靠相似度推断的（视觉匹配、残差）
    在后，纯人工辅助的（明水印）垫底。
    """
    # ===== 主干一：V4 认证路径（存在 V4 候选时独占）=====
    if v4_candidates:
        # 上传的就是**未加水印的原图**——它确实在库里，但身上没有水印，
        # 这是"没检出"而非"命中"。若不特判，后面的相似度类检测器
        # 会因为图片高度相似而误报命中。
        if is_registered_original_image(image):
            record_detection_result(False)
            raise not_found_error()
        v4_match = detect_v4_watermark(image, v4_candidates)
        if v4_match:
            record_detection_result(True)
            return v4_match
        # V4 未命中即终止，**不回退**到传统算法：
        # V4 有密码学认证背书，它的否定结论是可信的；
        # 继续尝试基于相关性的旧算法只会引入误报。
        record_detection_result(False)
        raise not_found_error()

    # ===== 主干二：传统多级回退链 =====
    # 第一优先级：全图 LSB。命中即可直接拿到完整元信息，无需查库。
    payload = extract_full_lsb(image)
    if not payload:
        if is_registered_original_image(image):
            record_detection_result(False)
            raise not_found_error()
        # 频域系回退整体受规模开关控制
        if should_run_frequency_fallbacks(image):
            # 点阵层：抗翻拍，且检测代价相对可控，排在频域系最前
            dot_matrix_match = detect_dot_matrix_trace(image)
            if dot_matrix_match:
                record_detection_result(True)
                return dot_matrix_match
            # 几何对齐认证检测：带独立的候选数与时间预算配置
            if state_value("aligned_authenticated_detection_enabled"):
                aligned_match = detect_aligned_authenticated_watermark(
                    image,
                    candidate_limit=state_value("aligned_candidate_limit"),
                    budget_seconds=state_value("watermark_detection_budget_seconds"),
                )
                if aligned_match:
                    record_detection_result(True)
                    return aligned_match
            # 密集水印系：四个检测器共用一个开关，按代价递增排列
            if state_value("dense_watermark_fallback_enabled"):
                if should_run_visual_match_fallback(image):
                    visual_match = detect_by_visual_match(image)
                    if visual_match:
                        record_detection_result(True)
                        return visual_match
                # 小裁剪追踪：短码密铺全图，只截一小块也能还原
                small_crop_match = detect_small_crop_trace(image)
                if small_crop_match:
                    record_detection_result(True)
                    return small_crop_match
                # 编码层：空域短码
                code_match = detect_watermark_code(image)
                if code_match:
                    record_detection_result(True)
                    return code_match
                # 鲁棒水印：三层频域图案的相关性检测
                robust_match = detect_robust_watermark(image)
                if robust_match:
                    record_detection_result(True)
                    return robust_match
        # 视觉匹配的独立开关：与上面 dense 系里的那次是同一个检测器，
        # 但这里不受 should_run_frequency_fallbacks 的规模限制，
        # 因此大图在跳过所有频域回退后，仍有机会走这一条。
        if state_value("visual_match_fallback_enabled"):
            visual_match = detect_by_visual_match(image)
            if visual_match:
                record_detection_result(True)
                return visual_match
        # 残差匹配：无开关，始终执行。它比对的是与原图的像素差异模式，
        # 代价低且不依赖任何水印层存活，是最后一道数字兜底。
        residual_match = detect_by_residual_match(image)
        if residual_match:
            record_detection_result(True)
            return residual_match
        # 明水印识别：读图上可见的版权文字。证据强度最弱（文字可被伪造），
        # 故排在最末，且默认可关。
        if state_value("visible_watermark_detection_enabled"):
            fallback = detect_visible_copyright(image)
            if fallback:
                record_detection_result(True)
                return fallback
        # 所有回退都落空，最后再试一次分块 LSB（抗裁剪版本）。
        # 放在最末是因为它要扫描大量候选位置，代价高于前面各项。
        payload = extract_block_lsb(image)
    if not payload:
        record_detection_result(False)
        raise not_found_error()
    # ===== LSB 命中：用载荷里的溯源号回查记录 =====
    current_records = records() if callable(records) else records
    matched = next((item for item in current_records if item.get("trace_id") == payload.get("trace_id")), None)
    record_detection_result(True)
    return with_evidence_fields({
        "id": payload.get("id"),
        "trace_id": payload.get("trace_id"),
        "evidence_uuid": payload.get("evidence_uuid"),
        "evidence_uuid_head": payload.get("evidence_uuid_head"),
        "evidence_uuid_tail": payload.get("evidence_uuid_tail"),
        "user_id": payload.get("user_id"),
        "mode": payload.get("mode"),
        "mode_label": payload.get("mode_label", mode_label(payload.get("mode", "dct"))),
        "created_at": payload.get("created_at"),
        # 库里能查到对应记录 → 98；查不到 → 92。
        # 后者说明水印本身读出来了、格式也合法，但数据库里没有这条记录
        # （记录被删、或图片来自另一套部署）。水印可信，归属存疑，故略降。
        "confidence": 98 if matched else 92,
        "phash_match": bool(matched),
        "status": "匹配" if matched else "检测到水印",
        "extracted_at": now_text(),
        "watermark_layers": matched.get("watermark_layers", watermark_layers) if matched else watermark_layers,
        "layer_scores": layer_scores_for_image(image, payload.get("trace_id")) if payload.get("trace_id") else {},
    }, matched)
