"""水印服务的默认算子装配层。

:mod:`trace_app.watermark.service` 刻意不 import 任何具体算法，它只声明了
:class:`~trace_app.watermark.service.WatermarkOperations` 这个回调集合。本模块
就是那份集合的**默认实现**：把散落在 ``lsb`` / ``frequency`` / ``robust`` /
``small_crop`` / ``dot_matrix`` / ``imaging`` / ``watermark_v4`` 各处的函数收拢过来，
逐一绑定成字段，交给 ``WatermarkService`` 使用。

**这一层为什么必须存在：**

* **打破循环依赖。** 算法模块需要 config、repository、settings 里的常量与路径，
  而 service 又需要算法。若让 service 直接 import 算法，两边就会绕成环。
  把绑定动作挪到这个"叶子"模块，谁都不用反向依赖谁；
* **让测试能替换任意算子。** ``WatermarkOperations`` 是冻结数据类，测试只需构造
  一份字段值为假函数的实例，就能在不做任何真实 DCT/FFT 运算的前提下验证编排逻辑；
* **隔离签名差异。** 底层函数的签名五花八门（关键字参数、依赖注入钩子、需要
  ``settings.data_dir`` 之类的运行期路径），service 却要求统一的调用形态。
  本模块用 ``lambda`` / 闭包把这些差异吸收掉，底层函数改签名时只需改这里一处。

模块内还实现了若干**跨算法共用的小工具**（时间戳、布尔解析、数值夹紧、证据 UUID
拆分等）。它们放在这里而不是放进某个算法模块，是因为几乎每个算子都要用到，
放进任何一个具体模块都会制造新的依赖边。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from trace_app.imaging.candidate_feature_index import (
    extract_feature_descriptors,
    load_feature_descriptors,
    save_feature_descriptors,
)
from PIL import Image
from trace_app.watermark.auth import auth_code_from_trace
from watermark_v4 import (
    V4Config,
    authentication_tag as v4_authentication_tag,
    embed_codeword as embed_v4_codeword,
    embed_pilot as embed_v4_pilot,
    encode_codeword as encode_v4_codeword,
)
from watermark_v4.detector import detect_v4
from watermark_v4.features import (
    extract_feature_index as extract_v4_feature_index,
    load_feature_index as load_v4_feature_index,
    save_feature_index as save_v4_feature_index,
)

from trace_app.config import (
    CODE_WATERMARK_VERSION,
    DEFAULT_ROBUST_WATERMARK_STRENGTH,
    DEFAULT_WATERMARK_AUTH_KEY,
    DOT_MATRIX_VERSION,
    ROBUST_WATERMARK_CODEC_V2,
    ROBUST_WATERMARK_CODEC_V3,
    ROBUST_WATERMARK_VERSION_V1,
    ROBUST_WATERMARK_VERSION_V2,
    ROBUST_WATERMARK_VERSION_V3,
    ROBUST_WATERMARK_VERSION_V4,
    SMALL_TRACE_VERSION,
    WATERMARK_LAYERS,
    Settings,
)
from trace_app.database.repositories import Repository
from trace_app.imaging import (
    feature_matching,
    fingerprints,
    io,
    output,
    visible_mark,
)
from trace_app.runtime import Runtime
from trace_app.watermark import detection, dot_matrix, frequency, lsb, robust, small_crop
from trace_app.watermark.service import WatermarkOperations


def _now_text() -> str:
    """统一的时间戳格式化。

    嵌入时间、提取时间、各检测器写回的 ``extracted_at`` 全部走这一个函数，
    保证证据链里所有时间字段格式一致（本地时区，秒级精度），事后比对不用做格式转换。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _evidence_uuid_fields(evidence_uuid: str) -> dict[str, str]:
    """把证据 UUID 拆成"完整值 + 首 4 位 + 末 4 位"三个字段。

    :param evidence_uuid: 任意形态的 UUID 字符串，带不带连字符都行。
    :return: 可直接展开进记录字典的三个键。

    先去掉连字符再转大写，是为了让"带横线"和"不带横线"两种写法归一到同一个值，
    否则同一条证据在不同入口生成的字段会对不上。首尾两段供前端展示——
    对外只露片段，既能人工核对，又不至于把完整编号泄露出去被拼接盗用。
    """
    normalized = evidence_uuid.replace("-", "").upper()
    return {
        "evidence_uuid": normalized,
        "evidence_uuid_head": normalized[:4],
        "evidence_uuid_tail": normalized[-4:],
    }


def _with_evidence_fields(
    result: dict[str, Any], record: dict[str, Any] | None
) -> dict[str, Any]:
    """给检测结果补上证据 UUID 三件套。

    :param result: 某个检测器刚组装出的证据字典。
    :param record: 命中的数据库记录；为 ``None``（未命中任何记录）时原样返回。
    :return: 补全后的 ``result``——**就是传进来的那个对象**，原地修改。

    每个检测器只关心自己那套算法字段，证据编号统一在这里回填，避免十来个检测器
    各写一遍。两个前提缺一不可：记录里确实有值，且 ``result`` 里还没有值——
    检测器自己算出来的优先，这个函数只做兜底填空，不覆盖已有结论。
    """
    if record:
        for key in ("evidence_uuid", "evidence_uuid_head", "evidence_uuid_tail"):
            if record.get(key) and not result.get(key):
                result[key] = record[key]
    return result


def _parse_bool(raw: Any) -> bool:
    """把表单里的开关值解析成布尔。

    HTTP 表单字段一律是字符串，``bool("false")`` 会得到 ``True``，所以不能直接转换。
    白名单里同时收了英文写法和中文"启用"，因为前端在不同页面下发的值不统一。
    不在白名单里的一律当作**关闭**：开关类参数出错时保守选择不启用该层。
    """
    if isinstance(raw, bool):
        return raw
    return str(raw or "").lower() in {"1", "true", "yes", "on", "启用"}


def _clamp_float(value: Any, default: float, low: float, high: float) -> float:
    """解析浮点参数并夹紧到 ``[low, high]``。

    :param default: 解析失败时的回落值；注意它**同样要过夹紧**，
        所以调用方给的默认值即便越界也不会漏出去。
    :return: 一定落在区间内的浮点数。

    嵌入强度这类参数越界会直接毁掉图片（比如强度给成 100），因此所有来自请求的
    数值参数都必须过这道关。解析失败时回落到默认值而非报错，是因为这些参数多为
    可选项，前端不填或填错不该让整次嵌入失败。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _normalize_mode(raw: str) -> str:
    """把前端传来的模式描述归一成内部模式标识。

    :param raw: 用户可见的模式文案，中英文混杂且历史上换过好几种写法。
    :return: ``lsb`` / ``hybrid`` / ``dwt`` / ``fft`` / ``dct`` 之一。

    用**子串匹配**而不是精确查表，是为了兼容前端各版本的文案变动（"最快"、
    "仅空间域"、"LSB" 都指同一种模式）。英文在小写后的 ``text`` 里匹配，中文直接在
    原串 ``raw`` 里匹配——中文没有大小写，先 ``lower()`` 反而多此一举。
    判断顺序即优先级，先命中先返回；识别不出时回落到 ``dct``（默认模式）。
    """
    text = (raw or "").lower()
    if "lsb" in text or "空间" in raw or "最快" in raw:
        return "lsb"
    if "全部" in raw or "hybrid" in text or "最强" in raw:
        return "hybrid"
    if "dwt" in text:
        return "dwt"
    if "fft" in text:
        return "fft"
    return "dct"


def _mode_label(mode: str) -> str:
    """把内部模式标识翻回给人看的中文文案。

    :func:`_normalize_mode` 的逆向操作，用于写进记录和检测结果，让前端不必自己维护
    一份映射表。传入未知标识时同样回落到 ``DCT + 空间域``，与归一化的默认值保持一致。
    """
    return {
        "lsb": "仅空间域",
        "dct": "DCT + 空间域",
        "dwt": "DWT + 空间域",
        "fft": "FFT + 空间域",
        "hybrid": "全部算法",
    }.get(mode, "DCT + 空间域")


def build_default_operations(
    *,
    settings: Settings,
    repository: Repository,
    runtime: Runtime,
    state_value: Callable[[str], Any],
    ensure_directories: Callable[[], None],
) -> WatermarkOperations:
    """装配生产环境使用的那一份 :class:`WatermarkOperations`。

    :param settings: 提供各算子需要的运行期路径（``data_dir`` / ``upload_dir``）。
    :param repository: 数据访问层，这里只用到读记录的能力。
    :param runtime: 进程级状态，用于取"最近生成的溯源号"做候选排序。
    :param state_value: 读取可在后台修改的动态开关（如小裁剪层的全局默认值）。
        传函数而不是传当前值，是因为开关随时可能被改，必须每次调用时现取。
    :param ensure_directories: 建目录的钩子，由应用层提供。
    :return: 字段全部填好、可直接注入 ``WatermarkService`` 的冻结实例。

    全部参数都是关键字参数：字段太多，位置传参极易错位且不可读。

    本函数只做绑定，不含任何算法逻辑。绑定手法有三类：

    * **直接引用**——底层签名与 ``WatermarkOperations`` 声明完全一致，
      如 ``lsb.embed_lsb``、``output.jpeg_subsampling``；
    * **lambda 包一层**——底层需要 ``settings.data_dir`` 这类运行期上下文，
      或需要注入依赖钩子，包装后对外只暴露 service 关心的那几个参数；
    * **具名闭包**——逻辑多到一行 lambda 写不下的（``candidates`` /
      ``detect_v4_current`` / ``aligned``）。
    """
    # ---- 先把几个要被反复复用的小闭包定下来 ----
    # 绑定成方法引用而非立即调用：读记录要按需发生，装配阶段不该碰数据库。
    records = repository.read_records
    # 每次调用都拷贝一份快照。运行期列表会被嵌入流程持续改写，
    # 直接把原列表交给检测器，会在长检测过程中被并发修改。
    generated = lambda: list(runtime.generated_trace_ids)
    # 特征索引的存取都要 settings.data_dir，且底层留了依赖注入钩子（便于测试替换
    # 提取/保存实现）。这里把路径和默认实现一次性固定好，对外只留 (图, 记录号)。
    save_feature = lambda image, record_id: feature_matching.save_record_feature_index(
        image,
        record_id,
        settings.data_dir,
        extract_feature_descriptors_fn=extract_feature_descriptors,
        save_feature_descriptors_fn=save_feature_descriptors,
    )
    # v4 的特征索引是另一套格式（存在 feature_index_v4/ 目录），与传统链路互不通用，
    # 所以必须单独绑一个算子，不能靠参数开关合并。
    save_feature_v4 = (
        lambda image, record_id: feature_matching.save_record_feature_index_v4(
            image,
            record_id,
            settings.data_dir,
            extract_v4_feature_index_fn=extract_v4_feature_index,
            save_v4_feature_index_fn=save_v4_feature_index,
        )
    )
    # 视觉一致性校验要回读记录对应的存档图片，因此必须先绑定 upload_dir。
    # 小裁剪层和编码层的检测器都用它做二次确认（短码位数少、易碰撞，命中后还要看画面像不像）。
    visual_consistency = lambda image, record: feature_matching.record_visual_consistency(
        image, record, settings.upload_dir
    )

    def candidates(current: list[dict[str, Any]]) -> Any:
        """筛出可参与 v4 检测的候选记录。

        :param current: 本次提取已经查好的记录集，由调用方传入以免重复查库。
        :return: ``V4Candidate`` 元组。

        底层是全关键字签名，且需要 ``data_dir``、版本号常量、配置工厂和特征索引
        读取函数四类外部依赖。这些在装配期就能确定，所以在这里一次性固定，
        对外只保留"给我记录集、还你候选集"这一个参数。
        """
        return detection.v4_candidate_records(
            records=current,
            data_dir=settings.data_dir,
            version_v4=ROBUST_WATERMARK_VERSION_V4,
            config_factory=V4Config,
            record_feature_index_path=feature_matching.record_feature_index_path,
            load_feature_index=load_v4_feature_index,
        )

    def detect_v4_current(
        image: Image.Image, current_candidates: Any, current: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """跑 v4 检测（FFT 同步导频 + DCT 认证码字）。

        :param current_candidates: 流水线预先算好的候选集，避免这里重复筛选。
        :param current: 本次提取的记录集快照。
        :return: 命中返回证据字典，否则 ``None``。

        候选集既作为参数传入、又以 ``candidate_records`` 回调形式再传一遍：
        前者是常规路径直接用现成结果，后者是底层在候选为空或需要重算时的兜底入口。
        底层判空后才会调那个回调，所以正常情况下不会真的重算一遍。
        """
        return detection.detect_v4_watermark(
            image,
            current_candidates,
            records=current,
            generated_trace_ids=generated(),
            version_v4=ROBUST_WATERMARK_VERSION_V4,
            config_factory=V4Config,
            candidate_records=lambda: candidates(current),
            detect=detect_v4,
            with_evidence_fields=_with_evidence_fields,
            now_text=_now_text,
        )

    def aligned(image: Image.Image, current: list[dict[str, Any]], **kwargs: Any):
        """几何对齐 + 认证鲁棒水印检测：截图、翻拍、缩放场景的主力检测器。

        :param kwargs: 流水线透传的调节项，目前认 ``candidate_limit``（最多试几条
            候选，默认 8）和 ``budget_seconds``（总耗时上限，默认 5 秒）。
            用 ``**kwargs`` 而非显式形参，是为了让流水线加新调节项时不用改这里。
        :return: 命中返回证据字典，否则 ``None``。

        流程是"先排序候选、再逐条对齐、最后按版本解码"，因此要注入三组回调：

        * ``rank_candidates``——按图像特征给候选打分排序。它还带
          ``save_record_feature_index_fn``：老记录可能还没建过特征索引，
          排序时顺手补建，下次就不用重算了；
        * ``align_query``——用特征点求单应矩阵，把查询图"摆正"回存档图的几何；
        * ``decode_v1/v2/v3``——三个版本各有解码器，全部注入，由底层按记录里的
          版本号挑用；这也是为什么 ``normalize_version`` 要在这里再绑一次。

        默认值 8 和 5.0 在这里以 ``kwargs.get`` 兜底而非写在签名上，
        是因为流水线不一定传这两个键。
        """
        rank = lambda subject, values: feature_matching.rank_aligned_candidates(
            subject,
            values,
            upload_dir=settings.upload_dir,
            data_dir=settings.data_dir,
            generated_trace_ids=generated(),
            save_record_feature_index_fn=save_feature,
            extract_feature_descriptors_fn=extract_feature_descriptors,
            load_feature_descriptors_fn=load_feature_descriptors,
        )
        return robust.detect_aligned_authenticated_watermark(
            image,
            kwargs.get("candidate_limit", 8),
            kwargs.get("budget_seconds", 5.0),
            records=current,
            rank_candidates=rank,
            align_query=lambda subject, record: feature_matching.align_query_to_record(
                subject, record, settings.upload_dir
            ),
            decode_v1=robust.decode_aligned_robust_trace,
            decode_v2=robust.decode_aligned_robust_trace_v2,
            decode_v3=robust.decode_aligned_robust_trace_v3,
            normalize_version=lambda value: robust.normalize_robust_watermark_version(
                value,
                version_v1=ROBUST_WATERMARK_VERSION_V1,
                version_v2=ROBUST_WATERMARK_VERSION_V2,
                version_v3=ROBUST_WATERMARK_VERSION_V3,
                version_v4=ROBUST_WATERMARK_VERSION_V4,
            ),
            with_evidence_fields=_with_evidence_fields,
            now_text=_now_text,
            version_v1=ROBUST_WATERMARK_VERSION_V1,
            version_v2=ROBUST_WATERMARK_VERSION_V2,
            version_v3=ROBUST_WATERMARK_VERSION_V3,
            codec_v2=ROBUST_WATERMARK_CODEC_V2,
            codec_v3=ROBUST_WATERMARK_CODEC_V3,
            watermark_layers=WATERMARK_LAYERS,
        )

    # 字段顺序与 WatermarkOperations 的声明顺序一致，方便两边对照着改。
    return WatermarkOperations(
        # ---- 通用工具：目录、解析、格式化 ----
        ensure_dirs=ensure_directories,
        evidence_uuid_fields=_evidence_uuid_fields,
        # 包一层只为固定签名：io 侧将来加可选参数时，这里不受影响。
        load_upload_image=lambda file: io.load_upload_image(file),
        normalize_mode=_normalize_mode,
        apply_visible_copyright=visible_mark.apply_visible_copyright,
        parse_bool=_parse_bool,
        clamp_float=_clamp_float,
        now_text=_now_text,
        mode_label=_mode_label,
        # ---- 强度与版本换算 ----
        # 保真度 → 嵌入强度的线性反比：保真度 0 得强度 1.0，保真度 1 得 0.28。
        # 下限刻意不取 0——强度归零等于没嵌水印，那样保存下来的记录会是假的。
        # 系数 0.72 是实测折中：再大画质开始肉眼可见受损，再小则抗攻击能力不够。
        fidelity_to_strength=lambda value: 1.0
        - _clamp_float(value, 0.75, 0.0, 1.0) * 0.72,
        # 双层 clamp：内层先把配置文件里的全局默认值夹进 [0, 2]，外层再夹用户传值。
        # 这样配置写错（比如强度填了 10）也不会经由"默认值"这条路径漏进来。
        robust_strength_to_scale=lambda value: _clamp_float(
            value,
            _clamp_float(DEFAULT_ROBUST_WATERMARK_STRENGTH, 1.0, 0.0, 2.0),
            0.0,
            2.0,
        ),
        # 四个版本号常量都在 config 里，算法模块不 import config（否则又成环），
        # 所以每次调用都得由这一层把常量喂进去。下面 aligned 闭包里还会再绑一次。
        normalize_robust_watermark_version=lambda value: robust.normalize_robust_watermark_version(
            value,
            version_v1=ROBUST_WATERMARK_VERSION_V1,
            version_v2=ROBUST_WATERMARK_VERSION_V2,
            version_v3=ROBUST_WATERMARK_VERSION_V3,
            version_v4=ROBUST_WATERMARK_VERSION_V4,
        ),
        # ---- V4 专用：配置与认证码 ----
        # 绑的是类本身而非实例：service 每次嵌入都现 new 一个，避免共享可变配置。
        v4_config=V4Config,
        # v4 的 HMAC 认证标签；auth_code_from_trace 是 v3 的同类物，
        # 两者算法与长度都不同，必须各绑各的，不能合并成一个算子。
        v4_authentication_tag=v4_authentication_tag,
        auth_code_from_trace=auth_code_from_trace,
        state_value=state_value,
        # ---- 小裁剪追踪层参数 ----
        small_crop_strength_to_scale=small_crop.small_crop_strength_to_scale,
        normalize_small_crop_density=small_crop.normalize_small_crop_density,
        # ---- 嵌入算子（v4 链路）----
        # 这三个来自独立的 watermark_v4 包，签名本就是为编排设计的，直接引用即可。
        embed_v4_pilot=embed_v4_pilot,
        encode_v4_codeword=encode_v4_codeword,
        embed_v4_codeword=embed_v4_codeword,
        # ---- 嵌入算子（传统链路）----
        # 三个鲁棒水印版本全部绑上，由 service 按记录版本号挑一个执行；
        # 不能只绑"当前版本"，因为历史图片的重新处理仍要走老算法。
        embed_robust_watermark=robust.embed_robust_watermark,
        embed_robust_watermark_v2=robust.embed_robust_watermark_v2,
        embed_robust_watermark_v3=robust.embed_robust_watermark_v3,
        apply_frequency_layers=frequency.apply_frequency_layers,
        apply_code_layer=small_crop.apply_code_layer,
        apply_small_crop_trace_layer=small_crop.apply_small_crop_trace_layer,
        # 点阵层需要两个来自别处的钩子：夹紧函数（本模块的）和载荷编码函数
        # （robust 的）。dot_matrix 不 import 它们，是为了让该模块保持可单测。
        apply_dot_matrix_trace_layer=lambda image, trace_id, strength: dot_matrix.apply_dot_matrix_trace_layer(
            image,
            trace_id,
            strength,
            clamp_float_fn=_clamp_float,
            watermark_payload_from_trace_fn=robust.watermark_payload_from_trace,
        ),
        embed_lsb=lsb.embed_lsb,
        # ---- 落盘与证据固化 ----
        jpeg_subsampling=output.jpeg_subsampling,
        save_watermarked_output=output.save_watermarked_output,
        save_thumbnail=io.save_thumbnail,
        save_record_feature_index=save_feature,
        save_record_feature_index_v4=save_feature_v4,
        path_md5=fingerprints.path_md5,
        path_sha256=fingerprints.path_sha256,
        image_content_sha256=fingerprints.image_content_sha256,
        layer_scores_for_image=frequency.layer_scores_for_image,
        # ---- 提取侧：入口与各级检测器 ----
        # 记录集以 lambda 形式传入（read_records=lambda: current）而非直接给列表：
        # 底层支持延迟求值，指纹没命中时就不必真的展开记录集。
        matched_file_fingerprint=lambda content, current: fingerprints.matched_file_fingerprint(
            content,
            read_records=lambda: current,
            with_evidence_fields=_with_evidence_fields,
            now_text=_now_text,
            watermark_layers=WATERMARK_LAYERS,
        ),
        load_image_from_bytes=io.load_image_from_bytes,
        # 传 upload_dir 是为了让底层能识别指向本站存档的 URL，直接读本地文件，
        # 不必绕一圈发 HTTP 请求。
        load_image_from_url=lambda url: io.load_image_from_url(url, settings.upload_dir),
        v4_candidate_records=candidates,
        detect_v4_watermark=detect_v4_current,
        # LSB 两个提取器不需要任何上下文，纯像素运算，直接引用。
        extract_full_lsb=lsb.extract_full_lsb,
        extract_block_lsb=lsb.extract_block_lsb,
        is_registered_original_image=lambda image, current: feature_matching.is_registered_original_image(
            image, records=current, upload_dir=settings.upload_dir
        ),
        # 两个"要不要跑昂贵回退"的闸门函数。前者只看图片尺寸，无需上下文；
        # 后者还要看记录集里有没有可比对的对象，故需包一层。
        should_run_frequency_fallbacks=detection.should_run_frequency_fallbacks,
        should_run_visual_match_fallback=lambda image, current: detection.should_run_visual_match_fallback(
            image, records=current
        ),
        # 点阵检测：候选集在调用现场就地算好（顺带把每条记录的期望载荷预计算出来），
        # 避免检测内层的多重循环里对每个候选反复算同一个值。
        detect_dot_matrix_trace=lambda image, current: dot_matrix.detect_dot_matrix_trace(
            image,
            dot_matrix.dot_matrix_candidate_records(
                current,
                watermark_payload_from_trace_fn=robust.watermark_payload_from_trace,
            ),
            hamming_distance_fn=robust.hamming_distance,
            code_crc16_fn=robust.code_crc16,
            now_text_fn=_now_text,
            with_evidence_fields_fn=_with_evidence_fields,
        ),
        detect_aligned_authenticated_watermark=aligned,
        detect_by_visual_match=lambda image, current: feature_matching.detect_by_visual_match(
            image,
            records=current,
            upload_dir=settings.upload_dir,
            with_evidence_fields=_with_evidence_fields,
            now_text=_now_text,
            watermark_layers=WATERMARK_LAYERS,
        ),
        # 小裁剪层与编码层的检测器参数完全一致（同一套短码体系，只是搜索策略不同）：
        # 都要传"最近生成的溯源号"优先试，并用 visual_consistency 做命中后的二次确认——
        # 短码位数少、碰撞概率高，光靠码本身对上不足以下结论。
        detect_small_crop_trace=lambda image, current: small_crop.detect_small_crop_trace(
            image,
            current,
            generated(),
            watermark_payload_from_trace=robust.watermark_payload_from_trace,
            record_visual_consistency=visual_consistency,
            recover_payload_from_code=robust.recover_payload_from_code,
            hamming_distance=robust.hamming_distance,
            code_crc16=robust.code_crc16,
            now_text=_now_text,
            with_evidence_fields=_with_evidence_fields,
        ),
        detect_watermark_code=lambda image, current: small_crop.detect_watermark_code(
            image,
            current,
            generated(),
            watermark_payload_from_trace=robust.watermark_payload_from_trace,
            record_visual_consistency=visual_consistency,
            recover_payload_from_code=robust.recover_payload_from_code,
            hamming_distance=robust.hamming_distance,
            code_crc16=robust.code_crc16,
            now_text=_now_text,
            with_evidence_fields=_with_evidence_fields,
        ),
        # 盲检测的鲁棒水印：只认 v1 编码，所以候选集要先滤掉 v2/v3/v4 的记录，
        # 否则会拿错误的期望码去比对。
        detect_robust_watermark=lambda image, current: robust.detect_robust_watermark(
            image,
            records=robust.legacy_robust_candidate_records(
                current,
                normalize_version=lambda value: robust.normalize_robust_watermark_version(
                    value,
                    version_v1=ROBUST_WATERMARK_VERSION_V1,
                    version_v2=ROBUST_WATERMARK_VERSION_V2,
                    version_v3=ROBUST_WATERMARK_VERSION_V3,
                    version_v4=ROBUST_WATERMARK_VERSION_V4,
                ),
                version_v1=ROBUST_WATERMARK_VERSION_V1,
            ),
            extract_code=lambda subject, values: robust.extract_robust_code(
                subject, records=values
            ),
            with_evidence_fields=_with_evidence_fields,
            now_text=_now_text,
            layer_scores_for_image=frequency.layer_scores_for_image,
            watermark_layers=WATERMARK_LAYERS,
        ),
        # 残差匹配：底层不接受记录集，包一层把 current 吃掉，好让它跟其他检测器
        # 保持同样的 (image, records) 调用形态，流水线才能一视同仁地依次调用。
        # 该函数目前恒返回 None——视觉/残差相似度只能排候选，不足以证明水印存在，
        # 最终归属必须由有码可依的检测器给出（其实现里有对应说明）。
        detect_by_residual_match=lambda image, current: feature_matching.detect_by_residual_match(
            image
        ),
        detect_visible_copyright=lambda image, current: visible_mark.detect_visible_copyright(
            image,
            records=current,
            with_evidence_fields=_with_evidence_fields,
            now_text=_now_text,
        ),
        with_evidence_fields=_with_evidence_fields,
        # 流水线本身也是注入进来的：它只负责"按序调用、命中即停"，
        # 具体调哪些检测器由 service 在调用时逐个传进去。
        watermark_detection_pipeline=detection.extract_watermark_from_image,
        # ---- 常量：密钥与各层版本号 ----
        # 直接从 config 搬过来。service 拿到的是值而非 config 模块，
        # 因而它完全不知道配置是怎么来的，测试里换个数字就能验边界。
        default_watermark_auth_key=DEFAULT_WATERMARK_AUTH_KEY,
        robust_version_v2=ROBUST_WATERMARK_VERSION_V2,
        robust_version_v3=ROBUST_WATERMARK_VERSION_V3,
        robust_version_v4=ROBUST_WATERMARK_VERSION_V4,
        robust_codec_v2=ROBUST_WATERMARK_CODEC_V2,
        robust_codec_v3=ROBUST_WATERMARK_CODEC_V3,
        small_trace_version=SMALL_TRACE_VERSION,
        dot_matrix_version=DOT_MATRIX_VERSION,
        code_watermark_version=CODE_WATERMARK_VERSION,
        watermark_layers=WATERMARK_LAYERS,
    )
