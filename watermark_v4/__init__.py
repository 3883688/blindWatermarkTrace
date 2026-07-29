"""V4 水印算法包：DCT 认证码字 + FFT 同步导频。

各模块职责：

* :mod:`~watermark_v4.config`   —— 全部算法参数与构造期校验
* :mod:`~watermark_v4.payload`  —— HMAC 认证标签、RS 纠错、比特置换
* :mod:`~watermark_v4.dct`      —— DCT 系数调制（信息的实际载体）
* :mod:`~watermark_v4.sync`     —— FFT 同步导频（几何变换的检测与校正）
* :mod:`~watermark_v4.features` —— ORB 特征索引（几何配准与候选粗排）
* :mod:`~watermark_v4.detector` —— 检测总入口，串起上述各环节

此处只重导出**嵌入/提取两条主链路**所需的名字，供
``trace_app.watermark`` 装配算子集合时使用；``detector``、``features``
等更细的接口需按需从子模块直接导入。
"""

from .config import V4Config
from .dct import TileScores, embed_codeword, extract_image_tiles
from .payload import (
    AuthContext,
    CandidateDecode,
    authentication_tag,
    bytes_to_bits,
    canonical_auth_message,
    decode_candidate_codeword,
    encode_codeword,
    inverse_permutation,
    phase_for_tile,
    phase_permutation,
    permute_codeword_bits,
    verify_authentication_tag,
)
from .sync import SyncEstimate, detect_pilot, embed_pilot

__all__ = (
    "AuthContext",
    "CandidateDecode",
    "TileScores",
    "SyncEstimate",
    "V4Config",
    "authentication_tag",
    "bytes_to_bits",
    "canonical_auth_message",
    "decode_candidate_codeword",
    "detect_pilot",
    "encode_codeword",
    "embed_codeword",
    "embed_pilot",
    "extract_image_tiles",
    "inverse_permutation",
    "phase_for_tile",
    "phase_permutation",
    "permute_codeword_bits",
    "verify_authentication_tag",
)
