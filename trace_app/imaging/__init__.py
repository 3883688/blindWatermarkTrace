"""成像辅助层：图片读写、指纹、特征匹配与可见水印渲染。

各模块职责：

* :mod:`~trace_app.imaging.io`              —— 上传/URL 图片的加载与格式规整
* :mod:`~trace_app.imaging.output`          —— 成品图落盘，尽量贴近源图的格式与体积
* :mod:`~trace_app.imaging.visible_mark`    —— 可见版权水印（明水印）渲染
* :mod:`~trace_app.imaging.fingerprints`    —— 感知哈希与文件指纹，检测流水线的兜底匹配
* :mod:`~trace_app.imaging.feature_matching`—— ORB 特征配准，供对齐检测把图"摆正"

本包只提供通用成像能力，不含任何水印编解码逻辑——
那些在 :mod:`trace_app.watermark` 与 :mod:`watermark_v4` 中。
"""
