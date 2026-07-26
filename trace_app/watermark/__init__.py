"""业务水印层：嵌入链路的编排与传统（非 V4）水印算法。

各模块职责：

* :mod:`~trace_app.watermark.service`            —— 嵌入与提取的总编排，系统中枢
* :mod:`~trace_app.watermark.default_operations` —— 把各处实现装配成服务所需的算子集合
* :mod:`~trace_app.watermark.detection`          —— 检测流水线：按优先级调度各检测器
* :mod:`~trace_app.watermark.modes`              —— 模式与强度参数的归一化
* :mod:`~trace_app.watermark.lsb`                —— LSB 明文载荷（最快但最脆弱的一层）
* :mod:`~trace_app.watermark.frequency`          —— DCT/DWT/FFT 三层频域图案
* :mod:`~trace_app.watermark.robust`             —— 鲁棒水印 v1/v2/v3，抗压缩主力层
* :mod:`~trace_app.watermark.small_crop`         —— 小裁剪追踪，短码密铺全图
* :mod:`~trace_app.watermark.dot_matrix`         —— 点阵追踪，抗翻拍与屏摄

本包**不含 V4 算法**——V4 的 DCT 认证码字与 FFT 同步导频在独立的
:mod:`watermark_v4` 包中，两者通过 :mod:`~trace_app.watermark.default_operations`
的算子注入衔接。

此处刻意不做任何重导出：各模块之间存在装配顺序依赖，
在包入口 import 会引入循环导入。请直接从子模块导入所需名字。
"""
