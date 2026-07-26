"""水印模式与强度参数的归一化。

前端传来的都是字符串，且历史上有多种写法（英文缩写、中文描述、营销话术），
本模块负责把它们统一成内部标准值，并把"保真度""强度"这类用户视角的参数
换算成算法实际使用的数值。
"""

from __future__ import annotations

from collections.abc import Callable


def mode_label(mode: str) -> str:
    """把内部模式标识翻成中文显示名。

    未知模式回落到 ``dct``，与 :func:`normalize_mode` 的默认值保持一致。
    """
    labels = {
        "lsb": "仅空间域",
        "dct": "DCT + 空间域",
        "dwt": "DWT + 空间域",
        "fft": "FFT + 空间域",
        "hybrid": "全部算法",
    }
    return labels.get(mode, "DCT + 空间域")


def normalize_mode(raw: str) -> str:
    """把任意写法的模式描述归一到五个内部标识之一。

    :return: ``lsb`` / ``hybrid`` / ``dwt`` / ``fft`` / ``dct``（默认）。

    用**包含匹配**而非精确相等，是为了同时容纳英文缩写（``"dct"``）、
    中文描述（``"DCT + 空间域"``）和界面上的营销话术（``"最快"``/``"最强"``），
    避免前端文案一改动后端就失配。

    注意大小写处理不对称：英文关键字在 ``text``（已转小写）中找，
    中文关键字在原始 ``raw`` 中找——中文没有大小写，转换是多余的。

    判定顺序即优先级：``lsb`` 与 ``hybrid`` 这两个语义最明确的先判，
    都不匹配才落到默认的 ``dct``。
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


def fidelity_to_strength(
    value: str,
    *,
    clamp: Callable[[str | float | None, float, float, float], float],
) -> float:
    """把用户视角的"保真度"反向换算成算法的"嵌入强度"。

    :param value: 保真度字符串，夹紧到 ``[0, 1]``，默认 0.75。
    :param clamp: 注入的夹紧函数，便于测试替换。
    :return: 嵌入强度，取值 ``[0.28, 1.0]``。

    二者天然对立：保真度越高（越保画质）→ 强度越低（越不抗攻击）。

    系数 0.72 而非 1.0，意味着即使保真度拉满，仍保留 0.28 的最低强度——
    **完全不嵌入是没有意义的**，那样水印形同虚设。这个下限保证了任何设置下
    都还有基本的可检出性。
    """
    fidelity = clamp(value, 0.75, 0.0, 1.0)
    return 1.0 - fidelity * 0.72


def robust_strength_to_scale(
    value: str | float | None,
    default_value: str | float | None,
    *,
    clamp: Callable[[str | float | None, float, float, float], float],
) -> float:
    """解析鲁棒水印强度倍率，取值 ``[0, 2]``。

    两级夹紧：先把配置里的默认值本身夹到合法区间（防止配置写错），
    再用这个安全的默认值去解析用户传入的值。这样即便两者都非法，
    结果也一定落在有效范围内。
    """
    default = clamp(default_value, 1.0, 0.0, 2.0)
    return clamp(value, default, 0.0, 2.0)
