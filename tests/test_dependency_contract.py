from pathlib import Path

from packaging.requirements import Requirement


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_opencv_headless_is_pinned_to_verified_baseline() -> None:
    requirements = tuple(
        Requirement(line)
        for line in (REPOSITORY_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    opencv = tuple(
        requirement
        for requirement in requirements
        if requirement.name.lower() == "opencv-python-headless"
    )

    assert len(opencv) == 1
    assert str(opencv[0].specifier) == "==4.13.0.92"
