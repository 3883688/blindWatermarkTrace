from pathlib import Path

import pytest
from packaging.requirements import Requirement


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = max(
    path
    for path in (REPOSITORY_ROOT / "release").glob(
        "trace-v4-centos-????????-??????"
    )
    if path.is_dir()
)
REQUIREMENTS_PATHS = (
    REPOSITORY_ROOT / "requirements.txt",
    RELEASE_ROOT / "requirements.txt",
)


def _ordinary_requirements(path: Path) -> tuple[Requirement, ...]:
    lines = (
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    return tuple(
        Requirement(line.split(" #", 1)[0].rstrip())
        for line in lines
        if line and not line.startswith(("#", "-"))
    )


@pytest.mark.parametrize("requirements_path", REQUIREMENTS_PATHS)
def test_opencv_headless_is_pinned_to_verified_baseline(
    requirements_path: Path,
) -> None:
    requirements = tuple(
        requirement
        for requirement in _ordinary_requirements(requirements_path)
        if requirement.name.lower() == "opencv-python-headless"
    )

    assert len(requirements) == 1
    assert str(requirements[0].specifier) == "==4.13.0.92"
    assert requirements[0].marker is None
