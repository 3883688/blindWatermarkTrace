import argparse
import base64
import os
import secrets
import tempfile
from pathlib import Path


MANAGED_KEYS = ("ROBUST_WATERMARK_VERSION", "WATERMARK_AUTH_KEY")


def _assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    name, value = stripped.split("=", 1)
    return name.strip(), value


def prepare_environment(path: Path) -> dict[str, object]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    assignments: dict[str, list[str]] = {}
    retained: list[str] = []
    for line in lines:
        parsed = _assignment(line)
        if parsed is not None:
            assignments.setdefault(parsed[0], []).append(parsed[1])
        if parsed is None or parsed[0] not in MANAGED_KEYS:
            retained.append(line)

    db_enabled = (
        assignments.get("DB_ENABLED", ["true"])[-1].strip().lower()
        not in {"0", "false", "no", "off"}
    )
    db_url = assignments.get("DB_URL", [""])[-1].strip()
    if db_enabled and not db_url:
        raise ValueError("DB_URL is required when DB_ENABLED=true")

    existing_keys = assignments.get("WATERMARK_AUTH_KEY", [])
    valid_key = (
        len(existing_keys) == 1
        and len(existing_keys[0].encode("utf-8")) >= 32
    )
    key = (
        existing_keys[0]
        if valid_key
        else base64.b64encode(secrets.token_bytes(48)).decode("ascii")
    )
    output = retained + [
        "ROBUST_WATERMARK_VERSION=4",
        f"WATERMARK_AUTH_KEY={key}",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write("\n".join(output).rstrip("\n") + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if os.name != "nt":
        os.chmod(path, 0o600)

    return {
        "generated": not valid_key,
        "utf8_bytes": len(key.encode("utf-8")),
        "entries": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    try:
        result = prepare_environment(args.env_file)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"environment preparation failed: {exc}\n")
    print(
        "deployment environment prepared: "
        f"generated={result['generated']} "
        f"utf8_bytes={result['utf8_bytes']} entries={result['entries']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
