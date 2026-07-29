import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Settings:
    base_dir: Path
    upload_dir: Path
    data_dir: Path
    db_url: str = field(repr=False)
    admin_user: str
    admin_pass: str = field(repr=False)
    app_name: str = "WatermarkSystem"
    environment: str = "development"
    v4_model_manifest_path: Path = Path("models/v4-manifest.json")
    v4_sync_worker_quota: int = 1
    v4_deep_worker_quota: int = 1
    media_public_base_url: str = ""
    v4_sync_p95_seconds: int = 120
    v4_sync_timeout_seconds: int = 300
    v4_deep_timeout_seconds: int = 1000

    @property
    def original_dir(self) -> Path:
        return self.upload_dir / "originals"

    @property
    def watermarked_dir(self) -> Path:
        return self.upload_dir / "watermarked"

    @property
    def thumbnail_dir(self) -> Path:
        return self.upload_dir / "thumbnails"

    @classmethod
    def from_values(
        cls,
        *,
        base_dir: str | Path,
        upload_dir: str | Path,
        data_dir: str | Path,
        db_url: str,
        admin_user: str,
        admin_pass: str,
        app_name: str | None = None,
        environment: str = "development",
        v4_model_manifest_path: str | Path = "models/v4-manifest.json",
        v4_sync_worker_quota: int = 1,
        v4_deep_worker_quota: int = 1,
        media_public_base_url: str = "",
        v4_sync_p95_seconds: int = 120,
        v4_sync_timeout_seconds: int = 300,
        v4_deep_timeout_seconds: int = 1000,
    ) -> "Settings":
        base_path = Path(base_dir).expanduser().resolve()
        upload_path = cls._resolve_path(base_path, upload_dir)
        data_path = cls._resolve_path(base_path, data_dir)
        manifest_path = cls._resolve_path(base_path, v4_model_manifest_path)
        normalized_environment = environment.strip().lower()
        normalized_db_url = db_url.strip()
        if normalized_environment == "production" and not normalized_db_url.lower().startswith(
            "postgresql"
        ):
            raise ValueError("Production V4 requires a PostgreSQL DB_URL")
        cls._validate_v4_limits(
            sync_worker_quota=v4_sync_worker_quota,
            deep_worker_quota=v4_deep_worker_quota,
            sync_p95_seconds=v4_sync_p95_seconds,
            sync_timeout_seconds=v4_sync_timeout_seconds,
            deep_timeout_seconds=v4_deep_timeout_seconds,
        )
        return cls(
            base_dir=base_path,
            upload_dir=upload_path,
            data_dir=data_path,
            db_url=normalized_db_url,
            admin_user=admin_user.strip(),
            admin_pass=admin_pass,
            app_name=(
                os.getenv("APP_NAME", "WatermarkSystem")
                if app_name is None
                else app_name
            ),
            environment=normalized_environment,
            v4_model_manifest_path=manifest_path,
            v4_sync_worker_quota=v4_sync_worker_quota,
            v4_deep_worker_quota=v4_deep_worker_quota,
            media_public_base_url=media_public_base_url.strip().rstrip("/"),
            v4_sync_p95_seconds=v4_sync_p95_seconds,
            v4_sync_timeout_seconds=v4_sync_timeout_seconds,
            v4_deep_timeout_seconds=v4_deep_timeout_seconds,
        )

    @staticmethod
    def _resolve_path(base_dir: Path, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else base_dir / candidate

    @staticmethod
    def _validate_v4_limits(
        *,
        sync_worker_quota: int,
        deep_worker_quota: int,
        sync_p95_seconds: int,
        sync_timeout_seconds: int,
        deep_timeout_seconds: int,
    ) -> None:
        if sync_worker_quota <= 0 or deep_worker_quota <= 0:
            raise ValueError("V4 worker quotas must be positive")
        if min(sync_p95_seconds, sync_timeout_seconds, deep_timeout_seconds) <= 0:
            raise ValueError("V4 deadlines must be positive")
        if sync_p95_seconds > 120:
            raise ValueError("V4 synchronous P95 may not exceed 120 seconds")
        if sync_timeout_seconds > 300:
            raise ValueError("V4 synchronous timeout may not exceed 300 seconds")
        if deep_timeout_seconds > 1000:
            raise ValueError("V4 deep timeout may not exceed 1000 seconds")
        if not sync_p95_seconds <= sync_timeout_seconds <= deep_timeout_seconds:
            raise ValueError("V4 deadlines must satisfy p95 <= hard <= deep")


settings = Settings.from_values(
    base_dir=BASE_DIR,
    upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
    data_dir=os.getenv("DATA_DIR", "./data"),
    db_url=os.getenv("DB_URL", ""),
    admin_user=os.getenv("ADMIN_USER", ""),
    admin_pass=os.getenv("ADMIN_PASS", ""),
    app_name=os.getenv("APP_NAME", "WatermarkSystem"),
    environment=os.getenv("ENVIRONMENT", "development"),
    v4_model_manifest_path=os.getenv(
        "V4_MODEL_MANIFEST_PATH", "models/v4-manifest.json"
    ),
    v4_sync_worker_quota=int(os.getenv("V4_SYNC_WORKER_QUOTA", "1")),
    v4_deep_worker_quota=int(os.getenv("V4_DEEP_WORKER_QUOTA", "1")),
    media_public_base_url=os.getenv("MEDIA_PUBLIC_BASE_URL", ""),
    v4_sync_p95_seconds=int(os.getenv("V4_SYNC_P95_SECONDS", "120")),
    v4_sync_timeout_seconds=int(os.getenv("V4_SYNC_TIMEOUT_SECONDS", "300")),
    v4_deep_timeout_seconds=int(os.getenv("V4_DEEP_TIMEOUT_SECONDS", "1000")),
)

UPLOAD_DIR = settings.upload_dir
DATA_DIR = settings.data_dir
ORIGINAL_DIR = settings.original_dir
WATERMARKED_DIR = settings.watermarked_dir
THUMBNAIL_DIR = settings.thumbnail_dir
DB_URL = settings.db_url
ADMIN_USER = settings.admin_user
ADMIN_PASS = settings.admin_pass
MAGIC = b"MWM1"
BLOCK_SIZE = 32
BLOCK_STRIDE = 32
ROBUST_MAGIC = 0b1010110011010011
ROBUST_BITS = 64
ROBUST_CELL = 16
ROBUST_GRID = 8
ROBUST_TILE = ROBUST_CELL * ROBUST_GRID
ROBUST_CHANNEL = 2
ROBUST_DELTA = 2
DEFAULT_ROBUST_WATERMARK_STRENGTH = os.getenv("ROBUST_WATERMARK_STRENGTH", "1.0")
DEFAULT_ROBUST_WATERMARK_VERSION = os.getenv("ROBUST_WATERMARK_VERSION", "1")
DEFAULT_WATERMARK_AUTH_KEY = os.getenv("WATERMARK_AUTH_KEY", "")
ROBUST_WATERMARK_VERSION_V1 = 1
ROBUST_WATERMARK_VERSION_V2 = 2
ROBUST_WATERMARK_VERSION_V3 = 3
ROBUST_WATERMARK_VERSION_V4 = 4
ROBUST_WATERMARK_CODEC_V2 = "rs_24_8_three_phase"
ROBUST_WATERMARK_CODEC_V3 = "hmac64_full_repeat_phase_permutation_v3"
FEATURE_MATCH_MIN_GOOD = 12
FEATURE_RECENT_RESERVE = 2
FEATURE_RECENT_BACKFILL = 4
DCT_BLOCK = 8
DCT_DELTA = 5.0
DWT_DELTA = 3.0
FFT_DELTA = 0.45
CODE_TILE = 160
CODE_CELL = 20
CODE_GRID = 8
CODE_DELTA = 9.0
CODE_WATERMARK_VERSION = 4
CODE_PHYSICAL_BITS = 64
CODE_PAYLOAD_BITS = 48
CODE_CHANNEL_WEIGHTS = (0.45, 0.75, 0.75)
SMALL_TRACE_TILE = 96
SMALL_TRACE_DELTA = 8.0
SMALL_TRACE_VERSION = 1
SMALL_TRACE_CHANNEL_WEIGHTS = (0.25, 0.85, 0.85)
SMALL_TRACE_SHORT_BITS = 16
DOT_MATRIX_VERSION = 1
DOT_MATRIX_TILE = 96
DOT_MATRIX_GRID = 8
DOT_MATRIX_CELL = DOT_MATRIX_TILE // DOT_MATRIX_GRID
DOT_MATRIX_DELTA = 7.5
DOT_MATRIX_CHANNEL_WEIGHTS = (0.80, 0.80, -0.28)

WATERMARK_LAYERS = {
    "lsb": True,
    "block": True,
    "dct": True,
    "dwt": True,
    "fft": True,
}

MENU_LABELS = {
    "watermark": "生成水印",
    "trace": "图片溯源",
    "manage": "图片管理",
    "role": "角色管理",
}

DEFAULT_ROLES = {
    "admin": {
        "label": "管理员",
        "menus": ["watermark", "trace", "manage", "role"],
    },
    "operator": {
        "label": "操作员",
        "menus": ["watermark", "trace", "manage"],
    },
    "viewer": {
        "label": "查看员",
        "menus": ["trace", "manage"],
    },
}
