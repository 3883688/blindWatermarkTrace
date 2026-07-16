import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Settings:
    base_dir: Path
    upload_dir: Path
    data_dir: Path
    db_url: str
    admin_user: str
    admin_pass: str
    app_name: str = "WatermarkSystem"

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
    ) -> "Settings":
        base_path = Path(base_dir)
        upload_path = Path(upload_dir)
        data_path = Path(data_dir)
        if not upload_path.is_absolute():
            upload_path = base_path / upload_path
        if not data_path.is_absolute():
            data_path = base_path / data_path
        return cls(
            base_dir=base_path,
            upload_dir=upload_path,
            data_dir=data_path,
            db_url=db_url.strip(),
            admin_user=admin_user.strip(),
            admin_pass=admin_pass,
            app_name=(
                os.getenv("APP_NAME", "WatermarkSystem")
                if app_name is None
                else app_name
            ),
        )


settings = Settings.from_values(
    base_dir=BASE_DIR,
    upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
    data_dir=os.getenv("DATA_DIR", "./data"),
    db_url=os.getenv("DB_URL", ""),
    admin_user=os.getenv("ADMIN_USER", ""),
    admin_pass=os.getenv("ADMIN_PASS", ""),
    app_name=os.getenv("APP_NAME", "WatermarkSystem"),
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
