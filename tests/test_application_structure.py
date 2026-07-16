import ast
import asyncio
import importlib
import os
import subprocess
import sys
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from collections import Counter
from pathlib import Path

import main
from database_store import DatabaseStore
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from trace_app.auth.service import AuthService
from trace_app.application import create_app
from trace_app.config import Settings
from trace_app.database.repositories import Repository
from trace_app.dependencies import get_auth_service
from trace_app.imaging.fingerprints import file_sha256
from trace_app.imaging.io import load_image_from_bytes
from trace_app.runtime import Runtime
from trace_app.watermark.service import WatermarkService
from trace_app.watermark import small_crop as small_crop_module
from trace_app.watermark import robust as robust_module
from trace_app.watermark.lsb import bits_from_bytes, bytes_from_bits
from trace_app.watermark.small_crop import small_trace_short_code

from PIL import Image


def test_watermark_service_preserves_injected_dependencies(tmp_path: Path) -> None:
    test_settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="",
        admin_user="",
        admin_pass="",
    )
    test_repository = object()
    test_runtime = Runtime()

    service = WatermarkService(
        settings=test_settings,
        repository=test_repository,
        runtime=test_runtime,
    )

    assert service.settings is test_settings
    assert service.repository is test_repository
    assert service.runtime is test_runtime


def test_watermark_service_factory_synchronizes_current_generated_trace_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = ["TR-CURRENT"]
    monkeypatch.setattr(main.app.state, "generated_trace_ids", generated)

    service = main.get_watermark_service()

    assert service.repository is main.repository
    assert service.runtime is main.runtime
    assert service.settings is main.settings
    assert service.runtime.generated_trace_ids is generated


class _WatermarkRepositorySpy:
    def __init__(self) -> None:
        self.read_calls = 0
        self.detection_results: list[bool] = []
        self.records = [{"trace_id": "TR-REPOSITORY"}]

    def read_records(self) -> list[dict[str, str]]:
        self.read_calls += 1
        return self.records

    def record_detection_result(self, success: bool) -> None:
        self.detection_results.append(success)


@pytest.mark.parametrize("v4_hit", [False, True])
def test_watermark_service_v4_extract_uses_one_repository_records_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v4_hit: bool,
) -> None:
    repository = _WatermarkRepositorySpy()
    seen: list[tuple[str, object]] = []
    expected = {"trace_id": "TR-V4-HIT", "mode": "v4_authenticated_dct"}

    def candidate_records(records):
        seen.append(("candidates", records))
        return (object(),)

    def detect_v4(image, candidates, records):
        seen.append(("detect", records))
        return expected if v4_hit else None

    def pipeline(image, **kwargs):
        seen.append(("pipeline", kwargs["records"]))
        return main.watermark_detection.extract_watermark_from_image(image, **kwargs)

    operations = replace(
        main.get_watermark_service().operations,
        v4_candidate_records=candidate_records,
        detect_v4_watermark=detect_v4,
        is_registered_original_image=lambda image, records: False,
        watermark_detection_pipeline=pipeline,
    )
    assert not hasattr(operations, "read_records")
    assert not hasattr(operations, "record_detection_result")
    monkeypatch.setattr(
        main,
        "repository",
        SimpleNamespace(
            read_records=lambda: (_ for _ in ()).throw(
                AssertionError("global repository read")
            ),
            record_detection_result=lambda success: (_ for _ in ()).throw(
                AssertionError("global repository write")
            ),
        ),
    )
    service = WatermarkService(
        settings=Settings.from_values(
            base_dir=tmp_path,
            upload_dir="uploads",
            data_dir="data",
            db_url="",
            admin_user="",
            admin_pass="",
        ),
        repository=repository,
        runtime=Runtime(),
        operations=operations,
    )

    if v4_hit:
        assert service.extract_image(Image.new("RGB", (1, 1))) is expected
    else:
        with pytest.raises(HTTPException) as exc_info:
            service.extract_image(Image.new("RGB", (1, 1)))
        assert (exc_info.value.status_code, exc_info.value.detail) == (
            404,
            "未检测到可识别的隐式水印",
        )

    assert repository.read_calls == 1
    assert repository.detection_results == [v4_hit]
    assert seen == [
        ("candidates", repository.records),
        ("pipeline", repository.records),
        ("detect", repository.records),
    ]


def test_watermark_service_non_v4_fallbacks_use_repository_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _WatermarkRepositorySpy()
    repository.records = []
    seen: list[tuple[str, object]] = []

    def record_aware(name, result):
        def callback(image, records, **kwargs):
            seen.append((name, records))
            return result

        return callback

    operations = replace(
        main.get_watermark_service().operations,
        v4_candidate_records=lambda records: (),
        extract_full_lsb=lambda image: None,
        is_registered_original_image=record_aware("original", False),
        should_run_frequency_fallbacks=lambda image: False,
        detect_by_residual_match=record_aware("residual", None),
        extract_block_lsb=lambda image: None,
        state_value=lambda name: False,
    )
    monkeypatch.setattr(
        main,
        "repository",
        SimpleNamespace(
            read_records=lambda: (_ for _ in ()).throw(
                AssertionError("global repository read")
            )
        ),
    )
    service = WatermarkService(
        settings=Settings.from_values(
            base_dir=tmp_path,
            upload_dir="uploads",
            data_dir="data",
            db_url="",
            admin_user="",
            admin_pass="",
        ),
        repository=repository,
        runtime=Runtime(),
        operations=operations,
    )

    with pytest.raises(HTTPException) as exc_info:
        service.extract_image(Image.new("RGB", (1, 1)))

    assert exc_info.value.status_code == 404
    assert seen == [
        ("original", repository.records),
        ("residual", repository.records),
    ]


def test_watermark_service_fingerprint_miss_uses_repository_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _WatermarkRepositorySpy()
    seen: list[object] = []

    def pipeline(image, **kwargs):
        seen.append(kwargs["records"])
        return {"mode": "miss"}

    operations = replace(
        main.get_watermark_service().operations,
        matched_file_fingerprint=lambda content, records: seen.append(records),
        load_image_from_bytes=lambda content: Image.new("RGB", (1, 1)),
        v4_candidate_records=lambda records: (),
        watermark_detection_pipeline=pipeline,
    )
    monkeypatch.setattr(
        main,
        "repository",
        SimpleNamespace(
            read_records=lambda: (_ for _ in ()).throw(
                AssertionError("global repository read")
            )
        ),
    )
    service = WatermarkService(
        settings=Settings.from_values(
            base_dir=tmp_path,
            upload_dir="uploads",
            data_dir="data",
            db_url="",
            admin_user="",
            admin_pass="",
        ),
        repository=repository,
        runtime=Runtime(),
        operations=operations,
    )

    result = asyncio.run(
        service.extract_upload(
            UploadFile(filename="miss.png", file=BytesIO(b"miss"))
        )
    )

    assert result == {"mode": "miss"}
    assert repository.read_calls == 1
    assert seen == [repository.records, repository.records]
    assert seen[0] is seen[1]


def test_record_adapters_support_positional_only_records() -> None:
    records = [{"trace_id": "TR-POSITIONAL"}]

    def detector(image, records, /):
        return records

    def candidates(records, /):
        return records

    def v4_detector(image, candidate_values, records, /):
        return records

    assert main._record_aware_callback(detector)(object(), records) is records
    assert main._record_candidate_callback(candidates)(records) is records
    assert main._v4_record_aware_callback(v4_detector)(
        object(), (), records
    ) is records


def test_record_adapters_fall_back_when_signature_is_unavailable() -> None:
    class Uninspectable:
        @property
        def __signature__(self):
            raise ValueError("signature unavailable")

        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return args

    records = [{"trace_id": "TR-FALLBACK"}]
    detector = Uninspectable()
    candidates = Uninspectable()
    v4_detector = Uninspectable()

    assert main._record_aware_callback(detector)("image", records) == ("image",)
    assert main._record_candidate_callback(candidates)(records) == ()
    assert main._v4_record_aware_callback(v4_detector)(
        "image", "candidates", records
    ) == ("image", "candidates")


def test_watermark_service_extract_upload_fingerprint_uses_repository_stats(
    tmp_path: Path,
) -> None:
    repository = _WatermarkRepositorySpy()
    operations = replace(
        main.get_watermark_service().operations,
        matched_file_fingerprint=lambda content, records: {
            "trace_id": "TR-HIT",
            "matched_file_type": "watermarked",
        },
    )
    service = WatermarkService(
        settings=Settings.from_values(
            base_dir=tmp_path,
            upload_dir="uploads",
            data_dir="data",
            db_url="",
            admin_user="",
            admin_pass="",
        ),
        repository=repository,
        runtime=Runtime(),
        operations=operations,
    )

    result = asyncio.run(
        service.extract_upload(
            UploadFile(filename="matched.png", file=BytesIO(b"matched"))
        )
    )

    assert result["trace_id"] == "TR-HIT"
    assert repository.read_calls == 1
    assert repository.detection_results == [True]


EXPECTED_ROUTES = {
    ("GET", "/"),
    ("GET", "/site-logo.png"),
    ("GET", "/favicon.ico"),
    ("GET", "/favico.ico"),
    ("POST", "/auth/login"),
    ("GET", "/api/roles"),
    ("PUT", "/api/roles/{role_key}"),
    ("GET", "/api/users"),
    ("POST", "/api/users"),
    ("PUT", "/api/users/{username}"),
    ("DELETE", "/api/users/{username}"),
    ("POST", "/api/watermark/embed"),
    ("POST", "/api/watermark/extract"),
    ("POST", "/api/watermark/extract-url"),
    ("GET", "/api/dashboard-stats"),
    ("GET", "/api/images"),
    ("DELETE", "/api/images/{image_id}"),
    ("POST", "/api/dev/reset"),
}


def _factory_settings(tmp_path: Path) -> Settings:
    return Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="",
        admin_user="",
        admin_pass="",
    )


def _application_routes(factory_app):
    for route in factory_app.routes:
        included_router = getattr(route, "original_router", None)
        if included_router is not None:
            yield from included_router.routes
        else:
            yield route


def test_application_factory_registers_each_route_once_without_database(
    tmp_path: Path,
) -> None:
    factory_app = create_app(
        settings=_factory_settings(tmp_path), initialize_database=False
    )
    actual_routes = Counter(
        (method, route.path)
        for route in _application_routes(factory_app)
        for method in getattr(route, "methods", set())
    )

    for expected_route in EXPECTED_ROUTES:
        assert actual_routes[expected_route] == 1
    assert factory_app.state.runtime.store is None
    assert factory_app.state.repository is not None
    assert (
        factory_app.state.generated_trace_ids
        is factory_app.state.runtime.generated_trace_ids
    )


def test_application_factory_does_not_share_route_objects_between_apps(
    tmp_path: Path,
) -> None:
    first = create_app(
        settings=_factory_settings(tmp_path / "first"), initialize_database=False
    )
    second = create_app(
        settings=_factory_settings(tmp_path / "second"), initialize_database=False
    )

    assert {id(route) for route in first.routes}.isdisjoint(
        {id(route) for route in second.routes}
    )


def test_application_dependencies_call_current_factories_per_request(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class AuthSpy:
        def login(self, username: str, password: str) -> dict[str, str]:
            calls.append(f"auth:{username}:{password}")
            return {"kind": "auth"}

    class WatermarkSpy:
        def extract_url(self, url: str) -> dict[str, str]:
            calls.append(f"watermark:{url}")
            return {"kind": "watermark"}

    class ManagementSpy:
        def dashboard_stats(self) -> dict[str, int]:
            calls.append("management")
            return {"today": 0}

    factory_app = create_app(
        settings=_factory_settings(tmp_path),
        initialize_database=False,
        auth_service_factory=lambda: AuthSpy(),
        watermark_service_factory=lambda: WatermarkSpy(),
        management_service_factory=lambda: ManagementSpy(),
    )
    client = TestClient(factory_app)

    assert client.post(
        "/auth/login", data={"username": "alice", "password": "secret"}
    ).json() == {"kind": "auth"}
    assert client.post(
        "/api/watermark/extract-url", data={"url": "https://example.test/a.png"}
    ).json() == {"kind": "watermark"}
    assert client.get("/api/dashboard-stats").json() == {"today": 0}
    assert calls == [
        "auth:alice:secret",
        "watermark:https://example.test/a.png",
        "management",
    ]


def test_application_async_dependency_override_replaces_state_service(
    tmp_path: Path,
) -> None:
    class AuthOverride:
        def login(self, username: str, password: str) -> dict[str, str]:
            return {"username": username, "source": "override"}

    factory_app = create_app(
        settings=_factory_settings(tmp_path), initialize_database=False
    )
    async def override_auth_service() -> AuthOverride:
        return AuthOverride()

    factory_app.dependency_overrides[get_auth_service] = override_auth_service

    response = TestClient(factory_app).post(
        "/auth/login", data={"username": "alice", "password": "secret"}
    )

    assert response.json() == {"username": "alice", "source": "override"}


def test_application_yield_dependency_override_runs_cleanup(tmp_path: Path) -> None:
    cleaned_up: list[bool] = []

    class AuthOverride:
        def login(self, username: str, password: str) -> dict[str, str]:
            return {"source": "yield-override"}

    async def override_auth_service():
        try:
            yield AuthOverride()
        finally:
            cleaned_up.append(True)

    factory_app = create_app(
        settings=_factory_settings(tmp_path), initialize_database=False
    )
    factory_app.dependency_overrides[get_auth_service] = override_auth_service

    with TestClient(factory_app) as client:
        response = client.post(
            "/auth/login", data={"username": "alice", "password": "secret"}
        )

    assert response.json() == {"source": "yield-override"}
    assert cleaned_up == [True]


def test_application_lifespan_disposes_sqlite_engine(tmp_path: Path) -> None:
    database_path = tmp_path / "runtime.sqlite3"
    test_settings = Settings.from_values(
        base_dir=Path(__file__).resolve().parents[1],
        upload_dir=tmp_path / "uploads",
        data_dir=tmp_path / "data",
        db_url=f"sqlite+pysqlite:///{database_path}",
        admin_user="admin",
        admin_pass="secret",
    )
    factory_app = create_app(settings=test_settings, initialize_database=True)

    with TestClient(factory_app) as client:
        assert client.get("/api/roles").status_code == 200
    client.close()
    database_path.unlink()

    assert not database_path.exists()


def test_failed_runtime_creation_disposes_sqlite_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trace_app.database import connection

    database_path = tmp_path / "failed-runtime.sqlite3"
    test_settings = Settings.from_values(
        base_dir=Path(__file__).resolve().parents[1],
        upload_dir=tmp_path / "uploads",
        data_dir=tmp_path / "data",
        db_url=f"sqlite+pysqlite:///{database_path}",
        admin_user="admin",
        admin_pass="secret",
    )

    def fail_seed(store, settings) -> None:
        raise SQLAlchemyError("seed failed")

    monkeypatch.setattr(connection, "seed_database_defaults", fail_seed)

    with pytest.raises(RuntimeError, match="Database initialization failed") as exc_info:
        create_app(settings=test_settings, initialize_database=True)

    failed_runtime = getattr(exc_info.value, "runtime")
    assert failed_runtime.db_error == "SQLAlchemyError"
    assert failed_runtime.store is None
    assert failed_runtime.engine is not None
    database_path.unlink()
    assert not database_path.exists()


def test_auth_service_filters_unknown_menu_keys() -> None:
    assert AuthService(repository=None).allowed_menu_keys(
        ["watermark", "unknown", "trace"]
    ) == ["watermark", "trace"]


def test_auth_service_defers_missing_repository_failure_until_database_access() -> None:
    service = AuthService(repository=None)

    assert service.repository is None
    with pytest.raises(HTTPException) as exc_info:
        service.list_roles()
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "数据库不可用"


def _auth_service() -> AuthService:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    store.replace_roles(main.DEFAULT_ROLES)
    store.create_user("admin", "admin-password", "admin")
    return AuthService(Repository(store))


def test_auth_service_preserves_login_contract() -> None:
    service = _auth_service()

    result = service.login("admin", "admin-password")

    assert result["token"].startswith("local-")
    assert result["username"] == "admin"
    assert result["role"] == "admin"
    assert result["menus"] == ["watermark", "trace", "manage", "role"]
    with pytest.raises(HTTPException) as exc_info:
        service.login("admin", "wrong-password")
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "用户名或密码错误"


def test_auth_service_preserves_user_projection_and_default_role() -> None:
    service = _auth_service()

    assert service.public_users({"alice": {}}) == {
        "alice": {"role": "operator"}
    }
    assert service.role_for_username("missing") == "operator"


def test_auth_service_preserves_role_update_rules() -> None:
    service = _auth_service()

    result = service.update_role("viewer", {"menus": ["trace", "unknown"]})

    assert result["roles"]["viewer"]["menus"] == ["trace"]
    with pytest.raises(HTTPException) as exc_info:
        service.update_role("missing", {"menus": []})
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "角色不存在"


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"password": "secret"}, "请输入用户名"),
        ({"username": "alice"}, "请输入密码"),
        (
            {"username": "alice", "password": "secret", "role": "missing"},
            "角色不存在",
        ),
    ],
)
def test_auth_service_preserves_create_user_validation(
    payload: dict[str, str], detail: str
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _auth_service().create_user(payload)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == detail


def test_auth_service_preserves_user_crud_rules() -> None:
    service = _auth_service()

    created = service.create_user(
        {"username": " alice ", "password": "secret", "role": "operator"}
    )
    assert created["users"]["alice"] == {"role": "operator"}

    with pytest.raises(HTTPException) as duplicate:
        service.create_user(
            {"username": "alice", "password": "secret", "role": "operator"}
        )
    assert (duplicate.value.status_code, duplicate.value.detail) == (
        409,
        "用户已存在",
    )

    updated = service.update_user("alice", {"role": "viewer"})
    assert updated["users"]["alice"] == {"role": "viewer"}
    assert "alice" not in service.delete_user("alice")["users"]

    for operation in (
        lambda: service.update_user("missing", {"role": "viewer"}),
        lambda: service.delete_user("missing"),
    ):
        with pytest.raises(HTTPException) as missing:
            operation()
        assert (missing.value.status_code, missing.value.detail) == (
            404,
            "用户不存在",
        )


def test_auth_service_factory_uses_current_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_repository = Repository(None)
    monkeypatch.setattr(main, "repository", current_repository)

    assert main.get_auth_service().repository is current_repository


def test_auth_routes_are_thin_service_delegators() -> None:
    from trace_app.api import auth, users

    module = ast.parse(
        Path(auth.__file__).read_text(encoding="utf-8")
        + "\n"
        + Path(users.__file__).read_text(encoding="utf-8")
    )
    route_names = {
        "login",
        "get_roles",
        "update_role",
        "get_users",
        "create_user",
        "update_user",
        "delete_user",
    }
    routes = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in route_names
    }

    assert set(routes) == route_names
    for route in routes.values():
        assert not any(isinstance(node, ast.If) for node in ast.walk(route))
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "service"
            for node in ast.walk(route)
        )


def test_lsb_byte_bits_round_trip() -> None:
    assert bytes_from_bits(bits_from_bytes(b"trace")) == b"trace"


def test_detection_pipeline_calls_detectors_in_order() -> None:
    detection = importlib.import_module("trace_app.watermark.detection")
    calls: list[str] = []

    pipeline = detection.DetectionPipeline(
        (
            lambda image: calls.append("first"),
            lambda image: calls.append("second") or {"trace_id": "matched"},
        )
    )

    assert pipeline(Image.new("RGB", (1, 1))) == {"trace_id": "matched"}
    assert calls == ["first", "second"]


def test_detection_pipeline_stops_after_first_match() -> None:
    detection = importlib.import_module("trace_app.watermark.detection")
    calls: list[str] = []

    pipeline = detection.DetectionPipeline(
        (
            lambda image: calls.append("first") or {"trace_id": "matched"},
            lambda image: calls.append("second") or {"trace_id": "late"},
        )
    )

    assert pipeline(Image.new("RGB", (1, 1))) == {"trace_id": "matched"}
    assert calls == ["first"]


def test_detection_pipeline_exposes_detect_method() -> None:
    detection = importlib.import_module("trace_app.watermark.detection")
    expected = {"trace_id": "matched"}

    pipeline = detection.DetectionPipeline((lambda image: expected,))

    assert pipeline.detect(Image.new("RGB", (1, 1))) == expected


def test_main_robust_code_uses_current_magic(monkeypatch) -> None:
    monkeypatch.setattr(main, "ROBUST_MAGIC", 0xBEEF)

    assert main.robust_code_from_trace("TRACE-DYNAMIC-MAGIC") >> 48 == 0xBEEF


def test_invalid_robust_magic_does_not_read_records(monkeypatch) -> None:
    invalid_code = (main.ROBUST_MAGIC ^ 0xFFFF) << 48
    monkeypatch.setattr(
        main,
        "read_records",
        lambda: (_ for _ in ()).throw(AssertionError("records loaded")),
    )

    assert main.robust_code_to_trace(invalid_code) is None
    assert main.robust_code_to_trace_fuzzy(invalid_code) == (
        None,
        main.ROBUST_BITS + 1,
    )


def test_aligned_owner_starts_budget_before_loading_candidates() -> None:
    events: list[str] = []
    times = iter((0.0, 10.0))
    record = {"trace_id": "TRACE-BUDGET", "robust_watermark": True}

    def perf_counter() -> float:
        events.append("clock")
        return next(times)

    def load_records():
        events.append("records")
        return [record]

    def rank_candidates(image, records):
        events.append("rank")
        return records

    result = robust_module.detect_aligned_authenticated_watermark(
        Image.new("RGB", (1, 1)),
        budget_seconds=5.0,
        records=load_records,
        rank_candidates=rank_candidates,
        align_query=lambda image, candidate: (_ for _ in ()).throw(
            AssertionError("budget must include candidate loading")
        ),
        decode_v1=lambda alignment, candidate: None,
        decode_v2=lambda alignment, candidate: None,
        decode_v3=lambda alignment, candidate: None,
        normalize_version=lambda value: 1,
        with_evidence_fields=lambda result, candidate: result,
        now_text=lambda: "now",
        version_v1=1,
        version_v2=2,
        version_v3=3,
        codec_v2="v2",
        codec_v3="v3",
        watermark_layers={},
        perf_counter=perf_counter,
    )

    assert result is None
    assert events == ["clock", "records", "rank", "clock"]


def test_legacy_robust_candidates_accept_records_loader() -> None:
    assert robust_module.legacy_robust_candidate_records(
        lambda: [],
        normalize_version=lambda value: 1,
        version_v1=1,
    ) == []


def test_v4_miss_does_not_load_records() -> None:
    detection = importlib.import_module("trace_app.watermark.detection")
    candidate = SimpleNamespace(record_id="record-1", trace_id="TRACE-V4")

    result = detection.detect_v4_watermark(
        Image.new("RGB", (1, 1)),
        (candidate,),
        records=lambda: (_ for _ in ()).throw(AssertionError("records loaded")),
        generated_trace_ids=[],
        version_v4=4,
        config_factory=lambda: SimpleNamespace(),
        candidate_records=lambda: (),
        detect=lambda *args, **kwargs: None,
        with_evidence_fields=lambda value, record: value,
        now_text=lambda: "now",
    )

    assert result is None


def test_v4_hit_loads_records_after_detection() -> None:
    detection = importlib.import_module("trace_app.watermark.detection")
    events: list[str] = []
    candidate = SimpleNamespace(record_id="record-1", trace_id="TRACE-V4")
    detected = SimpleNamespace(
        record_id="record-1",
        trace_id="TRACE-V4",
        bit_errors=0,
        geometry_method="identity",
        codec="codec-v4",
        candidate_count=1,
        tile_count=2,
        phase_count=1,
        corrected_symbols=0,
        erasure_count=0,
        mean_abs_score=1.0,
        orb_inliers=0,
        orb_ratio=0.0,
        sync_confidence=1.0,
        elapsed_seconds=0.001,
    )

    def detect(*args, **kwargs):
        events.append("detect")
        return detected

    def load_records():
        events.append("records")
        return [{"id": "record-1", "trace_id": "TRACE-V4", "robust_watermark_version": 4}]

    result = detection.detect_v4_watermark(
        Image.new("RGB", (1, 1)),
        (candidate,),
        records=load_records,
        generated_trace_ids=[],
        version_v4=4,
        config_factory=lambda: SimpleNamespace(),
        candidate_records=lambda: (),
        detect=detect,
        with_evidence_fields=lambda value, record: value,
        now_text=lambda: "now",
    )

    assert result is not None
    assert result["trace_id"] == "TRACE-V4"
    assert events == ["detect", "records"]


def test_main_extract_robust_code_uses_patchable_grid_decoder(monkeypatch) -> None:
    trace_id = "TRACE-DYNAMIC-GRID"
    expected_code = main.robust_code_from_trace(trace_id)
    calls: list[tuple[int, int, int]] = []

    def decode_grid(array, cell, offset_x, offset_y):
        calls.append((cell, offset_x, offset_y))
        return expected_code, 1.0, main.ROBUST_BITS

    monkeypatch.setattr(main, "extract_robust_from_grid", decode_grid)

    result = main.extract_robust_code(
        Image.new("RGB", (main.ROBUST_TILE * 2, main.ROBUST_TILE * 2)),
        records=[{"trace_id": trace_id}],
    )

    assert result == (trace_id, 1.0, main.ROBUST_BITS)
    assert calls


def test_full_lsb_match_does_not_read_aligned_state(monkeypatch) -> None:
    payload = {"trace_id": "TRACE-LSB", "mode": "lsb"}
    monkeypatch.setattr(main, "v4_candidate_records", lambda records: ())
    monkeypatch.setattr(main, "extract_full_lsb", lambda image: payload)
    monkeypatch.setattr(
        main,
        "repository",
        SimpleNamespace(
            read_records=lambda: [],
            record_detection_result=lambda success: None,
        ),
    )
    monkeypatch.delattr(main.app.state, "aligned_candidate_limit", raising=False)

    result = main.extract_watermark_from_image(Image.new("RGB", (1, 1)))

    assert result["trace_id"] == "TRACE-LSB"


def test_small_trace_short_code_is_deterministic() -> None:
    assert small_trace_short_code("TRACE-20260716") == 14136


def test_robust_code_to_trace_uses_current_records_without_cache(
    monkeypatch,
) -> None:
    trace_id = "TRACE-DYNAMIC-RECORDS"
    code = main.robust_code_from_trace(trace_id)
    monkeypatch.setattr(main, "read_records", lambda: [{"trace_id": trace_id}])

    assert not hasattr(main.robust_code_to_trace, "cache_info")
    assert main.robust_code_to_trace(code) == trace_id

    monkeypatch.setattr(main, "read_records", lambda: [])

    assert main.robust_code_to_trace(code) is None


def test_small_crop_pattern_helpers_cache_by_arguments() -> None:
    calls = [
        (small_crop_module.small_trace_marker_pattern, (16,)),
        (small_crop_module.small_trace_pattern, ("TRACE-CACHE", 16)),
        (small_crop_module.small_trace_code_carriers, (16,)),
        (small_crop_module.small_trace_short_carriers, (16,)),
        (small_crop_module.code_cell_carriers, (0, 16)),
        (small_crop_module.code_tile_carriers, (16,)),
        (small_crop_module.code_trace_pattern, ("TRACE-CACHE", 16)),
        (small_crop_module.code_marker_pattern, (16,)),
    ]

    for helper, args in calls:
        helper.cache_clear()
        first = helper(*args)
        second = helper(*args)
        assert second is first
        assert helper.cache_info().hits == 1


def test_imaging_io_loads_image_from_bytes() -> None:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")

    assert load_image_from_bytes(buffer.getvalue()).size == (1, 1)


def test_imaging_fingerprints_hashes_file_bytes() -> None:
    assert file_sha256(b"abc") == (
        "BA7816BF8F01CFEA414140DE5DAE2223"
        "B00361A396177A9CB410FF61F20015AD"
    )


def test_main_alignment_wrapper_uses_patchable_resize_helper(
    tmp_path: Path, monkeypatch
) -> None:
    upload_dir = tmp_path / "uploads"
    target_path = upload_dir / "watermarked" / "target.png"
    target_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(target_path)

    resize_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        main,
        "resize_for_residual",
        lambda image, max_side=1200: resize_calls.append(image.size) or image,
    )

    main.align_query_to_record(
        Image.new("RGB", (16, 16), "white"),
        {"download_url": "/uploads/watermarked/target.png"},
    )

    assert resize_calls == [(16, 16), (16, 16)]


def test_main_candidate_ranking_wrapper_passes_current_feature_constants(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(main.imaging_feature_matching, "rank_aligned_candidates", capture)
    monkeypatch.setattr(main, "FEATURE_MATCH_MIN_GOOD", 101)
    monkeypatch.setattr(main, "FEATURE_RECENT_BACKFILL", 102)
    monkeypatch.setattr(main, "FEATURE_RECENT_RESERVE", 103)

    main.rank_aligned_candidates(Image.new("RGB", (1, 1)), [])

    assert captured["feature_match_min_good"] == 101
    assert captured["feature_recent_backfill"] == 102
    assert captured["feature_recent_reserve"] == 103


def test_main_exposes_expected_routes() -> None:
    actual_routes = Counter(
        (method, route.path)
        for route in _application_routes(main.app)
        for method in getattr(route, "methods", set())
    )

    for expected_route in EXPECTED_ROUTES:
        assert actual_routes[expected_route] == 1


def test_main_exposes_required_python_api() -> None:
    required = {
        "ADMIN_PASS",
        "ADMIN_USER",
        "app",
        "ensure_dirs",
        "embed_robust_watermark",
        "embed_robust_watermark_v2",
        "embed_robust_watermark_v3",
        "detect_aligned_authenticated_watermark",
        "extract_watermark_from_image",
        "align_query_to_record",
        "file_sha256",
    }

    assert required <= set(dir(main))


def test_watermark_routes_are_thin_service_delegators() -> None:
    from trace_app.api import watermark

    module = ast.parse(Path(watermark.__file__).read_text(encoding="utf-8"))
    route_names = {
        "embed_watermark",
        "extract_watermark",
        "extract_watermark_url",
    }
    routes = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in route_names
    }

    assert set(routes) == route_names
    for route in routes.values():
        assert not any(isinstance(node, ast.If) for node in ast.walk(route))
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "service"
            for node in ast.walk(route)
        )


def test_main_defines_no_fastapi_routes() -> None:
    source = Path(main.__file__).read_text(encoding="utf-8")

    assert "@app." not in source
    assert "FastAPI(" not in source
    assert ".mount(" not in source


def test_settings_resolves_relative_directories_from_base_dir(tmp_path: Path) -> None:
    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="",
        admin_user="",
        admin_pass="",
    )

    assert settings.upload_dir == tmp_path / "uploads"
    assert settings.data_dir == tmp_path / "data"
    assert settings.original_dir == tmp_path / "uploads" / "originals"


def test_settings_direct_construction_uses_default_app_name(tmp_path: Path) -> None:
    settings = Settings(
        base_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        data_dir=tmp_path / "data",
        db_url="",
        admin_user="",
        admin_pass="",
    )

    assert settings.app_name == "WatermarkSystem"


def test_settings_from_values_reads_app_name_from_environment(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("APP_NAME", "EnvironmentApp")

    settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url="",
        admin_user="",
        admin_pass="",
    )

    assert settings.app_name == "EnvironmentApp"


def test_config_loads_dotenv_before_constructing_global_settings() -> None:
    script = """
import os
import sys
import types

dotenv = types.ModuleType("dotenv")

def load_dotenv():
    os.environ["APP_NAME"] = "LoadedFromDotenv"

dotenv.load_dotenv = load_dotenv
sys.modules["dotenv"] = dotenv

from trace_app.config import settings

assert settings.app_name == "LoadedFromDotenv", settings.app_name
"""
    env = os.environ.copy()
    env.pop("APP_NAME", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_starts_without_database_state() -> None:
    runtime = Runtime()

    assert runtime.engine is None
    assert runtime.store is None
    assert runtime.generated_trace_ids == []


def test_repository_without_store_reports_database_unavailable() -> None:
    repository = Repository(None)

    with pytest.raises(HTTPException) as exc_info:
        repository.read_records()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "数据库不可用"


def test_repository_replaces_and_reads_records() -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    repository = Repository(store)

    repository.replace_records([{"id": "one"}])

    assert repository.read_records() == [{"id": "one"}]


def test_disabled_database_initialization_preserves_existing_state(
    monkeypatch,
) -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    existing_runtime = Runtime(store=store)
    existing_repository = Repository(store)
    monkeypatch.setattr(main, "DB_ENABLED", False)
    monkeypatch.setattr(main, "runtime", existing_runtime)
    monkeypatch.setattr(main, "repository", existing_repository)
    monkeypatch.setattr(main, "db_store", store)

    main.initialize_database()

    assert main.runtime is existing_runtime
    assert main.repository is existing_repository
    assert main.db_store is store


class _DisposeSpy:
    def __init__(self) -> None:
        self.dispose_calls = 0

    def dispose(self) -> None:
        self.dispose_calls += 1


def test_database_reinitialization_disposes_replaced_engine(monkeypatch) -> None:
    previous_engine = _DisposeSpy()
    current_engine = _DisposeSpy()
    previous_runtime = Runtime(engine=previous_engine)
    current_runtime = Runtime(engine=current_engine)
    monkeypatch.setattr(main, "DB_ENABLED", True)
    monkeypatch.setattr(main, "runtime", previous_runtime)
    monkeypatch.setattr(main, "create_runtime", lambda settings: current_runtime)
    monkeypatch.setattr(main, "_sync_application_state", lambda: None)

    main.initialize_database()

    assert main.runtime is current_runtime
    assert previous_engine.dispose_calls == 1
    assert current_engine.dispose_calls == 0


def test_failed_database_reinitialization_disposes_replaced_engine(
    monkeypatch,
) -> None:
    previous_engine = _DisposeSpy()
    failed_engine = _DisposeSpy()
    previous_runtime = Runtime(engine=previous_engine)
    failed_runtime = Runtime(engine=failed_engine, db_error="OperationalError")
    failure = RuntimeError("Database initialization failed")
    setattr(failure, "runtime", failed_runtime)

    def fail_create_runtime(settings):
        failed_engine.dispose()
        raise failure

    monkeypatch.setattr(main, "DB_ENABLED", True)
    monkeypatch.setattr(main, "runtime", previous_runtime)
    monkeypatch.setattr(main, "create_runtime", fail_create_runtime)
    monkeypatch.setattr(main, "_sync_application_state", lambda: None)

    with pytest.raises(RuntimeError, match="Database initialization failed"):
        main.initialize_database()

    assert main.runtime is failed_runtime
    assert previous_engine.dispose_calls == 1
    assert failed_engine.dispose_calls >= 1


def test_failed_database_initialization_synchronizes_compatibility_state(
    tmp_path: Path, monkeypatch
) -> None:
    original_runtime = main.runtime
    original_repository = main.repository
    original_db_engine = main.db_engine
    original_db_store = main.db_store
    original_db_error = main.db_error
    monkeypatch.setattr(main, "runtime", original_runtime)
    monkeypatch.setattr(main, "repository", original_repository)
    monkeypatch.setattr(main, "db_engine", original_db_engine)
    monkeypatch.setattr(main, "db_store", original_db_store)
    monkeypatch.setattr(main, "db_error", original_db_error)
    unavailable_path = tmp_path / "missing" / "runtime.sqlite3"
    failing_settings = Settings.from_values(
        base_dir=tmp_path,
        upload_dir="uploads",
        data_dir="data",
        db_url=f"sqlite+pysqlite:///{unavailable_path}",
        admin_user="admin",
        admin_pass="secret",
    )
    monkeypatch.setattr(main, "DB_ENABLED", True)
    monkeypatch.setattr(main, "settings", failing_settings)

    with pytest.raises(RuntimeError, match="Database initialization failed"):
        main.initialize_database()

    assert main.runtime.db_error == "OperationalError"
    assert main.runtime.store is None
    assert main.db_store is None
    assert main.db_error == "OperationalError"

    monkeypatch.undo()

    assert main.runtime is original_runtime
    assert main.repository is original_repository
    assert main.db_engine is original_db_engine
    assert main.db_store is original_db_store
    assert main.db_error == original_db_error


def test_repository_ensures_directories_before_writing_stats() -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    calls: list[str] = []
    repository = Repository(store, ensure_dirs=lambda: calls.append("called"))

    repository.write_detection_stats({"attempts": 1, "successes": 1})
    repository.write_watermark_stats({"daily": {"2026-07-16": 1}})

    assert calls == ["called", "called"]


def test_main_today_watermark_count_uses_patchable_date_predicate(
    monkeypatch,
) -> None:
    store = DatabaseStore(create_engine("sqlite+pysqlite:///:memory:"))
    store.create_schema()
    monkeypatch.setattr(main, "repository", Repository(store))
    monkeypatch.setattr(main, "read_watermark_stats", lambda: {"daily": {}})
    monkeypatch.setattr(
        main,
        "is_today_record",
        lambda record: record.get("id") == "selected",
    )

    count = main.today_watermark_count([{"id": "selected"}, {"id": "other"}])

    assert count == 1
