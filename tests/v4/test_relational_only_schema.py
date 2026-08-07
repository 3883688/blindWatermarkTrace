from sqlalchemy import JSON, MetaData, create_engine
from sqlalchemy.dialects.postgresql import JSONB

from trace_app.database.store import DatabaseStore
from trace_app.v4.schema import V4Tables


def test_v4_schema_contains_no_json_columns() -> None:
    tables = V4Tables.build()

    json_columns = [
        f"{table.name}.{column.name}"
        for table in tables.metadata.tables.values()
        for column in table.columns
        if isinstance(column.type, (JSON, JSONB))
    ]

    assert json_columns == []
    assert "original_filename" in tables.v4_records.c
    assert "result_outcome" in tables.deep_forensics_jobs.c
    assert "result_media_id" in tables.deep_forensics_jobs.c
    assert "result_evidence_id" in tables.deep_forensics_jobs.c


def test_identity_schema_normalizes_role_menus_and_omits_legacy_json_tables() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    store = DatabaseStore(engine)

    store.create_schema(identity_only=True)

    reflected = MetaData()
    reflected.reflect(bind=engine)
    names = set(reflected.tables)
    assert names == {"roles", "role_menus", "users"}
    assert "menus" not in store.roles.c
    assert {"role_key", "menu_key", "position_index"} <= set(store.role_menus.c.keys())
