# Main Module Refactor Design

## Objective

Refactor the 3,946-line `main.py` into a FastAPI package with explicit boundaries for application setup, configuration, database access, authentication, watermark algorithms, image processing, services, and API routes.

The refactor must preserve existing HTTP behavior, deployment commands, and direct `import main` integrations. `main.py` will contain no business implementation. It will only expose the FastAPI application and compatibility exports.

## Scope

The refactor covers code currently implemented in `main.py`:

- FastAPI creation, static mounts, startup initialization, and route registration
- environment configuration and filesystem paths
- database initialization, repositories, records, statistics, roles, and users
- authentication and authorization-related operations
- watermark embedding and extraction algorithms
- image loading, hashing, thumbnails, visible marks, and feature matching
- HTTP endpoints for authentication, users, roles, watermarking, images, and dashboard data
- compatibility for tests and scripts that call `main.<name>`

Algorithm behavior, database schema, frontend behavior, API paths, request fields, response payloads, watermark formats, and detection thresholds are out of scope for intentional changes.

## Package Structure

```text
main.py
trace_app/
|-- __init__.py
|-- application.py
|-- config.py
|-- dependencies.py
|-- compat.py
|-- database/
|   |-- __init__.py
|   |-- connection.py
|   `-- repositories.py
|-- auth/
|   |-- __init__.py
|   |-- schemas.py
|   `-- service.py
|-- watermark/
|   |-- __init__.py
|   |-- service.py
|   |-- lsb.py
|   |-- frequency.py
|   |-- dot_matrix.py
|   |-- small_crop.py
|   |-- robust.py
|   `-- detection.py
|-- imaging/
|   |-- __init__.py
|   |-- io.py
|   |-- fingerprints.py
|   |-- feature_matching.py
|   `-- visible_mark.py
`-- api/
    |-- __init__.py
    |-- auth.py
    |-- users.py
    |-- watermark.py
    |-- images.py
    `-- dashboard.py
```

The exact allocation of tightly coupled helper functions may be adjusted during implementation, but dependencies must continue to point from HTTP adapters toward services and from services toward domain helpers and repositories.

## Module Responsibilities

### Application and Configuration

`trace_app.config` owns environment parsing, directory resolution, constants, and immutable settings. Importing configuration must not initialize the database or mutate application state.

`trace_app.application.create_app()` creates the FastAPI instance, initializes required directories and persistence resources, mounts static paths, registers routers, and attaches runtime state. Tests must be able to create an application without requiring a production database.

`trace_app.dependencies` exposes FastAPI dependencies and accessors for application-scoped resources. Database handles and generated trace state must not be stored as unrelated module globals.

### Database

`database.connection` owns SQLAlchemy engine construction, schema initialization, default seeding, and lifecycle management.

`database.repositories` adapts the existing `DatabaseStore` operations into focused persistence functions used by services. Route modules must not call `DatabaseStore` directly.

The existing relational schema and stored values remain unchanged.

### Authentication

`auth.service` owns login validation, role lookup, user creation and updates, allowed menu normalization, and public user serialization. `auth.schemas` contains request and response models where introducing them does not alter accepted payloads.

Password hashing continues to use the existing `password_security.py` behavior through the database layer. This refactor does not introduce a new token or session scheme.

### Watermark Algorithms

Algorithm modules contain pure or narrowly stateful operations grouped by watermark family:

- `lsb`: packet encoding, full-image LSB, and block LSB
- `frequency`: DCT, DWT, and FFT legacy layers
- `dot_matrix`: dot-matrix embedding and detection
- `small_crop`: small-crop carriers, embedding, scans, and matching
- `robust`: robust watermark versions 1 through 3 and aligned decoding helpers
- `detection`: detection fallbacks and result selection

Existing `watermark_v4` remains a separate package and is consumed by the watermark service. It is not duplicated into `trace_app`.

`watermark.service` orchestrates watermark creation and extraction, obtains candidates and records through repositories, and delegates pixel-level work to algorithm and imaging modules.

### Image Processing

`imaging.io` owns safe upload and URL loading plus image validation. `fingerprints` owns file and image hashes. `feature_matching` owns OpenCV alignment, residual matching, and feature comparisons. `visible_mark` owns visible copyright drawing, font loading, and visible-mark detection.

Image modules must not import FastAPI route modules or access database globals.

### API Routers

Routes use FastAPI `APIRouter` and remain thin adapters:

- `api.auth`: login
- `api.users`: role and user management
- `api.watermark`: embed, file extraction, and URL extraction
- `api.images`: image listing and deletion
- `api.dashboard`: dashboard statistics and development reset

Static homepage, logo, and favicon routes are registered by the application module because they are application delivery concerns rather than business operations.

Routes retain their current paths, methods, form fields, status codes, and response shapes.

## Runtime Data Flow

For watermark generation:

```text
HTTP request -> watermark router -> watermark service
             -> image loader -> embedding algorithms
             -> repository -> generated files and HTTP response
```

For watermark detection:

```text
HTTP request -> watermark router -> watermark service
             -> image loader -> ordered detection pipeline
             -> repository candidate lookup and statistics update
             -> HTTP response
```

Algorithm modules receive explicit inputs. Database and FastAPI application state are accessed by services through injected dependencies, not from the algorithm layer.

## Compatibility Strategy

`main.py` will be a thin deployment entry point:

```python
from trace_app.application import create_app
from trace_app.compat import *

app = create_app()
```

`trace_app.compat` re-exports the functions, constants, and runtime accessors currently used by tests and benchmark scripts. Compatibility wrappers may delegate to the new modules, but they contain no algorithm or route implementation.

Existing commands such as `uvicorn main:app` continue to work. Tests and scripts using `import main` remain valid. Monkeypatch-based tests will be migrated carefully: where a test patches `main.<dependency>`, the compatibility layer must preserve the effective replacement point or the test must be updated to patch the new owning module without changing the behavior under test.

Compatibility exports are transitional but will not be removed as part of this refactor.

## Error Handling

- Existing `HTTPException` status codes and Chinese error messages remain stable.
- Database unavailability continues to produce a controlled service-unavailable response where currently expected.
- Invalid uploads, image decoding failures, unsupported URLs, and missing records retain current response behavior.
- Algorithm modules report domain results or raise domain-level errors; only routers translate those errors into HTTP responses.
- Application startup failures include actionable context without logging credentials. Database URLs remain masked.

## Migration Sequence

1. Add characterization tests for the application entry point, route inventory, and critical compatibility exports.
2. Extract configuration and application lifecycle code.
3. Extract database connection and repository adapters.
4. Extract independent image utilities.
5. Extract watermark algorithms one family at a time, retaining compatibility exports after each move.
6. Introduce authentication and watermark services.
7. Move endpoints into routers and register them through the application factory.
8. Reduce `main.py` to the application entry point and compatibility import.
9. Run focused unit tests, API tests, the complete automated suite, and selected watermark benchmark smoke tests.

Each extraction step must leave the test suite runnable. Large-scale symbol moves are avoided so regressions can be attributed to a single boundary.

## Testing and Acceptance

The refactor is accepted when all of the following hold:

- `main.py` contains no business functions, algorithm implementations, database operations, or route implementations.
- `uvicorn main:app` remains the deployment entry point.
- all current API paths and methods are present
- existing request and response contracts remain unchanged
- existing database schema and records remain usable
- direct integrations required by current tests and benchmark scripts remain available through `main`
- the complete existing pytest suite passes
- focused embed/extract API smoke tests pass
- selected watermark quality and false-positive smoke cases do not regress from the pre-refactor baseline
- imports do not require production database credentials during pytest collection

New tests cover application factory creation, router registration, dependency overrides, compatibility exports, and representative service-to-repository interactions.

## Effort Estimate

Estimated implementation time is three to four working days:

- characterization tests and application/configuration boundaries: 0.5 day
- database, authentication, and image module extraction: 0.75 day
- watermark algorithm extraction: 1 to 1.5 days
- services, routers, compatibility layer, and thin entry point: 0.75 day
- complete regression verification and fixes: 0.5 day

Long-running commercial attack matrices are not included in this estimate. Running and investigating the entire benchmark matrix may add one to two working days depending on runtime and observed regressions.

## Risks and Controls

- Import cycles: enforce one-way dependencies and keep application creation at the outermost layer.
- Import-time database side effects: move initialization into the application lifecycle and expose test overrides.
- Monkeypatch compatibility: add characterization tests before moving symbols and preserve effective patch points.
- Algorithm regressions from implicit globals: pass configuration and repository data explicitly, then compare detection outputs against baseline fixtures.
- Oversized replacement modules: keep watermark families separate and use the service only for orchestration.
