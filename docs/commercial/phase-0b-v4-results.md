# Phase 0B V4 Implementation Results

Status: `IMPLEMENTATION_COMPLETE`, `LOCAL_PROMOTION_GATES_COMPLETE`, `REAL_ROUTE_PENDING`.

V4 now has an authenticated HMAC32 + RS(8,4) payload, repeated vectorized DCT carrier, FFT pilot synchronization, query-once spatial/scale-balanced ORB index, at-most-three candidate geometry, registered-tile decoding, exact candidate tag confirmation, unique attribution, and isolated generation/detection API paths.

## Fixed commercial bounds

- Online P95 target: `<=10s`.
- Hard timeout: `16s`.
- Candidate limit: `3`.
- Tile acceptance: coverage `>=70%`, at least `2` tiles and `2` phases.
- No v1/v2/v3, visual, residual, short-code, dot-matrix, DWT, or legacy frequency attribution fallback in the v4 path.
- Generation order: FFT pilot, then authenticated DCT carrier. Applying the pilot last was measured to invert a low-margin data bit.

## Focused verification

- Final v4 core suite including the quick matrix: `354 passed` with warnings treated as errors.
- Final v4 API suite: `8 passed` with one existing Starlette TestClient deprecation warning.
- FFT strict gate: `331 passed` with warnings treated as errors; package compilation passed.
- Full project checkpoint before detector/API work: `803 passed, 2 skipped, 184 existing deprecation warnings` in `162.58s`.
- Final full project verification after crop-geometry refinement: `838 passed, 2 skipped, 184 existing deprecation warnings` in `174.06s`.

## Quick positive matrix

One `1280x960` deterministic feature-rich v4 image was tested through intact, scale `0.5`, `0.75`, `1.5`, `2.0`, centered crop `0.3`, `0.5`, `0.8`, rotation `8 degrees`, JPEG quality `50`, and a combined resize/rotation/crop case.

- Correct authenticated trace: `11/11`.
- Wrong trace: `0`.
- Recovered codeword bit errors: `0` in every case.
- 30% crop: `4` authenticated tiles across `4` phases.
- Observed single-candidate runtime range on the current machine: approximately `0.10s..0.25s`. This is implementation evidence, not the final end-to-end P95 claim.

The automatic regression is `tests/test_watermark_v4_quick_matrix.py`.

## Negative development gate

Command configuration used the standard commercial negative benchmark with explicit v4 generation, four workers, OpenCV worker threads fixed to one, and `45` deterministic synthetic variants. Five source images each contributed the original plus ten attack variants, producing exactly `100` negatives.

- Total: `100`.
- False positives: `0`.
- Original-image false attributions: `0`.
- False-positive rate: `0.0`.
- Report: `test_output/commercial_negative_benchmark/commercial_negative_results.json`.

## Random-crop geometry refinement

The first standardized v4 crop run recovered `307/360` positives. All 53 failed artifacts decoded successfully with the benchmark's exact crop transform, proving that DCT, RS, and HMAC payload recovery were intact and that failure was confined to candidate retrieval and geometry registration.

The refinement keeps authentication thresholds unchanged:

- Coarse ranking still uses at most 256 query descriptors, but searches the complete spatially balanced stored candidate index.
- FFT scale and rotation constrain a fallback ORB transform after ordinary ORB registration fails.
- For pure scaling, the pilot is normalized before recovering crop origin modulo the 128-pixel tile grid.
- Translation uses bounded descriptor voting and tile-offset snapping. It only proposes geometry; RS decode plus exact HMAC tag confirmation remains the sole attribution evidence.
- Candidate geometry remains capped at three and v4 still has no visual, residual, DWT, or legacy attribution fallback.

Regression coverage includes a real `img/1.png` half-scale 30% random crop, scaled pilot-offset recovery, and candidate evidence stored beyond the first 256 descriptors.

## Standardized quality, crop, and latency gate

- Algorithm version: `4`.
- Recommended fidelity: `0.95`.
- Positive cases: `360`; correct trace: `342`; wrong trace: `0`; overall recall: `95.00%`.
- Crop 30%: `73/90`, recall `81.11%`.
- Crop 50%: `89/90`, recall `98.89%`.
- Crop 80% and intact: `90/90` each, recall `100%`.
- Matching unwatermarked crop matrix: `360` cases; false positives: `0`.
- Detection latency across all 720 cases: P95 `1296.195ms`; maximum `1737.332ms`.
- Verdict: `PASS`.
- Report: `test_output/commercial_trace_benchmark/commercial_trace_results.json`.

## Independent release negative gate

The release run used 300 deterministic synthetic negatives plus 55 source and source-attack negatives.

- Total: `355`.
- False positives: `0`.
- Original-image false attributions: `0`.
- False-positive rate: `0.0`.
- Algorithm version: `4`.
- Verdict: `PASS`.
- Report: `test_output/commercial_negative_benchmark/commercial_negative_results.json`.

## Credential state

- `WATERMARK_AUTH_KEY` was generated from 48 CSPRNG bytes and stored as one 64-byte Base64 value in `.env`.
- Application and benchmark processes load it through `python-dotenv`; benchmark scripts do not set a hardcoded key.
- Verification checks only presence, entry count, and byte length. The key is never printed in reports or logs.

## Remaining promotion gates

- Collect and pass the documented real social-platform route samples. Local JPEG/screenshot simulations are not real-route evidence.

All local implementation, quality, crop, latency, and agreed 300-negative release gates now pass. Commercial promotion remains pending only because real platform evidence requires external account/device collection and independent review under `docs/commercial/real-sample-intake.md`.

## CentOS V4 deployment package

The V1-era CentOS deployment shape is retained while production configuration is forced to V4.

- Entry point: `sudo ./deploy.sh install-service`.
- Existing MySQL is connection-checked only; the script does not create, reset, or migrate the database.
- Existing `.env`, database content, `data/`, and `uploads/` are preserved and archived before service restart.
- A valid existing `WATERMARK_AUTH_KEY` remains unchanged. Missing, short, or duplicate entries are replaced by one 48-byte CSPRNG Base64 key without printing its value.
- V4 frontend results now show authenticated carrier, tile, phase, correction, synchronization, and elapsed-time evidence instead of misleading empty legacy scores.
- V4 embedding no longer imposes the former 4,000,000-pixel or 4096-pixel-side limits. FFT pilot rows and DCT tiles are processed in bounded batches at the original image resolution; Pillow decompression-bomb protection remains enabled.
- Focused deployment/V4 verification: `35 passed`.
- Final full project verification: `860 passed, 2 skipped, 184 existing deprecation warnings`.
- Package smoke test: clean ZIP extraction started V4 and returned HTTP `200`.
- Package: `release/trace-v4-centos-20260715.zip`.
- SHA-256: `cbdb3462400d17f0fd77a94a8a58a50eb5e628be114faa6dbf4de0c943afd4d4`.
- Tabler Icons Webfont `3.44.0` is bundled under `assets/tabler-icons/`; the frontend has no icon CDN dependency.

The development host is Windows and has no Bash runtime, so actual `systemd`, MySQL client, firewalld, ownership, and Bash execution remain target-CentOS deployment checks. The script performs those checks before reporting deployment success.
