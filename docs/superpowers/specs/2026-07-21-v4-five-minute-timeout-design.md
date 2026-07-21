# V4 Five-Minute Detection Timeout Design

## Problem

V4 image attribution currently has a 16-second hard deadline. When `demo/5.png`
is checked immediately after generating a V4 image, the recent candidate is
evaluated first and detection reaches that deadline. The detector raises
`TimeoutError`, which FastAPI exposes as a 500 response.

## Change

Set `V4Config.hard_timeout_seconds` to `300.0` seconds and raise its validation
ceiling to the same value. Keep the existing deadline propagation and
`TimeoutError` behavior unchanged. This is a focused configuration change; it
does not alter candidate ranking, watermark decoding, or API response mapping.

## Testing

Update the V4 configuration contract tests to expect a 300-second default,
accept 300 seconds, and reject values above 300 seconds. Run the focused config
tests, then the relevant V4 detector and API tests. Finally restart the Uvicorn
service on `127.0.0.1:8000` and verify the endpoint is reachable.

## Operational Impact

A difficult V4 request can occupy one worker for up to five minutes. This is
the requested trade-off: attribution gets substantially more computation time,
while the existing hard deadline still bounds each detection attempt.
