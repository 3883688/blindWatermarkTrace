# MD5 Fingerprint Fast Path Design

## Objective

Make file fingerprint comparison the first step of image tracing. When an uploaded file is byte-for-byte identical to either a registered original image or its generated watermarked image, return a successful trace result immediately without decoding the image or running watermark detection algorithms.

MD5 provides the requested fast lookup key. SHA-256 remains the confirmation check so an MD5 collision cannot produce a false attribution.

## Current Behavior

Watermark generation stores these fingerprints:

- `original_file_sha256`
- `watermarked_file_sha256`
- `original_image_sha256`
- `watermarked_image_sha256`

Upload extraction already checks the file SHA-256 before image decoding. An exact watermarked-file match returns success, but an exact original-file match is deliberately converted to a 404 response. No MD5 values are stored.

## Required Behavior

Watermark generation will additionally store:

- `original_file_md5`
- `watermarked_file_md5`

Upload tracing will run in this order:

1. Read the uploaded bytes once.
2. Calculate the uppercase MD5 and SHA-256 digests once.
3. Compare the MD5 digest with stored original and watermarked file MD5 values.
4. For an MD5 candidate, require its stored SHA-256 value to equal the uploaded SHA-256 value.
5. Return a successful trace result immediately for a confirmed original or watermarked match.
6. If no MD5 candidate matches, compare SHA-256 against legacy records that do not contain MD5 fields.
7. Only when no exact file fingerprint matches, decode the image and run the existing watermark extraction pipeline.

Both original and watermarked file matches are successful attribution results. The response continues to identify which type matched through `matched_file_type` and its registered URL.

## Components

### Fingerprint Helpers

`trace_app.imaging.fingerprints` will add byte and path MD5 helpers alongside the existing SHA-256 helpers:

- `file_md5(content: bytes) -> str`
- `path_md5(path: Path) -> str`

Digests use uppercase hexadecimal strings, matching the existing file SHA-256 convention.

`matched_file_fingerprint` will calculate both digests before iterating records. It will use MD5 to select candidates and SHA-256 to confirm them. The existing pixel fingerprint fallback remains unchanged and only runs when exact file bytes do not match.

### Watermark Generation

`WatermarkOperations` will expose `path_md5`. `WatermarkService.embed` will calculate MD5 for the saved original and watermarked files and persist both values in the generated record.

MD5 is calculated from the saved files, not from reconstructed image pixels. This ensures the stored digest represents the exact bytes later uploaded for tracing.

### Upload Extraction

`WatermarkService.extract_upload` will keep fingerprint matching before `load_image_from_bytes`. Any confirmed original or watermarked match will:

- increment detection attempts and successes;
- return the fingerprint attribution result;
- skip image decoding and all watermark detectors.

The current special case that turns an original-file fingerprint match into a 404 response will be removed.

### Compatibility

Historical records without MD5 values remain traceable through their existing file SHA-256 fields. No database schema migration is needed because records are stored as flexible dictionaries in the existing relational record payload.

The public `main.file_sha256`, `main.path_sha256`, and fingerprint APIs remain available. New MD5 helpers will also be exported through the compatibility layer and included in the release package.

## Result Contract

An exact fingerprint match returns the existing fingerprint result shape with these values:

- `confidence`: `100`
- `mode`: `file_fingerprint`
- `mode_label`: `文件指纹一样`
- `status`: `文件指纹一样`
- `matched_file_type`: `original` or `watermarked`
- `matched_file_url`: the corresponding registered URL
- `matched_hash_type`: `file_md5_sha256` for new records or `file_sha256` for legacy records
- `file_md5`: uploaded file MD5
- `file_hash`: uploaded file SHA-256, retained for compatibility

Evidence UUID fields continue to be copied from the matched record.

## Error Handling

- An MD5 match without a matching SHA-256 value is not accepted and tracing continues.
- Missing or malformed stored MD5 values are treated as legacy data and do not raise errors.
- Non-image bytes may still succeed when their exact registered file fingerprint matches; otherwise existing image validation errors remain unchanged.
- Fingerprint helpers do not log file content or digest source paths.

## Testing

Tests will cover:

- deterministic MD5 vectors for byte and path helpers;
- generated records containing correct original and watermarked MD5 values;
- original file MD5 plus SHA-256 match returning HTTP 200 without image decoding;
- watermarked file MD5 plus SHA-256 match returning HTTP 200 without image decoding;
- MD5 collision candidate with mismatched SHA-256 not being accepted;
- legacy SHA-256-only records still matching;
- fingerprint success incrementing both detection attempts and successes;
- fingerprint miss continuing into the existing extraction pipeline;
- compatibility exports and CentOS release synchronization.

The complete existing pytest suite and focused V4/false-positive gates must remain green.

## Scope Boundaries

This change does not alter watermark algorithms, detector thresholds, API paths, upload form fields, database schema, or frontend behavior. It does not use MD5 as the sole attribution proof.
