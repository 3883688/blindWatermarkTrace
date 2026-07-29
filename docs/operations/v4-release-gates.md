# V4 Release Gates

The release report is fail-closed. It must identify the current Git commit, schema, codec, pinned model versions, disjoint dataset hashes, fixed seed, reference CPU/GPU/RAM, raw artifact hashes, stage timings, model health, indexed query evidence, and zero sensitive-log leaks.

Thresholds are: recall 99% for full/resize/JPEG and 95% for crops; final attribution 95% with zero wrong traces; 3,000 independent negatives with zero attribution; 1,000 same-source versions with indexed lookup; PSNR 38 and SSIM 0.95 in every stratum; P95 120 seconds, hard limit 300 seconds, and deep limit 1,000 seconds.

Run `python -m tests.v4.run_release_gates --manifest <evidence.json> --output test_output/v4-release-report.json` with `V4_RELEASE_REPORT_KEY` set. Release building rejects missing, invalid, unsigned, or stale reports.
