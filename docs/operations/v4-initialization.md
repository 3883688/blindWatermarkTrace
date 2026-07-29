# V4 Offline Initialization

Stop all web and worker processes before reset. The workflow requires an absolute upload directory below the configured workspace and the exact confirmation `RESET-V4:{environment}:{database_name}`.

Run preflight, create and verify both the PostgreSQL custom-format dump and upload archive, then apply. The ready marker is written only after identity preservation, schema verification, key rotation, and smoke tests succeed. It contains schema, model, and key IDs only; the key secret is never printed.

If any phase fails, keep the service offline and run restore against the verified database dump and upload archive. Restore removes only validated children of the configured upload directory and never follows archive links.
