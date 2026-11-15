# Phase 2: Watcher and event store

- Config: `configs/phase2_watcher.yaml`; implementation: `workstation/watcher/`
- Verified (local store): five events from the edge runtime imported → `detected` moved to `queued` → job directories created. A second run does not enqueue them again. Unit tests cover status transitions, idempotency, signature verification and request contents.
- **Verified (CloudKit, Public Database, Development environment)**: signed `records/query` requests succeed. A `detected` record is picked up by the watcher and updated to `queued` (confirmed by querying again). Updates with the server-to-server key succeed. Failures are recorded as `failed` with `errorMessage`.

## Design decisions

| Decision | Why | Rejected alternative |
|---|---|---|
| Swappable `EventStore`, local implementation first | queueing, transitions and resumption can be tested without credentials | CloudKit only (nothing testable without keys) |
| Transitions checked by `check_transition` | enforces in code that only the workstation advances the status, along a fixed path | write any status |
| Jobs stored as `<jobs_dir>/<recordName>/` | survive restarts; completed work can be detected (resumable) | in-memory queue (lost on restart), a database (extra dependency) |
| `cryptography` for signing | required for the CloudKit Web Services server-to-server signature (ECDSA-SHA256) | standard library only (no ECDSA) |
| Standard `urllib` for HTTP | no extra dependency | `requests` |
| `recordName` = `sceneId_start_end` | prevents duplicate events for the same interval | random UUID |

## CloudKit setup (manual)
1. Create a container and the `OODEvent` record type in the CloudKit Console (fields are listed in `CLAUDE.md`).
2. Add QUERYABLE indexes on `status` and `recordName` (needed for queries).
3. Create a key under "Server-to-Server Keys" and put the private key path, key ID and container ID in `.env` (see `.env.example`).
4. Check in the Public Database security roles that the server-to-server key can read and write the records.
