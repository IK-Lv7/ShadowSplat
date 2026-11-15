"""Event storage (EventStore). Local and CloudKit implementations share one interface.

Only the workstation updates the status; the edge only creates events as detected.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

STATUSES = ["detected", "queued", "extracting", "reconstructing", "rendered", "finetuned", "evaluated", "failed"]
# any status can move to failed
NEXT = {s: {STATUSES[i + 1], "failed"} for i, s in enumerate(STATUSES[:-2])}
NEXT["evaluated"] = set()
NEXT["failed"] = {"queued"}  # re-run


class InvalidTransition(ValueError):
    pass


def check_transition(cur: str, new: str) -> None:
    if new not in NEXT.get(cur, set()):
        raise InvalidTransition(f"{cur} -> {new}")


class EventStore(Protocol):
    def list_by_status(self, status: str) -> list[dict[str, Any]]: ...
    def update(self, record_name: str, status: str, error_message: str | None = None,
               metrics: dict | None = None) -> None: ...


class LocalStore:
    """One file per event: <events_dir>/<recordName>/event.json."""

    def __init__(self, events_dir: Path) -> None:
        self.dir = events_dir

    def create(self, ev: dict[str, Any]) -> str:
        """Creates an event as detected on behalf of the edge (no-op if it already exists)."""
        name = f"{ev['sceneId']}_{ev['startTimestamp']}_{ev['endTimestamp']}"
        p = self.dir / name / "event.json"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({**ev, "recordName": name, "status": "detected",
                                     "errorMessage": "", "metrics": "{}"}, indent=1))
        return name

    def _read(self, name: str) -> dict[str, Any]:
        return json.loads((self.dir / name / "event.json").read_text())

    def list_by_status(self, status: str) -> list[dict[str, Any]]:
        evs = [json.loads(p.read_text()) for p in sorted(self.dir.glob("*/event.json"))]
        return [e for e in evs if e["status"] == status]

    def update(self, record_name: str, status: str, error_message: str | None = None,
               metrics: dict | None = None) -> None:
        ev = self._read(record_name)
        if status != ev["status"]:
            check_transition(ev["status"], status)
        ev["status"] = status
        if error_message is not None:
            ev["errorMessage"] = error_message
        if metrics is not None:
            ev["metrics"] = json.dumps(metrics)
        p = self.dir / record_name / "event.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(ev, indent=1))
        tmp.replace(p)  # atomic replace


class CloudKitStore:
    """CloudKit Web Services (Public Database, server-to-server key).

    Signature: ECDSA-SHA256( "<ISO8601 date>:<base64(sha256(body))>:<subpath>" ).
    Note: QUERYABLE indexes are required on the status field and on recordName
    (set them in the CloudKit Console).
    """

    HOST = "https://api.apple-cloudkit.com"

    def __init__(self, container: str, env: str, key_id: str, key_path: Path, record_type: str = "OODEvent",
                 limit: int = 50) -> None:
        from cryptography.hazmat.primitives import serialization

        self.container, self.env, self.key_id = container, env, key_id
        self.record_type, self.limit = record_type, limit
        self.key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)

    @classmethod
    def from_env(cls, record_type: str = "OODEvent", limit: int = 50) -> "CloudKitStore":
        e = os.environ
        return cls(e["CLOUDKIT_CONTAINER"], e.get("CLOUDKIT_ENV", "development"), e["CLOUDKIT_KEY_ID"],
                   Path(e["CLOUDKIT_PRIVATE_KEY_PATH"]), record_type, limit)

    def sign_headers(self, subpath: str, body: bytes, now: datetime | None = None) -> dict[str, str]:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        date = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
        digest = base64.b64encode(hashlib.sha256(body).digest()).decode()
        sig = self.key.sign(f"{date}:{digest}:{subpath}".encode(), ec.ECDSA(hashes.SHA256()))
        return {"X-Apple-CloudKit-Request-KeyID": self.key_id,
                "X-Apple-CloudKit-Request-ISO8601Date": date,
                "X-Apple-CloudKit-Request-SignatureV1": base64.b64encode(sig).decode(),
                "Content-Type": "application/json"}

    def _post(self, op: str, payload: dict) -> dict:
        sub = f"/database/1/{self.container}/{self.env}/public/{op}"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.HOST + sub, data=body, headers=self.sign_headers(sub, body))
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    @staticmethod
    def _flat(rec: dict) -> dict[str, Any]:
        f = {k: v["value"] for k, v in rec.get("fields", {}).items()}
        return {**f, "recordName": rec["recordName"], "changeTag": rec.get("recordChangeTag")}

    def list_by_status(self, status: str) -> list[dict[str, Any]]:
        q = {"query": {"recordType": self.record_type,
                       "filterBy": [{"fieldName": "status", "comparator": "EQUALS",
                                     "fieldValue": {"value": status}}]},
             "resultsLimit": self.limit}
        return [self._flat(r) for r in self._post("records/query", q).get("records", [])]

    def _get(self, record_name: str) -> dict[str, Any]:
        r = self._post("records/lookup", {"records": [{"recordName": record_name}]})["records"][0]
        if "serverErrorCode" in r:
            raise RuntimeError(r)
        return self._flat(r)

    def create(self, ev: dict[str, Any]) -> str:
        """Creates an event as detected on behalf of the edge (fixed recordName prevents duplicates)."""
        name = f"{ev['sceneId']}_{ev['startTimestamp']}_{ev['endTimestamp']}"
        fields = {"sceneId": {"value": ev["sceneId"]}, "startTimestamp": {"value": ev["startTimestamp"]},
                  "endTimestamp": {"value": ev["endTimestamp"]}, "oodScore": {"value": ev["oodScore"]},
                  "status": {"value": "detected"}}
        res = self._post("records/modify", {"operations": [{"operationType": "create", "record": {
            "recordName": name, "recordType": self.record_type, "fields": fields}}]})["records"][0]
        if res.get("serverErrorCode") not in (None, "EXISTS"):
            raise RuntimeError(res)
        return name

    def update(self, record_name: str, status: str, error_message: str | None = None,
               metrics: dict | None = None) -> None:
        cur = self._get(record_name)
        if status != cur["status"]:
            check_transition(cur["status"], status)
        fields: dict[str, Any] = {"status": {"value": status}}
        if error_message is not None:
            fields["errorMessage"] = {"value": error_message}
        if metrics is not None:
            fields["metrics"] = {"value": json.dumps(metrics)}
        op = {"operationType": "update",
              "record": {"recordName": record_name, "recordType": self.record_type,
                         "recordChangeTag": cur["changeTag"], "fields": fields}}
        res = self._post("records/modify", {"operations": [op]})["records"][0]
        if "serverErrorCode" in res:  # e.g. CONFLICT
            raise RuntimeError(res)
