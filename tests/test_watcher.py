import base64
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from workstation.watcher.store import CloudKitStore, InvalidTransition, LocalStore, check_transition
from workstation.watcher.watcher import enqueue

EV = {"sceneId": "scene-1", "startTimestamp": 1, "endTimestamp": 2, "oodScore": 2.5}


def test_transitions():
    check_transition("detected", "queued")
    check_transition("reconstructing", "failed")
    check_transition("failed", "queued")
    for a, b in [("detected", "rendered"), ("evaluated", "failed"), ("queued", "detected")]:
        with pytest.raises(InvalidTransition):
            check_transition(a, b)


def test_enqueue_idempotent(tmp_path):
    s = LocalStore(tmp_path / "ev")
    name = s.create(EV)
    assert s.create(EV) == name
    assert len(enqueue(s, tmp_path / "jobs")) == 1
    assert (tmp_path / "jobs" / name / "job.json").exists()
    assert s.list_by_status("queued")[0]["recordName"] == name
    assert enqueue(s, tmp_path / "jobs") == []  # re-running does not enqueue twice


def test_local_update_rejects_skip(tmp_path):
    s = LocalStore(tmp_path)
    name = s.create(EV)
    with pytest.raises(InvalidTransition):
        s.update(name, "rendered")


def _store(tmp_path):
    k = ec.generate_private_key(ec.SECP256R1())
    p = tmp_path / "k.pem"
    p.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    return CloudKitStore("iCloud.x", "development", "KEYID", p), k.public_key()


def test_cloudkit_signature_verifies(tmp_path):
    st, pub = _store(tmp_path)
    body = b'{"a":1}'
    sub = "/database/1/iCloud.x/development/public/records/query"
    h = st.sign_headers(sub, body)
    msg = f"{h['X-Apple-CloudKit-Request-ISO8601Date']}:{base64.b64encode(hashlib.sha256(body).digest()).decode()}:{sub}"
    pub.verify(base64.b64decode(h["X-Apple-CloudKit-Request-SignatureV1"]), msg.encode(),
               ec.ECDSA(hashes.SHA256()))  # raises if the signature is invalid


def test_cloudkit_update_payload(tmp_path):
    st, _ = _store(tmp_path)
    calls = []

    def fake(op, payload):
        calls.append((op, payload))
        if op == "records/lookup":
            return {"records": [{"recordName": "r", "recordChangeTag": "t1",
                                 "fields": {"status": {"value": "detected"}}}]}
        return {"records": [{"recordName": "r"}]}

    st._post = fake
    st.update("r", "queued")
    op, payload = calls[-1]
    rec = payload["operations"][0]["record"]
    assert op == "records/modify" and rec["recordChangeTag"] == "t1"
    assert rec["fields"]["status"] == {"value": "queued"}


def test_extract_and_scene_index(tmp_path):
    from workstation.extract.extract import extract, scene_index

    meta = tmp_path / "v"
    meta.mkdir()
    w = lambda n, d: (meta / f"{n}.json").write_text(json.dumps(d))
    w("scene", [{"name": "s-a", "first_sample_token": "x"}, {"name": "s-b", "first_sample_token": "t0"}])
    w("sample", [])
    w("sensor", [{"token": "sn", "channel": "CAM_FRONT"}])
    w("calibrated_sensor", [{"token": "cs", "sensor_token": "sn"}])
    sd = [{"token": f"t{i}", "sample_token": "t0", "calibrated_sensor_token": "cs", "timestamp": i * 100,
           "is_key_frame": i % 2 == 0, "next": f"t{i + 1}" if i < 4 else ""} for i in range(5)]
    w("sample_data", sd)
    assert scene_index(meta, "s-b") == 1
    r = extract(tmp_path, "v", {"sceneId": "s-b", "startTimestamp": 100, "endTimestamp": 300}, "CAM_FRONT",
                tmp_path / "e.json")
    assert (r["n_frames_event"], r["n_keyframes_event"], r["scene_idx"]) == (3, 1, 1)
    with pytest.raises(KeyError):
        extract(tmp_path, "v", {"sceneId": "none", "startTimestamp": 0, "endTimestamp": 1}, "CAM_FRONT", tmp_path / "x")
