import numpy as np
from scipy.special import logsumexp  # noqa: F401  (reference for comparison)

from models.ood_score import OodConfig, extract_events, frame_score, pixel_energy

CFG = OodConfig(temperature=1.0, top_k_percent=10.0, threshold=0.0, margin_sec=1.0, min_frames=2)


def test_energy_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(19, 8, 8)) * 5
    for t in (1.0, 2.5):
        np.testing.assert_allclose(pixel_energy(x, t), -t * logsumexp(x / t, axis=0), atol=1e-10)


def test_energy_stable_for_large_logits():
    x = np.full((3, 2, 2), 1e4)
    assert np.isfinite(pixel_energy(x)).all()


def test_frame_score_top_k():
    e = np.arange(100, dtype=float).reshape(10, 10)
    assert frame_score(e, 10.0) == np.mean(np.arange(90, 100))


def test_events_min_frames_and_margin():
    ts = [i * 100_000 for i in range(10)]
    sc = [-1, 1, -1, 1, 1, 1, -1, -1, 1, -1]
    ev = extract_events(ts, sc, CFG)
    assert len(ev) == 1  # single frames (index 1, 8) are below min_frames
    assert (ev[0].start_us, ev[0].end_us) == (300_000 - 1_000_000, 500_000 + 1_000_000)
    assert ev[0].score == 1


def test_events_merge_overlapping_margins():
    ts = [i * 100_000 for i in range(10)]
    sc = [1, 1, -1, -1, 1, 1, -1, -1, -1, -1]
    assert len(extract_events(ts, sc, CFG)) == 1
