"""One-Euro de smooth_landmarks sur des coordonnees en metres (bodyPose = points 3D MediaPipe).

Bug vise : un filtre regle pour des pixels (beta=0.007) n'accelere jamais sur des metres ; il se
comporte en passe-bas fixe a 1 Hz, retarde le mouvement reel et devient pire que le signal brut.
Oracles : gain d'un passe-bas du 1er ordre calcule a la main, trajectoires reelles de la mocap
COMFI (Zenodo 18163808, CC BY 4.0), reduction du bruit sur un point immobile.
"""
from pathlib import Path

import numpy as np
import pytest

from backend.smoothing import smooth_landmarks

FPS = 40.0
MOCAP = Path(__file__).resolve().parents[1] / "data/comfi/1012/mocap/aligned/joint_center_positions.csv"


def _frames(track: np.ndarray) -> list[dict]:
    """track (T,3) -> frames ou les 33 points du corps suivent la meme trajectoire."""
    return [{"bodyPose": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "visibility": 1.0}] * 33,
             "handsL": [], "handsR": []} for p in track]


def _smoothed(track: np.ndarray, **kw) -> np.ndarray:
    out = smooth_landmarks(_frames(track), fps=FPS, max_gap_frames=0, **kw)
    return np.array([[f["bodyPose"][0][k] for k in "xyz"] for f in out])


def test_real_motion_amplitude_is_kept():
    # Bras de jumping jack : 0.3 m d'amplitude a 1.5 Hz. Un passe-bas fixe a 1 Hz n'en garderait que
    # 1/sqrt(1 + 1.5**2) = 0.55 (gain du 1er ordre, calcul a la main).
    t = np.arange(int(4 * FPS)) / FPS
    track = np.zeros((len(t), 3))
    track[:, 1] = 0.3 * np.sin(2 * np.pi * 1.5 * t)
    out = _smoothed(track)
    settled = out[int(FPS):, 1]
    assert np.ptp(settled) / 2 >= 0.9 * 0.3


def test_still_point_noise_is_still_reduced():
    rng = np.random.default_rng(1)
    noisy = rng.normal(0, 0.01, (400, 3))
    out = _smoothed(noisy)
    assert out[40:].std() < 0.7 * noisy[40:].std()


@pytest.mark.skipif(not MOCAP.exists(), reason="mocap COMFI absente (data/ non versionne)")
def test_noisy_real_mocap_gets_closer_to_truth():
    pd = pytest.importorskip("pandas")
    gt = pd.read_csv(MOCAP)
    rng = np.random.default_rng(0)
    for joint in ("Left_Knee", "Right_Wrist", "Left_Ankle"):
        clean = gt[[f"{joint}_{x}[mm]" for x in "XYZ"]].to_numpy() / 1000
        noisy = clean + rng.normal(0, 0.01, clean.shape)
        out = _smoothed(noisy)
        err_raw = np.sqrt(((noisy - clean) ** 2).sum(1).mean())
        err_out = np.sqrt(((out - clean) ** 2).sum(1).mean())
        assert err_out < err_raw, f"{joint}: filtre {err_out*100:.2f} cm >= brut {err_raw*100:.2f} cm"


@pytest.mark.parametrize("kw, msg", [
    ({"fps": 0.0}, "fps"),
    ({"fps": float("nan")}, "fps"),
    ({"fps": -30.0}, "fps"),
    ({"fps": "30"}, "fps"),
    ({"min_cutoff": 0.0}, "min_cutoff"),
    ({"min_cutoff": float("inf")}, "min_cutoff"),
    ({"beta": -1.0}, "beta"),
    ({"max_gap_frames": -1}, "max_gap_frames"),
])
def test_invalid_parameters_raise(kw, msg):
    args = {"fps": FPS, **kw}
    with pytest.raises(ValueError, match=msg):
        smooth_landmarks(_frames(np.zeros((3, 3))), **args)


def test_empty_and_single_frame():
    assert smooth_landmarks([], fps=FPS) == []
    one = _smoothed(np.array([[0.1, -0.2, 0.3]]))
    np.testing.assert_allclose(one, [[0.1, -0.2, 0.3]])
