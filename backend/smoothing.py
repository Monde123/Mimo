import numpy as np


def smooth_landmarks(frames, window=5):
    """Simple rolling average smoothing for landmark sequences."""
    if not frames:
        return []

    smoothed = []
    for i, frame in enumerate(frames):
        out = {}
        for key in ("handsL", "handsR", "predictions", "bodyPose"):
            if key not in frame:
                continue
            points = frame[key]
            if not points:
                out[key] = []
                continue

            smoothed_points = []
            for idx, point in enumerate(points):
                x_values = []
                y_values = []
                z_values = []
                v_values = []
                for delta in range(-window // 2, window // 2 + 1):
                    pos = i + delta
                    if pos < 0 or pos >= len(frames):
                        continue
                    candidate = frames[pos].get(key, [])
                    if idx < len(candidate):
                        p = candidate[idx]
                        x_values.append(float(p.get("x", 0.0)))
                        y_values.append(float(p.get("y", 0.0)))
                        z_values.append(float(p.get("z", 0.0)))
                        v_values.append(float(p.get("visibility", 1.0)))
                if not x_values:
                    smoothed_points.append(point)
                else:
                    smoothed_points.append({
                        "x": float(np.mean(x_values)),
                        "y": float(np.mean(y_values)),
                        "z": float(np.mean(z_values)),
                        "visibility": float(np.mean(v_values)),
                    })
            out[key] = smoothed_points
        out["frame"] = frame.get("frame", i)
        out["width"] = frame.get("width")
        out["height"] = frame.get("height")
        smoothed.append(out)
    return smoothed
