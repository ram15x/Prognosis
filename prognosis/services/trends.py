
import numpy as np

RANK = {"low": 0, "moderate": 1, "high": 2}


def summarise(points: list[dict]) -> dict:
    """points: [{'at': datetime, 'probability': float, 'band': str}] oldest first."""
    if len(points) < 2:
        return {"n": len(points), "direction": "not enough visits", "delta": None, "slope_per_month": None, "band_change": None}
    first, last = points[0], points[-1]
    delta = last["probability"] - first["probability"]
    days = np.array([(p["at"] - first["at"]).total_seconds() / 86400 for p in points])
    probs = np.array([p["probability"] for p in points])
    slope = None
    if days[-1] > 0:
        slope = float(np.polyfit(days, probs, 1)[0] * 30.44)
    # two readings a few points apart are mostly input noise, so we only call it a move above 5 points
    if abs(delta) < 0.05:
        direction = "stable"
    else:
        direction = "rising" if delta > 0 else "falling"
    change = None
    if first["band"] != last["band"]:
        change = f"{first['band']} to {last['band']}"
    return {"n": len(points), "direction": direction, "delta": float(delta),
            "slope_per_month": slope, "band_change": change}
