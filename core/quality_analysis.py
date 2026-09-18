import numpy as np

from config import settings
from core import lane_detection


def limit(value, low=0.0, high=100.0):
    if value < low:
        return low
    if value > high:
        return high
    return value


def segment_count_score(segments):
    count = len(segments)
    ratio = count / float(settings.GOOD_SEGMENT_COUNT)
    return limit(ratio * 100.0)


def coverage_score(line, height):
    if line is None:
        return 0.0
    x1, y1, x2, y2 = line
    roi_top = settings.ROI_TOP_LEFT[1] * height
    roi_height = height - roi_top
    if roi_height <= 0:
        return 0.0
    covered = abs(y1 - y2)
    ratio = covered / float(roi_height)
    return limit(ratio * 100.0)


def continuity_score(line, roi_edges, samples=40):
    if line is None:
        return 0.0
    x1, y1, x2, y2 = line
    height, width = roi_edges.shape[:2]
    found = 0
    checked = 0
    for i in range(samples):
        t = i / float(samples - 1)
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        if y < 0 or y >= height:
            continue
        top = max(0, y - 2)
        bottom = min(height, y + 3)
        left = max(0, x - 6)
        right = min(width, x + 7)
        if right <= left or bottom <= top:
            continue
        checked = checked + 1
        window = roi_edges[top:bottom, left:right]
        if np.count_nonzero(window) > 0:
            found = found + 1
    if checked == 0:
        return 0.0
    return limit((found / float(checked)) * 100.0)


def sample_strength(gradient, line, samples=40):
    if line is None:
        return 0.0
    x1, y1, x2, y2 = line
    height, width = gradient.shape[:2]
    values = []
    for i in range(samples):
        t = i / float(samples - 1)
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        if y < 0 or y >= height:
            continue
        left = max(0, x - 3)
        right = min(width, x + 4)
        if right <= left:
            continue
        values.append(float(np.max(gradient[y, left:right])))
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def strength_score(gradient, line):
    raw = sample_strength(gradient, line)
    ratio = raw / settings.GOOD_EDGE_STRENGTH
    return limit(ratio * 100.0), raw


def classify(score):
    if score <= 0:
        return "NOT DETECTED"
    if score >= settings.CLEAR_LIMIT:
        return "CLEAR"
    if score >= settings.FADED_LIMIT:
        return "FADED"
    return "POOR / DAMAGED"


def analyze_side(side_data, gradient, roi_edges, height):
    line = side_data["line"]
    segments = side_data["segments"]

    if line is None:
        return {
            "detected": False,
            "score": 0,
            "status": "NOT DETECTED",
            "segments": len(segments),
            "coverage": 0.0,
            "continuity": 0.0,
            "strength": 0.0,
            "strength_raw": 0.0,
        }

    count_part = segment_count_score(segments)
    coverage_part = coverage_score(line, height)
    continuity_part = continuity_score(line, roi_edges)
    strength_part, strength_raw = strength_score(gradient, line)

    score = (
        count_part * settings.WEIGHT_SEGMENTS
        + coverage_part * settings.WEIGHT_COVERAGE
        + continuity_part * settings.WEIGHT_CONTINUITY
        + strength_part * settings.WEIGHT_STRENGTH
    )
    score = int(round(limit(score)))

    return {
        "detected": True,
        "score": score,
        "status": classify(score),
        "segments": len(segments),
        "coverage": round(coverage_part, 1),
        "continuity": round(continuity_part, 1),
        "strength": round(strength_part, 1),
        "strength_raw": round(strength_raw, 1),
    }


def strength_label(value):
    if value >= 70:
        return "High"
    if value >= 40:
        return "Medium"
    if value > 0:
        return "Low"
    return "None"


def analyze_quality(lanes, gradient, roi_edges, height, edge_density=0.0):
    left = analyze_side(lanes["left"], gradient, roi_edges, height)
    right = analyze_side(lanes["right"], gradient, roi_edges, height)

    scores = []
    if left["detected"]:
        scores.append(left["score"])
    if right["detected"]:
        scores.append(right["score"])

    if len(scores) == 0:
        overall = 0
    elif len(scores) == 1:
        overall = int(round(scores[0] * 0.8))
    else:
        overall = int(round(sum(scores) / 2.0))

    total_segments = len(lanes.get("all_segments", []))
    average_strength = (left["strength"] + right["strength"]) / 2.0

    return {
        "left": left,
        "right": right,
        "overall_score": overall,
        "overall_status": classify(overall),
        "detected_lines": total_segments,
        "edge_density": round(edge_density * 100.0, 2),
        "strength_label": strength_label(average_strength),
        "lane_found": lane_detection.lane_found(lanes),
    }
