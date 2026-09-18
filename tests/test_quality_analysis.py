import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core import pipeline, quality_analysis


def make_lane_image(thickness=6, brightness=255, dashed=False):
    image = np.full((480, 640, 3), 40, dtype=np.uint8)
    colour = (brightness, brightness, brightness)
    if dashed:
        cv2.line(image, (150, 479), (200, 420), colour, thickness)
        cv2.line(image, (500, 479), (450, 420), colour, thickness)
    else:
        cv2.line(image, (150, 479), (300, 300), colour, thickness)
        cv2.line(image, (500, 479), (350, 300), colour, thickness)
    return image


def test_limit_clamps_values():
    assert quality_analysis.limit(-10) == 0.0
    assert quality_analysis.limit(150) == 100.0
    assert quality_analysis.limit(55) == 55


def test_segment_count_score_is_capped():
    segments = [(0, 0, 1, 1)] * (settings.GOOD_SEGMENT_COUNT * 3)
    assert quality_analysis.segment_count_score(segments) == 100.0
    assert quality_analysis.segment_count_score([]) == 0.0


def test_coverage_score_without_line():
    assert quality_analysis.coverage_score(None, 480) == 0.0


def test_coverage_score_of_full_line():
    score = quality_analysis.coverage_score((100, 479, 200, 279), 480)
    assert score > 80


def test_continuity_without_edges():
    empty = np.zeros((480, 640), dtype=np.uint8)
    assert quality_analysis.continuity_score((100, 479, 200, 300), empty) == 0.0


def test_continuity_without_line():
    empty = np.zeros((480, 640), dtype=np.uint8)
    assert quality_analysis.continuity_score(None, empty) == 0.0


def test_continuity_on_full_line():
    edges = np.zeros((480, 640), dtype=np.uint8)
    cv2.line(edges, (100, 479), (200, 300), 255, 3)
    score = quality_analysis.continuity_score((100, 479, 200, 300), edges)
    assert score > 90


def test_classification_thresholds():
    assert quality_analysis.classify(0) == "NOT DETECTED"
    assert quality_analysis.classify(20) == "POOR / DAMAGED"
    assert quality_analysis.classify(55) == "FADED"
    assert quality_analysis.classify(85) == "CLEAR"
    assert quality_analysis.classify(100) == "CLEAR"


def test_classification_matches_limits():
    assert quality_analysis.classify(settings.CLEAR_LIMIT) == "CLEAR"
    assert quality_analysis.classify(settings.CLEAR_LIMIT - 1) == "FADED"
    assert quality_analysis.classify(settings.FADED_LIMIT) == "FADED"
    assert quality_analysis.classify(settings.FADED_LIMIT - 1) == "POOR / DAMAGED"


def test_strength_label():
    assert quality_analysis.strength_label(80) == "High"
    assert quality_analysis.strength_label(50) == "Medium"
    assert quality_analysis.strength_label(10) == "Low"
    assert quality_analysis.strength_label(0) == "None"


def test_score_stays_in_range():
    analysis = pipeline.process_frame(make_lane_image())["analysis"]
    assert 0 <= analysis["overall_score"] <= 100
    assert 0 <= analysis["left"]["score"] <= 100
    assert 0 <= analysis["right"]["score"] <= 100


def test_blank_image_scores_zero():
    blank = np.full((480, 640, 3), 60, dtype=np.uint8)
    analysis = pipeline.process_frame(blank)["analysis"]
    assert analysis["overall_score"] == 0
    assert analysis["overall_status"] == "NOT DETECTED"


def test_clear_lane_scores_higher_than_dashed_lane():
    clear = pipeline.process_frame(make_lane_image())["analysis"]
    dashed = pipeline.process_frame(make_lane_image(dashed=True))["analysis"]
    assert clear["overall_score"] > dashed["overall_score"]


def test_faint_lane_scores_lower_than_bright_lane():
    bright = pipeline.process_frame(make_lane_image(brightness=255))["analysis"]
    faint = pipeline.process_frame(make_lane_image(brightness=80, thickness=2))["analysis"]
    assert faint["overall_score"] <= bright["overall_score"]


def test_analysis_contains_reported_fields():
    analysis = pipeline.process_frame(make_lane_image())["analysis"]
    for key in ("left", "right", "overall_score", "overall_status", "detected_lines"):
        assert key in analysis
