import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import edge_detection, lane_detection, pipeline, preprocessing


def make_lane_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :] = (40, 40, 40)
    cv2.line(image, (150, 479), (300, 300), (255, 255, 255), 6)
    cv2.line(image, (500, 479), (350, 300), (255, 255, 255), 6)
    return image


def make_blank_image():
    return np.full((480, 640, 3), 60, dtype=np.uint8)


def test_canny_returns_binary_image():
    gray = preprocessing.to_grayscale(make_lane_image())
    edges = edge_detection.detect_edges(gray)
    values = np.unique(edges)
    for value in values:
        assert value in (0, 255)


def test_roi_removes_top_of_image():
    edges = np.full((480, 640), 255, dtype=np.uint8)
    masked = edge_detection.apply_roi(edges)
    assert np.count_nonzero(masked[0:100, :]) == 0
    assert np.count_nonzero(masked) > 0


def test_edge_density_range():
    edges = np.zeros((100, 100), dtype=np.uint8)
    assert edge_detection.edge_density(edges) == 0.0
    edges[:, :] = 255
    assert edge_detection.edge_density(edges) == 1.0


def test_segment_slope():
    assert lane_detection.segment_slope((0, 0, 10, 10)) == 1.0
    assert lane_detection.segment_slope((5, 0, 5, 10)) is None
    assert lane_detection.segment_slope((0, 10, 10, 0)) == -1.0


def test_split_segments_by_side():
    left = (100, 400, 200, 300)
    right = (500, 400, 400, 300)
    horizontal = (100, 300, 400, 305)
    left_list, right_list = lane_detection.split_segments([left, right, horizontal], 640)
    assert left in left_list
    assert right in right_list
    assert horizontal not in left_list
    assert horizontal not in right_list


def test_fit_lane_line_with_no_segments():
    assert lane_detection.fit_lane_line([], 480) is None


def test_detect_lanes_on_clear_image():
    result = pipeline.process_frame(make_lane_image())
    lanes = result["lanes"]
    assert lane_detection.lane_found(lanes) is True
    assert lanes["left"]["line"] is not None
    assert lanes["right"]["line"] is not None


def test_detect_lanes_on_blank_image():
    result = pipeline.process_frame(make_blank_image())
    assert lane_detection.lane_found(result["lanes"]) is False
    assert result["analysis"]["overall_status"] == "NOT DETECTED"


def test_overlay_has_same_shape_as_original():
    result = pipeline.process_frame(make_lane_image())
    assert result["overlay"].shape == result["original"].shape


def test_pipeline_returns_expected_keys():
    result = pipeline.process_frame(make_lane_image())
    for key in ("original", "processed", "edges", "overlay", "analysis"):
        assert key in result
