import cv2
import numpy as np

from config import settings


def find_line_segments(roi_edges):
    lines = cv2.HoughLinesP(
        roi_edges,
        settings.HOUGH_RHO,
        np.pi / settings.HOUGH_THETA_STEPS,
        settings.HOUGH_THRESHOLD,
        minLineLength=settings.HOUGH_MIN_LINE_LENGTH,
        maxLineGap=settings.HOUGH_MAX_LINE_GAP,
    )
    if lines is None:
        return []
    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        segments.append((int(x1), int(y1), int(x2), int(y2)))
    return segments


def segment_slope(segment):
    x1, y1, x2, y2 = segment
    if x2 == x1:
        return None
    return (y2 - y1) / float(x2 - x1)


def split_segments(segments, width):
    left = []
    right = []
    middle = width / 2.0
    for segment in segments:
        slope = segment_slope(segment)
        if slope is None:
            continue
        if abs(slope) < settings.MIN_SLOPE or abs(slope) > settings.MAX_SLOPE:
            continue
        x1, y1, x2, y2 = segment
        center_x = (x1 + x2) / 2.0
        if slope < 0 and center_x < middle:
            left.append(segment)
        elif slope > 0 and center_x > middle:
            right.append(segment)
    return left, right


def fit_lane_line(segments, height):
    if len(segments) == 0:
        return None
    x_points = []
    y_points = []
    for x1, y1, x2, y2 in segments:
        x_points.append(x1)
        x_points.append(x2)
        y_points.append(y1)
        y_points.append(y2)

    if max(y_points) - min(y_points) < 5:
        return None

    fit = np.polyfit(y_points, x_points, 1)
    y_bottom = height - 1
    y_top = int(min(y_points))
    x_bottom = int(np.polyval(fit, y_bottom))
    x_top = int(np.polyval(fit, y_top))
    return (x_bottom, y_bottom, x_top, y_top)


def detect_lanes(roi_edges):
    height, width = roi_edges.shape[:2]
    segments = find_line_segments(roi_edges)
    left_segments, right_segments = split_segments(segments, width)

    lanes = {
        "left": {
            "segments": left_segments,
            "line": fit_lane_line(left_segments, height),
        },
        "right": {
            "segments": right_segments,
            "line": fit_lane_line(right_segments, height),
        },
        "all_segments": segments,
    }
    return lanes


def draw_lanes(image, lanes, show_segments=True):
    output = image.copy()
    if show_segments:
        for x1, y1, x2, y2 in lanes.get("all_segments", []):
            cv2.line(output, (x1, y1), (x2, y2), settings.SEGMENT_COLOR, 2)

    for side in ("left", "right"):
        line = lanes[side]["line"]
        if line is None:
            continue
        x1, y1, x2, y2 = line
        cv2.line(output, (x1, y1), (x2, y2), settings.LANE_COLOR, settings.LANE_THICKNESS)
    return output


def lane_found(lanes):
    return lanes["left"]["line"] is not None or lanes["right"]["line"] is not None
