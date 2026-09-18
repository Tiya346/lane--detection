import cv2
import numpy as np

from config import settings


def detect_edges(gray):
    return cv2.Canny(gray, settings.CANNY_LOW, settings.CANNY_HIGH)


def clean_edges(edges):
    size = settings.MORPH_KERNEL
    kernel = np.ones((size, size), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed


def build_roi_polygon(height, width):
    points = [
        settings.ROI_BOTTOM_LEFT,
        settings.ROI_TOP_LEFT,
        settings.ROI_TOP_RIGHT,
        settings.ROI_BOTTOM_RIGHT,
    ]
    polygon = []
    for x_ratio, y_ratio in points:
        x = int(x_ratio * width)
        y = int(y_ratio * height)
        polygon.append((x, y))
    return np.array([polygon], dtype=np.int32)


def apply_roi(edges):
    height, width = edges.shape[:2]
    mask = np.zeros_like(edges)
    polygon = build_roi_polygon(height, width)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def edge_density(edges):
    total = edges.shape[0] * edges.shape[1]
    if total == 0:
        return 0.0
    white = int(np.count_nonzero(edges))
    return white / float(total)


def gradient_magnitude(gray):
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return magnitude


def run_edge_stage(processed_gray):
    raw_edges = detect_edges(processed_gray)
    cleaned = clean_edges(raw_edges)
    masked = apply_roi(cleaned)
    return {
        "edges": cleaned,
        "roi_edges": masked,
        "density": edge_density(masked),
        "gradient": gradient_magnitude(processed_gray),
    }
