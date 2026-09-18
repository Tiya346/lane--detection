import cv2

from core import preprocessing
from core import edge_detection
from core import lane_detection
from core import quality_analysis
from utils import validation


def process_frame(frame):
    steps = preprocessing.preprocess(frame)
    edges = edge_detection.run_edge_stage(steps["processed"])
    lanes = lane_detection.detect_lanes(edges["roi_edges"])

    height = steps["resized"].shape[0]
    analysis = quality_analysis.analyze_quality(
        lanes, edges["gradient"], edges["roi_edges"], height, edges["density"]
    )
    overlay = lane_detection.draw_lanes(steps["resized"], lanes)

    return {
        "original": steps["resized"],
        "processed": steps["processed"],
        "edges": edges["roi_edges"],
        "overlay": overlay,
        "analysis": analysis,
        "dark": steps["dark"],
        "lanes": lanes,
    }


def process_image_file(path):
    valid, message = validation.validate_image_file(path)
    if not valid:
        return None, message

    image = validation.load_image(path)
    if image is None:
        return None, "The image file could not be opened or is corrupted."

    valid, message = validation.validate_image_array(image)
    if not valid:
        return None, message

    result = process_frame(image)
    result["source"] = path

    if not result["analysis"]["lane_found"]:
        return result, "No lane marking could be detected in this image."

    return result, ""


def open_video(path):
    valid, message = validation.validate_video_file(path)
    if not valid:
        return None, message

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return None, "The video file could not be opened."
    return capture, ""
