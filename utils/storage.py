import json
import os
import time

import cv2

from config import settings


def ensure_output_dir():
    if not os.path.exists(settings.OUTPUT_DIR):
        os.makedirs(settings.OUTPUT_DIR)


def load_history():
    if not os.path.exists(settings.HISTORY_FILE):
        return []
    try:
        with open(settings.HISTORY_FILE, "r") as history_file:
            data = json.load(history_file)
    except (ValueError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return data


def save_history(records):
    ensure_output_dir()
    try:
        with open(settings.HISTORY_FILE, "w") as history_file:
            json.dump(records, history_file, indent=2)
        return True
    except OSError:
        return False


def build_record(source_path, analysis, saved_image):
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file": os.path.basename(source_path) if source_path else "unknown",
        "score": analysis["overall_score"],
        "status": analysis["overall_status"],
        "left": analysis["left"]["status"],
        "right": analysis["right"]["status"],
        "lines": analysis["detected_lines"],
        "saved_image": saved_image,
    }


def save_result(source_path, overlay_image, analysis):
    ensure_output_dir()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = "result_" + stamp + ".png"
    target = os.path.join(settings.OUTPUT_DIR, name)

    saved = cv2.imwrite(target, overlay_image)
    if not saved:
        return None, "The result image could not be saved."

    records = load_history()
    records.append(build_record(source_path, analysis, target))
    if not save_history(records):
        return target, "Image saved but history could not be updated."

    return target, ""


def clear_history():
    return save_history([])
