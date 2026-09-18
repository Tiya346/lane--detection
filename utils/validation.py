import os

import cv2
import numpy as np

from config import settings


def get_extension(path):
    return os.path.splitext(path)[1].lower()


def validate_image_file(path):
    if path is None or str(path).strip() == "":
        return False, "No file was selected."
    if not os.path.exists(path):
        return False, "The selected file does not exist."
    if os.path.getsize(path) == 0:
        return False, "The selected file is empty."
    if get_extension(path) not in settings.ALLOWED_IMAGE_TYPES:
        allowed = ", ".join(settings.ALLOWED_IMAGE_TYPES)
        return False, "Unsupported file type. Allowed types: " + allowed
    return True, ""


def validate_video_file(path):
    if path is None or str(path).strip() == "":
        return False, "No file was selected."
    if not os.path.exists(path):
        return False, "The selected file does not exist."
    if get_extension(path) not in settings.ALLOWED_VIDEO_TYPES:
        allowed = ", ".join(settings.ALLOWED_VIDEO_TYPES)
        return False, "Unsupported video type. Allowed types: " + allowed
    return True, ""


def load_image(path):
    try:
        image = cv2.imread(path)
    except Exception:
        return None
    return image


def validate_image_array(image):
    if image is None:
        return False, "The image could not be read."
    if not isinstance(image, np.ndarray):
        return False, "The image data is not valid."
    if image.size == 0:
        return False, "The image is empty."
    height, width = image.shape[:2]
    if width < settings.MIN_IMAGE_WIDTH or height < settings.MIN_IMAGE_HEIGHT:
        return False, "The image is too small to analyse."
    return True, ""
