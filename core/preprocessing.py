import cv2
import numpy as np

from config import settings


def resize_image(image, width=settings.RESIZE_WIDTH):
    height, original_width = image.shape[:2]
    if original_width == width:
        return image
    ratio = width / float(original_width)
    new_height = int(height * ratio)
    if new_height < 1:
        new_height = 1
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)


def to_grayscale(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(gray):
    k = settings.BLUR_KERNEL
    if k % 2 == 0:
        k = k + 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def is_dark_image(gray):
    return float(np.mean(gray)) < settings.DARK_MEAN_LIMIT


def enhance_contrast(gray):
    clahe = cv2.createCLAHE(
        clipLimit=settings.CLAHE_CLIP,
        tileGridSize=(settings.CLAHE_GRID, settings.CLAHE_GRID),
    )
    return clahe.apply(gray)


def equalize_histogram(gray):
    return cv2.equalizeHist(gray)


def preprocess(image):
    resized = resize_image(image)
    gray = to_grayscale(resized)
    blurred = reduce_noise(gray)

    dark = is_dark_image(blurred)
    if dark:
        processed = equalize_histogram(blurred)
    else:
        processed = enhance_contrast(blurred)

    return {
        "resized": resized,
        "gray": gray,
        "blurred": blurred,
        "processed": processed,
        "dark": dark,
    }
