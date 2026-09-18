import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import preprocessing
from utils import validation


def make_colour_image(width=800, height=600, value=120):
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_resize_keeps_target_width():
    image = make_colour_image()
    resized = preprocessing.resize_image(image, 640)
    assert resized.shape[1] == 640


def test_resize_keeps_aspect_ratio():
    image = make_colour_image(800, 400)
    resized = preprocessing.resize_image(image, 400)
    assert resized.shape[0] == 200


def test_grayscale_has_two_dimensions():
    gray = preprocessing.to_grayscale(make_colour_image())
    assert len(gray.shape) == 2


def test_grayscale_on_gray_input():
    gray = np.zeros((50, 50), dtype=np.uint8)
    assert preprocessing.to_grayscale(gray).shape == (50, 50)


def test_noise_reduction_keeps_shape():
    gray = preprocessing.to_grayscale(make_colour_image())
    blurred = preprocessing.reduce_noise(gray)
    assert blurred.shape == gray.shape


def test_dark_image_detection():
    dark = np.full((100, 100), 20, dtype=np.uint8)
    bright = np.full((100, 100), 200, dtype=np.uint8)
    assert preprocessing.is_dark_image(dark) is True
    assert preprocessing.is_dark_image(bright) is False


def test_preprocess_returns_all_steps():
    steps = preprocessing.preprocess(make_colour_image())
    for key in ("resized", "gray", "blurred", "processed", "dark"):
        assert key in steps


def test_small_image_is_rejected():
    small = np.zeros((20, 20, 3), dtype=np.uint8)
    valid, message = validation.validate_image_array(small)
    assert valid is False
    assert message != ""


def test_normal_image_is_accepted():
    valid, message = validation.validate_image_array(make_colour_image())
    assert valid is True
    assert message == ""


def test_none_image_is_rejected():
    valid, message = validation.validate_image_array(None)
    assert valid is False


def test_unsupported_file_type():
    valid, message = validation.validate_image_file("road.txt")
    assert valid is False


def test_empty_path_is_rejected():
    valid, message = validation.validate_image_file("")
    assert valid is False


def test_missing_file_is_rejected():
    valid, message = validation.validate_image_file("no_such_image.png")
    assert valid is False
