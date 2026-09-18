# LaneMark Vision

### Road Lane Marking Quality Detector

A desktop Computer Vision application that takes a road image (or a road video) and estimates the visibility of its lane markings using classical image processing techniques.

---

## Overview

I built LaneMark Vision as part of my Computer Vision course project. It is a rule-based prototype, not a deep learning system: every result it produces comes from measurements I compute directly from pixels using techniques covered in the syllabus, mainly Canny edge detection, region-of-interest segmentation, and the Hough Line Transform.

The application accepts a road image, runs it through the full Computer Vision pipeline, detects the left and right lane lines, and scores their visibility from 0 to 100. That score is turned into a simple classification: **CLEAR**, **FADED**, **POOR / DAMAGED**, or **NOT DETECTED**.

This is a Computer Vision prototype built for academic purposes. It is not an official road-safety inspection tool, and the thresholds used for classification are project-defined values I tuned myself, not any recognised road authority standard.

---

## Features

- Select and load a road image (`.jpg`, `.jpeg`, `.png`, `.bmp`)
- Optionally load and play a road video (`.mp4`, `.avi`, `.mov`), with frames analysed live
- Full Computer Vision pipeline: resize, grayscale, noise reduction, contrast enhancement (CLAHE or histogram equalisation), Canny edge detection, region-of-interest masking, and Hough Line Transform
- Left and right lane detection with line fitting and overlay drawing
- Explainable quality scoring built from four measurable features: segment count, coverage, continuity, and edge strength
- Left/right lane classification and an overall status
- A four-panel display showing the original image, the processed image, the edge map, and the final lane overlay together
- Local history of past analyses, saved automatically and viewable from the app
- Graceful handling of bad input: unsupported files, corrupted images, images too small or too dark, and images with no detectable lane marking
- Everything runs locally. No external APIs, no internet requirement, and no personal data is collected

---

## Technologies / Tools Used

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| OpenCV (`opencv-python`) | Image processing: grayscale conversion, blurring, CLAHE, Canny edge detection, Hough Transform, morphology |
| NumPy | Array operations, line fitting (`polyfit`), feature calculations |
| Tkinter | Desktop GUI |
| Pillow (`PIL`) | Converting OpenCV images for display in Tkinter |
| pytest | Automated testing |

No deep learning frameworks (TensorFlow, PyTorch, YOLO) are used. The project is intentionally built on classical Computer Vision methods.

---

## Project Structure

```
lanemark-vision/
├── main.py
├── requirements.txt
├── .gitignore
├── config/
│   └── settings.py
├── core/
│   ├── preprocessing.py
│   ├── edge_detection.py
│   ├── lane_detection.py
│   ├── quality_analysis.py
│   └── pipeline.py
├── gui/
│   └── app.py
├── utils/
│   ├── validation.py
│   ├── visualization.py
│   └── storage.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_lane_detection.py
│   └── test_quality_analysis.py
├── data/
│   └── sample/
└── outputs/
```

---

## Steps to Install & Run

**1. Clone or unzip the project**

```bash
cd lanemark-vision
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

On Linux, Tkinter may need to be installed separately:

```bash
sudo apt install python3-tk
```

**4. Run the application**

```bash
python main.py
```

**5. Using the app**

- Click **Select Image** and choose a road photo, or **Select Video** to load a road video.
- Click **Process Image** to run the pipeline and see the results.
- Use **Save Result** to store the analysis in local history, and **View History** to browse past results.
- Use **Reset** to clear the current session.

Sample images can be placed in `data/sample/` for quick testing.

---

## Instructions for Testing

The project includes 38 automated tests written with `pytest`, covering preprocessing, lane detection, and quality scoring. The tests build their own synthetic images in code, so they run without needing any sample photos on disk.

**Run the full test suite:**

```bash
python -m pytest
```

**Run a specific test file:**

```bash
python -m pytest tests/test_quality_analysis.py
```

**Run with detailed output:**

```bash
python -m pytest -v
```

All 38 tests pass in the submitted version of the project.

---


