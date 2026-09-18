# Project Statement: LaneMark Vision

## Problem Statement

Road lane markings fade over time because of traffic wear, weather, and inconsistent repainting. Faded or damaged markings reduce lane discipline and create a genuine safety concern, particularly at night and in poor weather. At present, checking the condition of lane markings on a stretch of road is largely a manual, visual task carried out by inspection staff. That makes it slow, inconsistent between inspectors, and difficult to record in any measurable way.

The problem this project addresses is a narrower and more realistic one for a Computer Vision course project: **given a single road image, can lane markings be automatically detected and given a repeatable, explainable numeric estimate of how visible they are, without using machine learning and without pretending the result is an official assessment?**

LaneMark Vision is a rule-based Computer Vision prototype built to answer that question. It demonstrates classical image processing techniques taught in the syllabus, mainly Canny edge detection, region-of-interest segmentation, and the Hough Line Transform, rather than relying on a trained model.

**Important limitation:** The scoring thresholds used in this project (for example, what counts as CLEAR versus FADED) are project-defined values chosen and tuned during development. They are not derived from, and must not be confused with, any official road authority standard. This is a prototype for demonstrating Computer Vision concepts, not a certified road-safety inspection system.

---

## Scope of the Project

**In scope:**
- Accepting a single road image as the primary input, with optional support for a road video (frame-by-frame analysis).
- A classical Computer Vision pipeline: preprocessing, noise reduction, contrast enhancement, edge detection, region-of-interest masking, and Hough-based line detection.
- Detection and separate handling of the left and right lane markings.
- A measurable, explainable scoring system based on four visual features (segment count, coverage, continuity, and edge strength).
- Classification of lane condition into CLEAR, FADED, POOR / DAMAGED, or NOT DETECTED.
- A desktop GUI (Tkinter) showing the original image, the processed stages, and the final results together.
- Local storage of analysis history, with no external database or server.
- Basic automated testing of the core logic.

**Out of scope:**
- Deep learning or pretrained models of any kind (explicitly excluded by the project requirements).
- Real-time detection from a live camera feed.
- Curved lane detection (the current line fitting assumes straight lane segments).
- Any form of face recognition, license plate recognition, or personal data collection.
- Integration with any external road-safety database or official inspection system.
- Claims of certified accuracy or compliance with any road authority standard.

---

## Target Users

- **Myself, as the student developer**, using the project to demonstrate practical understanding of the Computer Vision concepts covered in the course, and to prepare for the project viva.
- **Course evaluators and reviewers**, who need to see a working, explainable pipeline built from classical techniques rather than a black-box model.
- **Other Computer Vision students**, as a reference example of a rule-based, feature-driven scoring pipeline that avoids deep learning.
- **Hobbyist or early-career developers** interested in seeing how Canny edge detection and the Hough Transform can be combined into a small, complete, end-to-end application.

This project is not intended for use by road authorities, traffic engineers, or anyone making real safety decisions. It is an academic prototype.

---

## High-Level Features

1. **Image and video input** – select a road image or video from the local file system, with validation for file type, size, and integrity.
2. **Classical Computer Vision pipeline** – grayscale conversion, Gaussian blur, CLAHE or histogram equalisation, Canny edge detection, morphological cleanup, and a trapezoidal region-of-interest mask.
3. **Lane detection** – probabilistic Hough Line Transform, slope-based filtering into left and right candidates, and least-squares line fitting for each side.
4. **Explainable quality scoring** – a 0–100 score for each lane side, built from four weighted, measurable features rather than an arbitrary guess.
5. **Classification** – CLEAR, FADED, POOR / DAMAGED, or NOT DETECTED, applied separately to the left and right lane and combined into an overall status.
6. **Full-stage visualisation** – the original image, the processed (enhanced) image, the edge map, and the final lane overlay are all displayed together in the GUI.
7. **Local history** – every saved analysis is recorded locally in a JSON file and can be reviewed later from within the application.
8. **Error handling** – the application does not crash on unsupported files, corrupted images, very small or very dark images, or images with no detectable lane marking; it reports a clear message instead.
9. **Privacy by design** – all processing happens on the local machine, with no external API calls and no personal data collected.
