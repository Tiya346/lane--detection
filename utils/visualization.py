import cv2
from PIL import Image, ImageTk


def to_rgb(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def fit_size(image, max_width, max_height):
    height, width = image.shape[:2]
    scale = min(max_width / float(width), max_height / float(height))
    if scale >= 1:
        return image
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def to_photo(image, max_width=380, max_height=260):
    resized = fit_size(image, max_width, max_height)
    rgb = to_rgb(resized)
    pil_image = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_image)


def build_report_text(analysis):
    left = analysis["left"]
    right = analysis["right"]
    lines = []
    lines.append("Lane marking detected: " + ("YES" if analysis["lane_found"] else "NO"))
    lines.append("")
    lines.append("Left lane : " + left["status"] + "  (score " + str(left["score"]) + ")")
    lines.append("Right lane: " + right["status"] + "  (score " + str(right["score"]) + ")")
    lines.append("")
    lines.append("Quality Score: " + str(analysis["overall_score"]) + "/100")
    lines.append("")
    lines.append("Detected Lines : " + str(analysis["detected_lines"]))
    lines.append("Left Coverage  : " + str(left["coverage"]) + "%")
    lines.append("Right Coverage : " + str(right["coverage"]) + "%")
    lines.append("Left Continuity: " + str(left["continuity"]) + "%")
    lines.append("Right Continuity: " + str(right["continuity"]) + "%")
    lines.append("Edge Strength  : " + analysis["strength_label"])
    lines.append("Edge Density   : " + str(analysis["edge_density"]) + "%")
    lines.append("")
    lines.append("Overall Status: " + analysis["overall_status"])
    lines.append("")
    lines.append("Thresholds used are project-defined and are not")
    lines.append("official road authority standards.")
    return "\n".join(lines)
