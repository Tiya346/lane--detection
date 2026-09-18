import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config import settings
from core import pipeline
from utils import storage, visualization


class LaneMarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LaneMark Vision")
        self.root.geometry("900x760")
        self.root.configure(bg="#f2f2f2")

        self.image_path = None
        self.result = None
        self.photos = {}
        self.video_capture = None
        self.video_running = False
        self.frame_number = 0

        self.build_header()
        self.build_buttons()
        self.build_image_area()
        self.build_analysis_area()
        self.build_status_bar()

    def build_header(self):
        header = tk.Frame(self.root, bg="#f2f2f2")
        header.pack(fill="x", pady=(10, 4))

        title = tk.Label(
            header,
            text="LANEMARK VISION",
            font=("Arial", 20, "bold"),
            bg="#f2f2f2",
        )
        title.pack()

        subtitle = tk.Label(
            header,
            text="Road Lane Marking Quality Detector  (Computer Vision prototype)",
            font=("Arial", 10),
            bg="#f2f2f2",
        )
        subtitle.pack()

    def build_buttons(self):
        bar = tk.Frame(self.root, bg="#f2f2f2")
        bar.pack(pady=8)

        tk.Button(bar, text="Select Image", width=13, command=self.select_image).grid(row=0, column=0, padx=4)
        tk.Button(bar, text="Process Image", width=13, command=self.process_image).grid(row=0, column=1, padx=4)
        tk.Button(bar, text="Select Video", width=13, command=self.select_video).grid(row=0, column=2, padx=4)
        self.stop_button = tk.Button(bar, text="Stop Video", width=13, command=self.stop_video, state="disabled")
        self.stop_button.grid(row=0, column=3, padx=4)
        tk.Button(bar, text="Save Result", width=13, command=self.save_result).grid(row=0, column=4, padx=4)
        tk.Button(bar, text="View History", width=13, command=self.show_history).grid(row=0, column=5, padx=4)
        tk.Button(bar, text="Reset", width=13, command=self.reset).grid(row=0, column=6, padx=4)

    def build_image_area(self):
        area = tk.Frame(self.root, bg="#f2f2f2")
        area.pack(pady=6)

        self.panels = {}
        titles = [
            ("original", "Original Image", 0, 0),
            ("processed", "Processed (Gray + Enhanced)", 0, 1),
            ("edges", "Canny Edges inside ROI", 1, 0),
            ("overlay", "Detected Lane Markings", 1, 1),
        ]

        for key, text, row, column in titles:
            box = tk.LabelFrame(area, text=text, bg="#f2f2f2", font=("Arial", 9, "bold"))
            box.grid(row=row, column=column, padx=8, pady=6)
            label = tk.Label(box, bg="#dddddd", width=54, height=11)
            label.pack(padx=4, pady=4)
            self.panels[key] = label

    def build_analysis_area(self):
        box = tk.LabelFrame(self.root, text="Analysis", bg="#f2f2f2", font=("Arial", 10, "bold"))
        box.pack(fill="both", expand=True, padx=14, pady=6)

        self.analysis_text = tk.Text(box, height=11, font=("Consolas", 10), bg="white")
        self.analysis_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.analysis_text.insert("1.0", "Select a road image and click Process Image.")
        self.analysis_text.config(state="disabled")

    def build_status_bar(self):
        self.status = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief="sunken",
            anchor="w",
            bg="#e6e6e6",
        )
        self.status.pack(fill="x", side="bottom")

    def set_status(self, message):
        self.status.config(text=message)

    def set_analysis_text(self, text):
        self.analysis_text.config(state="normal")
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", text)
        self.analysis_text.config(state="disabled")

    def show_panel(self, key, image):
        photo = visualization.to_photo(image)
        self.photos[key] = photo
        self.panels[key].config(image=photo, width=photo.width(), height=photo.height())

    def clear_panels(self):
        for key in self.panels:
            self.panels[key].config(image="", width=54, height=11)
        self.photos = {}

    def select_image(self):
        self.stop_video()
        path = filedialog.askopenfilename(
            title="Select a road image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return

        self.image_path = path
        self.result = None
        result, message = pipeline.process_image_file(path)
        if result is None:
            messagebox.showerror("Invalid image", message)
            self.image_path = None
            self.set_status("Invalid image")
            return

        self.clear_panels()
        self.show_panel("original", result["original"])
        self.set_analysis_text("Image loaded. Click Process Image to analyse the lane markings.")
        self.set_status("Loaded: " + path)

    def process_image(self):
        if self.image_path is None:
            messagebox.showwarning("No image", "Please select an image first.")
            return

        result, message = pipeline.process_image_file(self.image_path)
        if result is None:
            messagebox.showerror("Processing failed", message)
            self.set_status("Processing failed")
            return

        self.result = result
        self.show_panel("original", result["original"])
        self.show_panel("processed", result["processed"])
        self.show_panel("edges", result["edges"])
        self.show_panel("overlay", result["overlay"])

        report = visualization.build_report_text(result["analysis"])
        if result["dark"]:
            report = "Note: the image is dark, histogram equalisation was applied.\n\n" + report
        self.set_analysis_text(report)

        if message:
            messagebox.showinfo("Result", message)
            self.set_status(message)
        else:
            self.set_status(
                "Analysis complete - "
                + result["analysis"]["overall_status"]
                + " ("
                + str(result["analysis"]["overall_score"])
                + "/100)"
            )

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select a road video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")],
        )
        if not path:
            return

        self.stop_video()
        capture, message = pipeline.open_video(path)
        if capture is None:
            messagebox.showerror("Invalid video", message)
            return

        self.video_capture = capture
        self.video_running = True
        self.frame_number = 0
        self.image_path = path
        self.stop_button.config(state="normal")
        self.set_status("Playing video: " + path)
        self.play_video()

    def play_video(self):
        if not self.video_running or self.video_capture is None:
            return

        ok, frame = self.video_capture.read()
        if not ok:
            self.stop_video()
            self.set_status("Video finished")
            return

        self.frame_number = self.frame_number + 1
        if self.frame_number % 2 == 0:
            self.root.after(20, self.play_video)
            return

        try:
            result = pipeline.process_frame(frame)
        except Exception:
            self.stop_video()
            messagebox.showerror("Video error", "This video frame could not be processed.")
            return

        self.result = result
        self.result["source"] = self.image_path
        self.show_panel("original", result["original"])
        self.show_panel("processed", result["processed"])
        self.show_panel("edges", result["edges"])
        self.show_panel("overlay", result["overlay"])
        self.set_analysis_text(visualization.build_report_text(result["analysis"]))
        self.root.after(20, self.play_video)

    def stop_video(self):
        self.video_running = False
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.stop_button.config(state="disabled")

    def save_result(self):
        if self.result is None:
            messagebox.showwarning("Nothing to save", "Please process an image first.")
            return

        target, message = storage.save_result(
            self.result.get("source", self.image_path),
            self.result["overlay"],
            self.result["analysis"],
        )
        if target is None:
            messagebox.showerror("Save failed", message)
            return

        if message:
            messagebox.showwarning("Saved with warning", message)
        else:
            messagebox.showinfo("Saved", "Result saved to:\n" + target)
        self.set_status("Saved: " + target)

    def show_history(self):
        records = storage.load_history()
        window = tk.Toplevel(self.root)
        window.title("Analysis History")
        window.geometry("720x340")

        if len(records) == 0:
            tk.Label(window, text="No previous analysis found.", font=("Arial", 11)).pack(pady=30)
            return

        columns = ("time", "file", "score", "status", "left", "right", "lines")
        table = ttk.Treeview(window, columns=columns, show="headings")
        for column in columns:
            table.heading(column, text=column.capitalize())
            table.column(column, width=100, anchor="center")
        table.column("file", width=150)
        table.column("time", width=140)

        for record in reversed(records):
            table.insert(
                "",
                "end",
                values=(
                    record.get("time", ""),
                    record.get("file", ""),
                    record.get("score", ""),
                    record.get("status", ""),
                    record.get("left", ""),
                    record.get("right", ""),
                    record.get("lines", ""),
                ),
            )
        table.pack(fill="both", expand=True, padx=8, pady=8)

    def reset(self):
        self.stop_video()
        self.image_path = None
        self.result = None
        self.clear_panels()
        self.set_analysis_text("Select a road image and click Process Image.")
        self.set_status("Ready")


def run():
    root = tk.Tk()
    LaneMarkApp(root)
    root.mainloop()
