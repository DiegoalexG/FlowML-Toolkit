import os
import cv2
import importlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
                               QMessageBox, QFrame, QComboBox, QTextEdit, QApplication)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from scipy.interpolate import UnivariateSpline

"""VOF prediction window"""
class VOFPredictionView(QWidget):
    def __init__(self, params):
        super().__init__()
        """
        View to load data, select a model, run predictions, display videos of expected and predicted behavior, 
        show plots of geometric parameters, and display a log of computed metrics.
        """
        self.params = params

        module_path, class_name = self.params["methods"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        self.VOFClass = getattr(module, class_name)()

        self.data_loaded = False
        self.model_selected = False

        main_layout = QVBoxLayout()

        # File load button
        self.load_button = QPushButton("Load .npz file")
        self.load_button.clicked.connect(self.load_file)
        main_layout.addWidget(self.load_button)

        # Parameters label
        self.parameters_label = QLabel("No data loaded")
        self.parameters_label.setAlignment(Qt.AlignCenter)
        self.parameters_label.setStyleSheet("""font-size: 16px;
                                               font-weight: bold;""")
        main_layout.addWidget(self.parameters_label)

        # Video area
        video_layout = QHBoxLayout()
        self.expected_frame, self.expected_label = self.create_video_box("Expected")
        video_layout.addWidget(self.expected_frame, stretch=1)
        self.predicted_frame, self.predicted_label = self.create_video_box("Predicted")
        video_layout.addWidget(self.predicted_frame, stretch=1)
        main_layout.addLayout(video_layout, stretch=3)

        # Model selection
        self.model_selector = QComboBox()
        self.model_selector.addItem("Select a model")
        self.model_selector.addItems(["Model 50→1", "Model 50→50", "Model 100→100"])
        self.model_selector.currentIndexChanged.connect(self.on_model_selected)
        main_layout.addWidget(self.model_selector)

        # Compare all models
        self.compare_button = QPushButton("Compare all models")
        self.compare_button.clicked.connect(self.compare_models)
        self.compare_button.setEnabled(False)
        main_layout.addWidget(self.compare_button)

        # Prediction button
        self.predict_button = QPushButton("Run prediction")
        self.predict_button.clicked.connect(self.run_prediction)
        self.predict_button.setEnabled(False)
        main_layout.addWidget(self.predict_button)       

        # Save results button
        self.save_button = QPushButton("Save results to .npz")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        main_layout.addWidget(self.save_button)

        # Loading label
        self.loading_label = QLabel("Running prediction...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setVisible(False)
        main_layout.addWidget(self.loading_label)

        # Plots and log area
        bottom_layout = QHBoxLayout()

        self.plot_label = QLabel("[Plots will appear here]")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setFrameStyle(QFrame.Box)
        self.plot_label.setMinimumHeight(250)
        bottom_layout.addWidget(self.plot_label, stretch=3)

        self.text_box = QTextEdit()
        self.text_box.setPlaceholderText("Logs...")
        self.text_box.setMinimumHeight(250)
        bottom_layout.addWidget(self.text_box, stretch=1)
        main_layout.addLayout(bottom_layout, stretch=2)

        self.setLayout(main_layout)

    def load_file(self):
        """
        Loads a sample stored in an NPZ file, passes it to the class responsible for handling the methods
        and creates a video to visualize the expected values.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Open NPZ File", self.params["examples"], "NPZ Files (*.npz)")

        if not file_path:
            return

        try:
            self.data = np.load(file_path, allow_pickle=True)

            clean_files = [x.strip("'") for x in self.data.files]
            required_keys = {"sample", "Re", "We", "beta", "Wi", "x_range", "y_range", "t_range"}
            if not required_keys.issubset(set(clean_files)):
                raise ValueError("Invalid file structure")

            # Clears previous prediction video
            self.predicted_label.clear()
            self.predicted_label.setText("[Video will appear here]")

            # Disables save button
            self.save_button.setEnabled(False)

            # Enables compare models button
            self.compare_button.setEnabled(True)

            # Clears previous plot
            self.plot_label.clear()
            self.plot_label.setText("[Parameter plot will appear here]")

            # Clears previous logs
            self.text_box.clear()

            # Stops prediction timer if it exists
            if hasattr(self, "timer_prediction"):
                self.timer_prediction.stop()

            # Sets the sample for the methods class
            self.VOFClass.set_data(self.data)

            # Marks that a valid sample has been loaded
            self.data_loaded = True
            self.update_predict_button()

            # Updates parameters label
            Re = self.data["Re"]
            We = self.data["We"]
            beta = self.data["beta"]
            Wi = self.data["Wi"]
            self.parameters_label.setText(f"Re = {Re:.4f}, We = {We:.4f}, β = {beta:.4f} and Wi = {Wi:.4f}")

            # Loads the video
            sample = self.data["sample"]
            self.frames = sample
            self.current_frame = 0

            if hasattr(self, "timer"):
                self.timer.stop()

            self.timer = QTimer()
            self.timer.timeout.connect(self.update_expected_video)
            self.timer.start(30)

        except Exception:
            QMessageBox.critical(self, "Error", "Invalid .npz file")
            self.data_loaded = False
            self.update_predict_button()

    def update_expected_video(self):
        """
        Updates the video showing the behavior of the original simulation.
        """
        # Loops back to the beginning when the video ends
        if self.current_frame >= len(self.frames):
            self.current_frame = 0

        frame = self.frames[self.current_frame]

        # Apply viridis colormap
        colored = cm.viridis(frame)

        # Converts to uint8 RGB
        img = (colored[:, :, :3] * 255).astype(np.uint8)

        # Resizes image
        scale = 6
        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        img = np.ascontiguousarray(img)

        h, w, ch = img.shape
        bytes_per_line = ch * w

        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.expected_label.setPixmap(QPixmap.fromImage(qimg))

        self.current_frame += 1

    def update_prediction_video(self):
        """
        Updates the comparison video between expected and predicted simulation.
        """
        # Loops back to the beginning when the video ends
        if self.current_frame >= len(self.expected_frames):
            self.current_frame = 0

        # The initial simulation context is extracted from expected values
        t = self.current_frame

        expected = self.expected_frames[t]

        h, w = expected.shape
        w_half = w // 2

        # Initial simulation context
        if t < self.context_steps:
            frame = expected.copy()

            colored = cm.viridis(frame)
            img = (colored[:, :, :3] * 255).astype(np.uint8)

            # Resizes image
            scale = 6
            img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
            h_img, w_img, _ = img.shape

            # Centered text
            text = "Initial simulation context"
            (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(img, text, ((w_img - text_w) // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Predictions start
        else:
            pred_idx = t - self.context_steps

            predicted = self.predicted_frames[pred_idx]

            # comparison image
            img_comp = np.zeros((h, w + 2))

            # Left side for expected values
            img_comp[:, :w_half] = expected[:, :w_half]

            # Right side for predicted values
            img_comp[:, w_half + 2:] = predicted[:, w_half:]

            colored = cm.viridis(img_comp)
            img = (colored[:, :, :3] * 255).astype(np.uint8)

            # Resizes image
            scale = 6
            img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
            h_img, w_img, _ = img.shape

            # Black separator
            sep_x = (w_half + 1) * scale
            img[:, sep_x - 5:sep_x + 5] = 0

            # Expected text
            cv2.putText(img, "Expected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Predicted text
            text = "Predicted"
            (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(img, text, (w_img - text_w - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        img = np.ascontiguousarray(img)

        h, w_img, ch = img.shape
        bytes_per_line = ch * w_img

        qimg = QImage(img.data, w_img, h, bytes_per_line, QImage.Format_RGB888)

        self.predicted_label.setPixmap(QPixmap.fromImage(qimg))

        self.current_frame += 1

    def create_video_box(self, title):
        """
        Creates a box in which a video will be displayed.

        :param title: title of the video
        """

        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)

        layout = QVBoxLayout()

        label_title = QLabel(title)
        label_title.setAlignment(Qt.AlignCenter)

        video_label = QLabel("[Video will appear here]")
        video_label.setAlignment(Qt.AlignCenter)
        video_label.setMinimumSize(350, 350)
        video_label.setScaledContents(True)

        layout.addWidget(label_title, stretch=1)
        layout.addWidget(video_label, stretch=14)

        frame.setLayout(layout)

        return frame, video_label

    def on_model_selected(self):
        """
        If the selected model is valid, sets a flag to notify the prediction button.
        """
        self.model_selected = self.model_selector.currentIndex() != 0
        self.update_predict_button()

    def update_predict_button(self):
        """
        Enables the prediction button if a valid sample is loaded and a model is selected.
        """
        self.predict_button.setEnabled(self.data_loaded and self.model_selected)

    def update_metrics_plot(self, metrics, sample):
        """
        Generates and displays the geometric metrics plot aligned with sample size
        and using physical time range from dataset.
        """
        # How many initial steps are used in the prediction
        offset = len(metrics["d_horizontal_test"]) - sample.shape[0]
        offset = max(offset, 0)

        fig_met = plt.figure(figsize=(20, 6), facecolor="white")
        gs_met = gridspec.GridSpec(1, 3, figure=fig_met)
        plt.subplots_adjust(wspace=0.25)

        metric_keys = [("Horizontal Diameter", "d_horizontal_test", "d_horizontal_pred"),
                        ("Vertical Diameter", "d_vertical_test", "d_vertical_pred"),
                        ("Center of Mass (y)", "c_mass_test", "c_mass_pred")]
        for i, (label, k_gt, k_pred) in enumerate(metric_keys):
            ax = fig_met.add_subplot(gs_met[0, i])

            # Skips initial context offset
            y_gt = metrics[k_gt][offset:]
            y_pred = metrics[k_pred][offset:]

            # Physical time axis
            full_n = len(metrics["d_horizontal_test"])

            # Adjusts initial time
            tmin, tmax = self.data["t_range"]
            t_start = tmin + (offset / full_n) * (tmax - tmin)

            x = np.linspace(t_start, tmax, len(y_gt))
            x_dense = np.linspace(t_start, tmax, 300)

            # Smoothing the curves
            spline_gt = UnivariateSpline(x, y_gt, s=5)
            spline_pred = UnivariateSpline(x, y_pred, s=5)

            ax.plot(x_dense, spline_gt(x_dense), color="black", lw=3, label="Expected")
            ax.plot(x_dense, spline_pred(x_dense), "--", color="#1f77b4", lw=2, label="Predicted")
            ax.set_title(label, fontsize=18)
            ax.set_xlabel("Time", fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.grid(True, alpha=0.3, linestyle="--")
            if i == 0:
                ax.legend(fontsize=14)

        # Render to Qt QLabel
        fig_met.suptitle("Physical Metrics Comparison", fontsize=18, y=1.05)

        canvas = FigureCanvasAgg(fig_met)
        canvas.draw()

        buf = np.asarray(canvas.buffer_rgba())
        buf = np.ascontiguousarray(buf)

        h, w, ch = buf.shape
        bytes_per_line = ch * w

        qimg = QImage(buf.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)

        self.plot_label.setPixmap(pixmap.scaled(self.plot_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        plt.close(fig_met)

    def update_log(self, metrics):
        """
        Updates the log panel with geometric and error metrics comparing
        predicted and reference simulation data.

        :param metrics: dictionary containing geometric metrics and error metrics
        """
        dif_d_horizontal = abs(metrics["d_horizontal_pred"] - metrics["d_horizontal_test"])
        dif_d_vertical = abs(metrics["d_vertical_pred"] - metrics["d_vertical_test"])
        dif_c_mass = abs(metrics["c_mass_pred"] - metrics["c_mass_test"])
        dif_t_contact = abs(metrics["t_contact_pred"] - metrics["t_contact_test"])

        self.text_box.clear()
        self.text_box.append(f"Maximum horizontal diameter (test): {max(metrics["d_horizontal_test"]):.4f}")
        self.text_box.append(f"Maximum horizontal diameter (model): {max(metrics["d_horizontal_pred"]):.4f}")
        self.text_box.append(f"Difference in maximum horizontal diameter: {max(dif_d_horizontal):.4f}")

        self.text_box.append(f"\nMaximum vertical diameter (test): {max(metrics["d_vertical_test"]):.4f}")
        self.text_box.append(f"Maximum vertical diameter (model): {max(metrics["d_vertical_pred"]):.4f}")
        self.text_box.append(f"Difference in maximum vertical diameter: {max(dif_d_vertical):.4f}")

        self.text_box.append(f"\nMaximum center of mass (test): {max(metrics["c_mass_test"]):.4f}")
        self.text_box.append(f"Maximum center of mass (model): {max(metrics["c_mass_pred"]):.4f}")
        self.text_box.append(f"Difference in center of mass: {max(dif_c_mass):.4f}")

        self.text_box.append(f"\nContact time (test): {metrics["t_contact_test"]:.4f}")
        self.text_box.append(f"Contact time (model): {metrics["t_contact_pred"]:.4f}")
        self.text_box.append(f"Difference in contact time: {dif_t_contact:.4f}")

        self.text_box.append(f"\nMean absolute error of horizontal diameter: {np.mean(dif_d_horizontal):.4f}")
        self.text_box.append(f"Mean absolute error of vertical diameter: {np.mean(dif_d_vertical):.4f}")
        self.text_box.append(f"Mean absolute error of center of mass: {np.mean(dif_c_mass):.4f}")
        self.text_box.append(f"Mean absolute error of contact time: {np.mean(dif_t_contact):.4f}")

        self.text_box.append(f"\nR2-score: {metrics["r2s"]:.4f}")
        self.text_box.append(f"RMSE: {metrics["rmse"]:.4f}")
        self.text_box.append(f"SSIM: {metrics["ssim"]:.4f}")

    def run_prediction(self):
        """
        Runs the prediction pipeline using the selected model and updates the UI with results.
        """
        self.loading_label.setVisible(True)
        QApplication.processEvents()

        path_architecture = self.params["architectures"]
        path_model = os.path.join(os.getcwd(), self.params["models"][self.model_selector.currentText()])
        self.predicted, self.metrics = self.VOFClass.predict(path_architecture, path_model)

        self.loading_label.setVisible(False)

        # Generates the prediction video
        self.expected_frames = self.data["sample"]
        self.predicted_frames = self.predicted

        self.context_steps = (len(self.expected_frames) - len(self.predicted_frames))

        self.current_frame = 0

        self.timer_prediction = QTimer()
        self.timer_prediction.timeout.connect(self.update_prediction_video)
        self.timer_prediction.start(30)

        # Generates the geometric metrics plot
        self.update_metrics_plot(self.metrics, self.predicted)

        # Writes the metrics to the log
        self.update_log(self.metrics)

        # Enables save button
        self.save_button.setEnabled(True)
    
    def compare_models(self):
        """
        Performs data prediction using each of the available models and opens a new window to display the evaluation metrics for each result.
        """
        results = {}

        # Generates the results using each of the models
        path_architecture = self.params["architectures"]
        for model_name, path_model in self.params["models"].items():
            full_path = os.path.join(os.getcwd(), path_model)
            sample, metrics = self.VOFClass.predict(path_architecture, full_path)
            results[model_name] = {"sample": sample, "metrics": metrics, "t_range": self.data["t_range"]}

        # Opens a new window to display the model comparisons
        self.compare_window = ComparePredictionsWindow(results, self.frames)
        self.compare_window.showMaximized()

    def save_results(self):
        """
        Saves prediction results and metrics into a .npz file.
        """
        file_path, _ = QFileDialog.getSaveFileName(self, "Save results", "", "NPZ Files (*.npz)")

        if not file_path:
            return

        if not file_path.endswith(".npz"):
            file_path += ".npz"

        try:
            results = {
                "predicted": self.predicted,
                "d_horizontal_test": self.metrics["d_horizontal_test"],
                "d_horizontal_pred": self.metrics["d_horizontal_pred"],
                "d_vertical_test": self.metrics["d_vertical_test"],
                "d_vertical_pred": self.metrics["d_vertical_pred"],
                "c_mass_test": self.metrics["c_mass_test"],
                "c_mass_pred": self.metrics["c_mass_pred"],
                "t_contact_test": self.metrics["t_contact_test"],
                "t_contact_pred": self.metrics["t_contact_pred"],
                "r2s": self.metrics["r2s"],
                "rmse": self.metrics["rmse"],
                "ssim": self.metrics["ssim"]
            }
            np.savez(file_path, **results)

            QMessageBox.information(self, "Success", "Results successfully saved.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results:\n{e}")

"""Comparison between different models window"""
class ComparePredictionsWindow(QWidget):
    def __init__(self, results, expected_frames):
        """
        Creates a window to compare predictions from different models, displaying
        synchronized videos, geometric metric plots and quantitative evaluation logs.

        :param results: dictionary containing prediction samples, metrics and time ranges
                        for each evaluated model
        :param expected_frames: reference simulation frames used as expected values
        """
        super().__init__()

        self.results = results
        self.expected_frames = expected_frames

        self.setWindowTitle("Model comparison")

        self.current_frame = 0

        main_layout = QVBoxLayout()

        # Videos
        video_layout = QHBoxLayout()

        self.video_labels = {}
        for model_name, data in self.results.items():
            column = QVBoxLayout()

            # Video title
            title = QLabel(model_name)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("""font-size: 16px;
                                   font-weight: bold;""")
            column.addWidget(title)

            # Video
            video = QLabel()
            video.setMinimumSize(350, 350)
            video.setScaledContents(True)
            column.addWidget(video)

            self.video_labels[model_name] = video

            video_layout.addLayout(column)

        main_layout.addLayout(video_layout, stretch=3)

        # Geometric properties plot
        self.figure = Figure(figsize=(18, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.ax1 = self.figure.add_subplot(131)
        self.ax2 = self.figure.add_subplot(132)
        self.ax3 = self.figure.add_subplot(133)

        main_layout.addWidget(self.canvas, stretch=2)

        # Logs for quantitative metrics
        logs_layout = QHBoxLayout()
        for model_name, data in self.results.items():
            metrics = data["metrics"]

            dif_d_horizontal = abs(metrics["d_horizontal_pred"] - metrics["d_horizontal_test"])
            dif_d_vertical = abs(metrics["d_vertical_pred"] - metrics["d_vertical_test"])
            dif_c_mass = abs(metrics["c_mass_pred"] - metrics["c_mass_test"])
            dif_t_contact = abs(metrics["t_contact_pred"] - metrics["t_contact_test"])

            text_box = QTextEdit()
            text_box.setReadOnly(True)
            text_box.setMinimumHeight(350)
            text_box.append(f"Model: {model_name}\n")
            text_box.append(f"Maximum horizontal diameter (test): {max(metrics['d_horizontal_test']):.4f}")
            text_box.append(f"Maximum horizontal diameter (model): {max(metrics['d_horizontal_pred']):.4f}")
            text_box.append(f"Difference in maximum horizontal diameter: {max(dif_d_horizontal):.4f}")

            text_box.append(f"\nMaximum vertical diameter (test): {max(metrics['d_vertical_test']):.4f}")
            text_box.append(f"Maximum vertical diameter (model): {max(metrics['d_vertical_pred']):.4f}")
            text_box.append(f"Difference in maximum vertical diameter: {max(dif_d_vertical):.4f}")

            text_box.append(f"\nMaximum center of mass (test): {max(metrics['c_mass_test']):.4f}")
            text_box.append(f"Maximum center of mass (model): {max(metrics['c_mass_pred']):.4f}")
            text_box.append(f"Difference in center of mass: {max(dif_c_mass):.4f}")

            text_box.append(f"\nContact time (test): {metrics['t_contact_test']:.4f}")
            text_box.append(f"Contact time (model): {metrics['t_contact_pred']:.4f}")
            text_box.append(f"Difference in contact time: {dif_t_contact:.4f}")

            text_box.append(f"\nMean absolute error of horizontal diameter: {np.mean(dif_d_horizontal):.4f}")
            text_box.append(f"Mean absolute error of vertical diameter: {np.mean(dif_d_vertical):.4f}")
            text_box.append(f"Mean absolute error of center of mass: {np.mean(dif_c_mass):.4f}")
            text_box.append(f"Mean absolute error of contact time: {np.mean(dif_t_contact):.4f}")

            text_box.append(f"\nR2-score: {metrics['r2s']:.4f}")
            text_box.append(f"RMSE: {metrics['rmse']:.4f}")
            text_box.append(f"SSIM: {metrics['ssim']:.4f}")

            logs_layout.addWidget(text_box)

        main_layout.addLayout(logs_layout, stretch=2)

        self.setLayout(main_layout)

        self.generate_plots()

        # Timer for videos
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_videos)
        self.timer.start(30)

    def generate_plots(self):
        """
        Generates comparison plots for geometric metrics between expected data
        and predictions from all evaluated models.
        """
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        colors = ["#1f77b4", "#d62728", "#2ca02c"]

        metric_info = [("Horizontal Diameter", "d_horizontal_test", "d_horizontal_pred", self.ax1),
                       ("Vertical Diameter", "d_vertical_test", "d_vertical_pred", self.ax2),
                       ("Center of Mass (y)", "c_mass_test", "c_mass_pred", self.ax3)]
        for i, (title, k_gt, k_pred, ax) in enumerate(metric_info):
            # Expected values
            first_metrics = list(self.results.values())[0]["metrics"]

            y_gt_full = first_metrics[k_gt]

            full_n = len(y_gt_full)

            tmin, tmax = self.results[list(self.results.keys())[0]]["t_range"]

            # Predicted values from each model
            for color, (model_name, data) in zip(colors, self.results.items()):
                metrics = data["metrics"]

                # How many initial steps were used as context
                offset = len(metrics[k_gt]) - len(data["sample"])
                offset = max(offset, 0)

                # Skip initial context
                y_gt = metrics[k_gt][offset:]
                y_pred = metrics[k_pred][offset:]

                # Adjust physical initial time
                t_start = tmin + (offset / full_n) * (tmax - tmin)

                x = np.linspace(t_start, tmax, len(y_gt))
                x_dense = np.linspace(t_start, tmax, 400)

                # Smoothing the curves
                spline_gt = UnivariateSpline(x, y_gt, s=5)
                spline_pred = UnivariateSpline(x, y_pred, s=5)

                # Plot expected only once
                if model_name == list(self.results.keys())[0]:
                    ax.plot(x_dense, spline_gt(x_dense), color="black", lw=3, label="Expected")

                # Plot prediction
                ax.plot(x_dense, spline_pred(x_dense), lw=2, linestyle="--", color=color, label=model_name)

            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Time", fontsize=13)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(axis="both", labelsize=11)
            if i == 0:
                ax.legend(fontsize=10)

        self.figure.tight_layout()
        self.canvas.draw()

    def update_videos(self):
        """
        Updates all synchronized comparison videos, displaying the initial simulation context followed by side-by-side 
        comparisons between expected and predicted volume fraction fields.
        """
        # Loops back to the beginning when the video ends
        if self.current_frame >= len(self.expected_frames):
            self.current_frame = 0

        # Current timestep
        t = self.current_frame
        for model_name, data in self.results.items():
            pred_frames = data["sample"]

            # Number of context frames before prediction starts
            context_steps = len(self.expected_frames) - len(pred_frames)

            expected = self.expected_frames[t]

            h, w = expected.shape

            w_half = w // 2

            # Initial simulation context
            if t < context_steps:

                frame = expected.copy()

                colored = cm.viridis(frame)

                img = (colored[:, :, :3] * 255).astype(np.uint8)

                # Resizes image
                scale = 5

                img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

                h_img, w_img, _ = img.shape

                # Centered text
                text = "Initial simulation context"

                (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.putText(img, text, ((w_img - text_w) // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Predicted values
            else:
                pred_idx = t - context_steps

                predicted = pred_frames[pred_idx]

                # Comparison image
                img_comp = np.zeros((h, w + 2))

                # Left side for expected values
                img_comp[:, :w_half] = expected[:, :w_half]

                # Right side for predicted values
                img_comp[:, w_half + 2:] = predicted[:, w_half:]

                colored = cm.viridis(img_comp)

                img = (colored[:, :, :3] * 255).astype(np.uint8)

                # Resizes image
                scale = 5

                img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

                h_img, w_img, _ = img.shape

                # Black separator
                sep_x = (w_half + 1) * scale

                img[:, sep_x - 5:sep_x + 5] = 0

                # Expected text
                cv2.putText(img, "Expected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # Predicted text
                text = "Predicted"
                (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.putText(img, text, (w_img - text_w - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            img = np.ascontiguousarray(img)

            h2, w2, ch = img.shape

            qimg = QImage(img.data, w2, h2, ch * w2, QImage.Format_RGB888)

            self.video_labels[model_name].setPixmap(QPixmap.fromImage(qimg))

        self.current_frame += 1