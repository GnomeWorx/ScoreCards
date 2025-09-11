import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QRadioButton, QGroupBox, QLabel, QTextEdit, QFileDialog
)
from PyQt5.QtGui import QPalette, QColor, QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Import the scoring engine
from scoring_engine.score import run_scoring_pipeline

# Dark Mode Stylesheet (same as before)
DARK_STYLESHEET = """
QWidget {
    background-color: #2b2b2b;
    color: #ffffff;
    font-size: 14px;
}
QMainWindow {
    background-color: #2b2b2b;
}
QPushButton {
    background-color: #555555;
    border: 1px solid #777777;
    padding: 5px;
    border-radius: 3px;
}
QPushButton:hover {
    background-color: #666666;
}
QPushButton:pressed {
    background-color: #444444;
}
QGroupBox {
    border: 1px solid #777777;
    border-radius: 5px;
    margin-top: 1ex;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 3px;
}
QLabel {
    border: 1px solid #444444;
    padding: 5px;
}
QTextEdit {
    background-color: #333333;
    border: 1px solid #777777;
}
"""

class Worker(QThread):
    """
    Worker thread for running the CV pipeline without freezing the GUI.
    """
    # Signal that emits the results dictionary when finished
    finished = pyqtSignal(dict)

    def __init__(self, image_path, calibre):
        super().__init__()
        self.image_path = image_path
        self.calibre = calibre

    def run(self):
        """Run the scoring pipeline and emit the results."""
        if self.image_path:
            results = run_scoring_pipeline(self.image_path, self.calibre)
            self.finished.emit(results)


class MainWindow(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NSRA Target Scoring Application")
        self.setGeometry(100, 100, 1000, 700) # Increased size

        # --- Central Widget and Layouts ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        main_layout.addLayout(left_layout, 3) # Image display takes more space
        main_layout.addLayout(right_layout, 1)

        # --- Left side: Image Display ---
        self.image_label = QLabel("Please open an image file to begin.")
        self.image_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.image_label)

        # --- Right side: Controls and Results ---
        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image_dialog)
        right_layout.addWidget(self.open_button)

        calibre_box = QGroupBox("Calibre")
        calibre_layout = QVBoxLayout()
        self.radio_177 = QRadioButton(".177 (4.5mm)")
        self.radio_22 = QRadioButton(".22 (5.6mm)")
        self.radio_177.setChecked(True)
        calibre_layout.addWidget(self.radio_177)
        calibre_layout.addWidget(self.radio_22)
        calibre_box.setLayout(calibre_layout)
        right_layout.addWidget(calibre_box)

        results_box = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit("Results will be shown here.")
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        results_box.setLayout(results_layout)
        right_layout.addWidget(results_box)

        right_layout.addStretch()

    def open_image_dialog(self):
        """Opens a file dialog to select an image and starts the scoring process."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Target Image", "",
                                                  "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            calibre = 0.177 if self.radio_177.isChecked() else 0.22
            self.image_label.setText("Processing...")
            self.open_button.setEnabled(False) # Disable button during processing

            # Create and start the worker thread
            self.worker = Worker(image_path=file_path, calibre=calibre)
            self.worker.finished.connect(self.update_results)
            self.worker.start()

    def update_results(self, results):
        """Receives the results from the worker thread and updates the GUI."""
        if results and results['annotated_image'] is not None:
            # Format results text
            total_score = results['total_score']
            individual_scores = sorted(results['individual_scores'], reverse=True)
            results_str = f"Total Score: {total_score}\n\n"
            results_str += f"Individual Scores ({len(individual_scores)} shots):\n"
            results_str += ", ".join(map(str, individual_scores))
            self.results_text.setText(results_str)

            # Convert OpenCV image (BGR) to QPixmap (RGB)
            h, w, ch = results['annotated_image'].shape
            bytes_per_line = ch * w
            qt_image = QImage(results['annotated_image'].data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qt_image)

            # Display the pixmap, scaling it to fit the label
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.results_text.setText("Scoring failed. Please check the console output.")
            self.image_label.setText("Could not process image.")

        self.open_button.setEnabled(True) # Re-enable button


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
