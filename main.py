import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QCheckBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from PIL import Image
from PIL.ImageQt import ImageQt  # Import ImageQt module
import app.image_processing.functions as ip  # Your image processing module

class ImageTransformer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Transformer')
        self.setGeometry(100, 100, 800, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.image_layout = QVBoxLayout()

        self.original_label = QLabel()
        self.transformed_label = QLabel()

        self.image_layout.addWidget(self.original_label)
        self.image_layout.addWidget(self.transformed_label)

        open_button = QPushButton('Open Image')
        open_button.clicked.connect(self.open_image)
        transform_button = QPushButton('Transform Image')
        transform_button.clicked.connect(self.transform_image)

        self.t0_checkbox = QCheckBox("T0 Image")  # Add the checkbox
        self.t0_checkbox.setChecked(True)  # Set the initial state of the checkbox

        self.layout.addLayout(self.image_layout)
        self.layout.addWidget(open_button)
        self.layout.addWidget(transform_button)
        self.layout.addWidget(self.t0_checkbox)  # Add the checkbox

        self.original_image = None
        self.original_image_path = None

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)', options=options)

        if file_path:
            image = QImage(file_path)

            if not image.isNull():
                pixmap = QPixmap.fromImage(image)
                self.original_label.setPixmap(pixmap)
                self.original_image = Image.open(file_path)
                self.original_image_path = file_path

    def transform_image(self):
        if self.original_image:
            transform_t0 = self.t0_checkbox.isChecked()  # Check the state of the checkbox
            transformed_image = Image.fromarray(ip.test_process(self.original_image_path, transform_t0)).convert("L")
            image = ImageQt(transformed_image)  # Use ImageQt to convert PIL image to QImage
            pixmap = QPixmap.fromImage(image)
            self.transformed_label.setPixmap(pixmap)

def main():
    app = QApplication(sys.argv)
    window = ImageTransformer()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
