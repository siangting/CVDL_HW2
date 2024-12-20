import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from inference import run_inference
from show_training_images import show_training_images
from model_structure import show_model_structure


class ImageWindow(QMainWindow):
    """新視窗用於顯示圖片"""
    def __init__(self, image_path):
        super().__init__()

        # 設定視窗標題與大小
        self.setWindowTitle("Training Loss")
        self.setGeometry(200, 200, 800, 600)

        # 主窗口設置
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 設置佈局與圖片
        layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # 加載圖片
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap.scaled(
                800, 600, Qt.KeepAspectRatio
            ))
        else:
            self.image_label.setText("Error: Unable to load the image.")

        layout.addWidget(self.image_label)
        self.central_widget.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 設定主視窗
        self.setWindowTitle("CVDL_HW2_DcGAN")
        self.setGeometry(100, 100, 300, 200)

        # 建立主視窗的中心小組框
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 建立小組框
        self.group_box = QGroupBox("Question2 DcGAN")
        self.group_box.setAlignment(Qt.AlignCenter)

        # 建立主佈局
        self.main_layout = QVBoxLayout()

        # 設定按鈕
        self.show_training_images_button = QPushButton("1. Show Training Images")
        self.show_model_structure_button = QPushButton("2. Show Model Structure")
        self.show_training_loss_button = QPushButton("3. Show Training Loss")
        self.inference_button = QPushButton("4. Inference")

        # 將按鈕加入佈局
        self.main_layout.addWidget(self.show_training_images_button)
        self.main_layout.addWidget(self.show_model_structure_button)
        self.main_layout.addWidget(self.show_training_loss_button)
        self.main_layout.addWidget(self.inference_button)

        # 將佈局加入小組框
        self.group_box.setLayout(self.main_layout)

        # 設定中心佈局
        layout = QVBoxLayout()
        layout.addWidget(self.group_box)
        self.central_widget.setLayout(layout)

        # 綁定按鈕事件
        self.show_training_images_button.clicked.connect(self.on_show_training_images_click)
        self.show_model_structure_button.clicked.connect(self.on_show_model_structure_click)
        self.show_training_loss_button.clicked.connect(self.on_show_training_loss_click)
        self.inference_button.clicked.connect(self.on_inference_click)


    def on_show_training_images_click(self):
        """按下顯示圖片按鈕時執行"""
        show_training_images(data_root="../Q2_images/data/mnist")

    def on_show_model_structure_click(self):
        """按下顯示模型結構按鈕時執行"""
        show_model_structure()

    def on_show_training_loss_click(self):
        """按下顯示訓練損失按鈕時執行"""
        image_path = "./results/training_losses_4000.jpg"
        self.image_window = ImageWindow(image_path)
        self.image_window.show()

    def on_inference_click(self):
        """按下推論按鈕時執行"""
        run_inference(generator_path="./results/generator.pth", dataset_path="../Q2_images/data/mnist")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
