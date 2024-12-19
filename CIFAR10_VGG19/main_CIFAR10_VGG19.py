import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import torch
import q5  # 匯入 q5.py


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MainWindow")
        self.setGeometry(100, 100, 800, 400)  # 增加窗口寬度以容納圖片

        # 主視窗
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 主佈局
        self.main_layout = QHBoxLayout()

        # 左側按鈕區域佈局
        self.button_layout = QVBoxLayout()

        # 按鈕
        self.load_image_button = QPushButton("Load Image")
        self.augmented_images_button = QPushButton("1. Show Augmented Images")
        self.model_structure_button = QPushButton("2. Show Model Structure")
        self.accuracy_loss_button = QPushButton("3. Show Accuracy and Loss")
        self.inference_button = QPushButton("4. Inference")

        # 加入按鈕到左側布局
        self.button_layout.addWidget(self.load_image_button)
        self.button_layout.addWidget(self.augmented_images_button)
        self.button_layout.addWidget(self.model_structure_button)
        self.button_layout.addWidget(self.accuracy_loss_button)
        self.button_layout.addWidget(self.inference_button)

        # 右側圖片與推論結果區域
        self.image_layout = QVBoxLayout()

        # 圖像顯示區域
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(300, 300)  # 固定圖片顯示大小
        self.image_layout.addWidget(self.image_label)

        # 推論結果
        self.prediction_label = QLabel("Predicted = None")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.image_layout.addWidget(self.prediction_label)

        # 將左側和右側佈局加入到主佈局
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(self.image_layout)

        # 設定主佈局
        self.central_widget.setLayout(self.main_layout)

        # 綁定按鈕事件
        self.load_image_button.clicked.connect(self.load_image)
        self.augmented_images_button.clicked.connect(self.show_augmented_images)
        self.model_structure_button.clicked.connect(self.show_model_structure)
        self.accuracy_loss_button.clicked.connect(self.show_accuracy_loss)
        self.inference_button.clicked.connect(self.run_inference)

        # 初始化變數
        self.image_path = None  # 用於保存加載圖片的路徑
        self.model = torch.load("./models/trained_model_20241219-154917.pth", map_location=torch.device('cpu'))

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.image_path = file_path  # 保存圖片路徑
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))

    def show_augmented_images(self):
        q5.showLabeledArgImg(self)

    def show_model_structure(self):
        q5.showModelStructure()

    def show_accuracy_loss(self):
        q5.showModelAccLoss(self)

    def run_inference(self):
        if self.image_path:  # 確保已經加載圖片
            q5.inference(self.image_path, self.model, self.prediction_label)
        else:
            self.prediction_label.setText("No image loaded!")  # 提示未加載圖片


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
