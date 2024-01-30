import cv2
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox
from PySide6.QtGui import QFont, QPainter
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from ultralytics import YOLO
import sys
from datetime import datetime
import os
import shutil

class MyWidget(QWidget):


    def __init__(self):
        super().__init__()

        title_text_size = 20
        middle_text_size = 13

        self.model = YOLO('./best.pt')
        self.CONFIDENCE_THRESHOLD = 0.7
        self.txt_directory = './detection_record'
        self. video_default_directory = './human_detection_result'

        self.flag = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video = cv2.VideoWriter()


        font_title = QFont()
        font_middle = QFont()
        font_title.setPointSize(title_text_size)
        font_title.setBold(True)
        font_middle.setPointSize(middle_text_size)
        font_middle.setBold(True)



        # 제목
        self.title = QLabel("Human Detection Program", alignment=QtCore.Qt.AlignCenter)
        self.title.setFont(font_title)

        # 오늘 날짜
        self.today_date = QLabel("시간 정보 받아오는 중...")

        # 저장경로
        self.save_dir_label = QLabel("저장 경로")
        self.save_dir_label.setFont(font_middle)
        self.save_dir = QLabel("경로가 선택되지 않았습니다.")
        self.save_dir_button = QPushButton("저장 경로 선택")

        self.previous_video_directory = ""
        self.previous_txt_directory = ""

        # 카메라 선택
        self.choose_camera_label = QLabel("카메라 선택")
        self.choose_camera_label.setFont(font_middle)


        self.webcamComboBox = QComboBox(self)
        self.webcamComboBox.currentIndexChanged.connect(self.change_webcam)

        self.webcamLabel = QLabel(self)
        self.webcamLabel.setAlignment(Qt.AlignCenter)


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.refresh_webcam_list()
        self.current_webcam_index = 0
        self.cap = cv2.VideoCapture(self.current_webcam_index)


        # 녹화 중단
        self.record_stop = QPushButton("녹화 저장 후 녹화 재실행")


        # 레이아웃 설정
        self.layout = QVBoxLayout(self)


        self.horizontal_save_dir = QHBoxLayout(self)
        self.horizontal_webcam = QHBoxLayout(self)


        self.layout.addWidget(self.title)
        self.layout.addWidget(self.today_date)
        self.layout.addWidget(self.save_dir_label)
        self.layout.addLayout(self.horizontal_save_dir)

        self.horizontal_save_dir.addWidget(self.save_dir)
        self.horizontal_save_dir.addWidget((self.save_dir_button))

        self.layout.addLayout(self.horizontal_webcam)
        self.horizontal_webcam.addWidget(self.choose_camera_label)
        self.horizontal_webcam.addWidget(self.webcamComboBox)

        self.layout.addWidget(self.webcamLabel)

        self.layout.addWidget(self.record_stop)



        self.save_dir_button.clicked.connect(self.show_file_dialog)
        self.record_stop.clicked.connect(self.record_stop_func)


    # 저장 경로 바꿨을 때 바뀐 저장 경로로 동영상 저장하는 로직 만들어야함
    def show_file_dialog(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_():
            dirName = dialog.selectedFiles()
            self.save_dir.setText(dirName[0])
            print(f"--- 저장 경로 설정 ---> {self.save_dir.text()}")


    def refresh_webcam_list(self):
        self.webcamComboBox.clear()
        for i in range(1):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.webcamComboBox.addItem(f"Webcam {i}")
                cap.release()

    def change_webcam(self, index):
        self.current_webcam_index = index
        if hasattr(self, 'cap'):
            self.cap.release()
        self.cap = cv2.VideoCapture(self.current_webcam_index)


    # 여기서 웹캠을 띄움과 동시에 모델로 object detection함
    def update_frame(self):
        
        # 시간 설정
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        detect_day_change = current_time.strftime("%H:%M:%S")
        save_time = current_time.strftime("%H_%M_%S")
        self.today_date.setText(str(formatted_time))


        if str(self.save_dir.text()) == "경로가 선택되지 않았습니다.":
            # 이거 빨간 글씨로 굵게 바꾸기
            self.record_stop.setText("저장 경로를 먼저 선택하세요")
            self.record_stop.setStyleSheet("Color : red")
            self.record_stop.setDisabled(True)
            return # 저장 경로가 설정되지 않았다면 캠 동작 x
            save_directory = self.video_default_directory
        else:
            save_directory = f'{str(self.save_dir.text())}/human_detection_result'
            self.record_stop.setText("녹화 저장 후 녹화 재실행")



        ret, frame = self.cap.read()

        txt_directory = f'{save_directory}/detection_record'
        video_directory = f'{save_directory}/videos'



        os.makedirs(f'{txt_directory}/{current_time.year}_{current_time.month}_{current_time.day}', exist_ok=True)
        os.makedirs(f'{video_directory}/{current_time.year}_{current_time.month}_{current_time.day}', exist_ok=True)
        txt_file_path = f'{txt_directory}/{current_time.year}_{current_time.month}_{current_time.day}/{current_time.year}_{current_time.month}_{current_time.day}.txt'
        video_file_save_date = f'{current_time.year}_{current_time.month}_{current_time.day}'




        if ret:
            self.record_stop.setEnabled(True)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(rgb_frame, formatted_time, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)
            cv2.putText(rgb_frame, f"{self.current_webcam_index} cam", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            if str(detect_day_change) == "00:00:00": # 날짜가 바뀐다면 동영상 촬영을 중단하고 새로운 날짜 폴더에 다시 촬영 시작.
                self.video.release()
                self.flag = 0

            if self.flag == 0:
                self.video = cv2.VideoWriter(f"{video_directory}/{video_file_save_date}/{str(current_time.year)}_{str(current_time.month)}_{str(current_time.day)}_start_{save_time}.avi",
                                             self.fourcc, 10.0, (rgb_frame.shape[1], rgb_frame.shape[0]))
                self.previous_video_directory = f"{video_directory}/{video_file_save_date}/{str(current_time.year)}_{str(current_time.month)}_{str(current_time.day)}_start_{save_time}.avi"
                self.flag = 1


            detection = self.model(rgb_frame)[0]
            # print(detection.boxes.data.tolist())
            # print(len(detection)) # 감지된 object 갯수

            if detection.boxes.data.tolist() != []: # 만약 모델이 어떤것이라도 detection 했다면
                data = detection.boxes.data.tolist()[0] # data : [xmin, ymin, xmax, ymax, confidence, class_id]
                confidence = float(data[4])
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

                    cv2.rectangle(rgb_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

                    if os.path.exists(txt_file_path):
                        self.previous_txt_directory = txt_file_path
                        with open(txt_file_path, 'a') as file:
                            file.write(f"{current_time.hour}_{current_time.minute}_{current_time.second} : {len(detection)} people detected\n")
                    else:
                        with open(txt_file_path, 'w') as file:
                            file.write(f"{current_time.hour}_{current_time.minute}_{current_time.second} : {len(detection)} people detected\n")
            else: # 하지 못했다면
                confidence = 0.0 # confidence를 선언해주지 않으면 에러가 발생해서 detection이 아무것도 안되었을 때 웹캠이 업데이트가 안됨.

            if self.flag == 1:
                save_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB) # 웹캠에서는 rgb_frame이 rgb로 잘 나오는데 동영상에 동영상에는 반전되서 다시 반전시켜서 save_frame을 사용
                self.video.write(save_frame)



            height, width, channel = rgb_frame.shape
            bytes_per_line = 3 * width

            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            self.webcamLabel.setPixmap(pixmap)
        else:
            self.record_stop.setDisabled(True)

    def record_stop_func(self):
        self.video.release()
        self.flag = 0




if __name__ == "__main__":
    app = QApplication([])

    widget = MyWidget()

    # widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec())