from ultralytics import YOLO
import cv2
import numpy as np

# 1. YOLO 모델 불러오기 (직접 학습한 모델 또는 사전 학습된 모델)
model = YOLO("dog.pt")  # 경로 설정

# 2. 동영상 로드
cap = cv2.VideoCapture(r"video\dog.mp4")  # 동영상 경로

# 3. 실시간 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. YOLO 추론 (BGR 이미지를 바로 넣을 수 있음)
    results = model(frame, conf=0.3)  # confidence threshold 설정 가능

    # 5. 결과를 프레임 위에 그리기
    annotated_frame = results[0].plot()  # 클래스 이름, 박스, conf 포함된 시각화된 이미지

    # 6. OpenCV 창에 표시
    cv2.imshow("Vehicle Detection", annotated_frame)

    # 종료 키: q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()