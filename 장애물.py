from ultralytics import YOLO
import cv2
import numpy as np

# 1. 학습한 모델 로드
model = YOLO("자동차장애물.pt")

# 2. 동영상 로드
cap = cv2.VideoCapture(0)

# 3. 실시간 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. YOLO 추론
    results = model(frame, conf=0.3)

    # 5. 결과 시각화
    annotated_frame = results[0].plot()

    # 6. car 클래스 감지 여부 확인
    car_detected = False
    for cls_id in results[0].boxes.cls.tolist():
        label = model.names[int(cls_id)]
        if label == "car":
            car_detected = True
            break

    # 7. 감지되면 화면에 메시지 표시
    if car_detected:
        cv2.putText(
            annotated_frame,
            "CAR DETECTED",
            org=(30, 60),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.8,
            color=(0, 255, 255),
            thickness=3,
            lineType=cv2.LINE_AA,
        )

    # 8. 화면 출력
    cv2.imshow("Car Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. 자원 해제
cap.release()
cv2.destroyAllWindows()
