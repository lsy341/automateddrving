from ultralytics import YOLO
import cv2
import numpy as np

# 1. 학습한 모델 로드
model = YOLO("stoplight.pt")      # 클래스: ['green', 'red', 'yellow'] 등

# 2. 동영상 로드
cap = cv2.VideoCapture(r"video\traffic_light.mp4")

# 3. 실시간 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. YOLO 추론 (BGR 그대로 전달)
    results = model(frame, conf=0.3)

    # 5-a. 시각화된 이미지 얻기
    annotated_frame = results[0].plot()

    # 5-b. 신호등 색 판별 → 메시지 결정
    msg = None
    for cls_id in results[0].boxes.cls.tolist():              # 클래스 id 목록
        label = model.names[int(cls_id)]                      # id → 이름
        if label == "red":
            msg = "STOP"
            break                                             # STOP이 가장 우선
        elif label == "green":
            msg = "GO"
            # 계속 찾다가 red가 있으면 덮어쓰기 방지 위해 break하지 않음

    # 5-c. 메시지가 있으면 화면에 표시
    if msg:
        cv2.putText(
            annotated_frame,
            msg,
            org=(30, 60),            # 좌표 (x, y)
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 0, 255) if msg == "STOP" else (0, 255, 0),
            thickness=4,
            lineType=cv2.LINE_AA,
        )

    # 6. 출력
    cv2.imshow("Traffic-Light Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
