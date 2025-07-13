import cv2
from ultralytics import YOLO
import time

# 1. 모델 불러오기
model = YOLO(r"C:\Users\geonh\Desktop\중앙대학교\2학년\미래제품연구회 자율주행차량 경진대회\AFB\automateddrving\best.pt")  # Roboflow에서 학습한 pt 파일

# 2. 비디오 로드
cap = cv2.VideoCapture(r"C:\Users\geonh\Desktop\중앙대학교\2학년\미래제품연구회 자율주행차량 경진대회\AFB\automateddrving\video\dog.mp4")
if not cap.isOpened():
    print("Video load failed")
    exit()

# 3. 이전 중심점 추적용 변수
prev_center_x = None
direction_text = ""
frame_display_count = 0  # 텍스트 유지용 카운터

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. YOLO 추론
    results = model(frame, verbose=False)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if label.lower() == "dog":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 중심점 이동량 기반 방향 판단
                if prev_center_x is not None:
                    dx = cx - prev_center_x
                    if dx > 15:
                        direction_text = "Avoid to the right"
                        frame_display_count = 60  # 약 2초 (30fps 기준)
                    elif dx < -15:
                        direction_text = "Avoid to the left"
                        frame_display_count = 60
                prev_center_x = cx

                # bounding box 및 중심점 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # 텍스트 시각화 (2초 동안 유지)
    if frame_display_count > 0:
        cv2.putText(frame, direction_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        frame_display_count -= 1

    # 5. 결과 출력
    cv2.imshow("dog obstacle avoidance", frame)
    
    if cv2.waitKey(1) == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
