from ultralytics import YOLO
import cv2, time, collections
# ─────────────────────────────────────────
# 설정값
MODEL_PATH   = "dog.pt"
VIDEO_PATH   = r"video/dog.mp4"
WIN          = "dog obstacle avoidance"
CONF_THRESH  = 0.3          # 탐지 신뢰도 임계값
MOVE_THRESH  = 10           # 이동 감지 픽셀
MSG_DURATION = 2.0          # 문구 유지 시간(s)
HIST_FRAMES  = 10           # 중심좌표 평균 프레임 수
FONT         = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────────────────
# ① 모델·비디오·창 초기화
model = YOLO(MODEL_PATH) #pt파일로부터 객체 생성
cap   = cv2.VideoCapture(VIDEO_PATH) #비디오 읽는 객체
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL) #
# 원하는 고정 크기가 있다면 ↓ 해제
# cv2.resizeWindow(WIN, 1280, 720)

center_hist = collections.deque(maxlen=HIST_FRAMES) #collection 모듈 안의 deque 클라스, 칸은 10칸
msg, msg_ts = "", 0.0 #메세지와 메세지 출력 시간 저장

while cap.isOpened(): #캡처 객체가 열려 있다면 다음 실행
    ok, frame = cap.read()
    if not ok:
        break

    # ② 추론
    results = model(frame, conf=CONF_THRESH) 
    boxes   = results[0].boxes #객체model의 반환값, 리스트 형태로 저장, boxes는 속성

    annotated = frame.copy()          # 원본 해상도 유지

    # ③ dog 탐지 있으면 0번만 사용 (단일 클래스 가정)
    if boxes: #탐지된 객체가 있다면 실행
        # 가장 confidence 높은 박스를 사용, 가장 개에 가까운
        top_idx  = int(boxes.conf.argmax())
        box      = boxes[top_idx]
        cls_id   = int(box.cls[0]) #미리 정해진 클래스 아이디가 있음. 라벨링 할 때 정해짐 개는 0
        conf     = float(box.conf[0]) #신뢰도 값 실수 형태로 가져옴

        if cls_id == 0:               # dog 클래스
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            center_hist.append(cx)

            # 바운딩 박스 & 라벨+conf
            label = f"dog {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10), FONT, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)

            # ④ 이동 방향 계산
            if len(center_hist) >= 2:
                dx = center_hist[-1] - center_hist[0] #deque에 들어간 중심 좌표 나중-처음 즉 변화량
                if   dx >  MOVE_THRESH: msg, msg_ts = "Avoid to the right", time.time()
                elif dx < -MOVE_THRESH: msg, msg_ts = "Avoid to the left" , time.time()

    # ⑤ 회피 문구 2초 표시
    if msg and time.time() - msg_ts < MSG_DURATION:
        cv2.putText(annotated, msg, (30, 50), FONT, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    # ⑥ 결과 창 출력
    cv2.imshow(WIN, annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
