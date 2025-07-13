import cv2
import numpy as np

# 두 벡터 사잇각 계산 (0~180도)
def get_angle(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)

# 추세선 계산
def get_tangent(points, x1, x2):
    x_pts = [x for x, y in points]
    y_pts = [y for x, y in points]
    coeffs = np.polyfit(x_pts, y_pts, 1)
    a, b = coeffs
    return [(x1, int(a * x1 + b)), (x2, int(a * x2 + b))]

# 이상치 제거
def filter_outliers(points, axis='x', threshold=50):
    if not points:
        return []
    coords = np.array(points)
    center = np.median(coords[:, 0] if axis == 'x' else coords[:, 1])
    diffs = np.abs((coords[:, 0] if axis == 'x' else coords[:, 1]) - center)
    return coords[diffs < threshold].tolist()

# 교점 계산
def get_intersection(p1, v1, p2, v2):
    A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    if np.linalg.matrix_rank(A) < 2:
        return None
    t = np.linalg.solve(A, b)
    intersection = p1 + t[0] * v1
    return tuple(intersection.astype(int))

# 동영상 캡처 시작 (카메라: 0, 또는 파일 경로)
cap = cv2.VideoCapture(r'vod_test.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30  # ms 단위

prev_angle = None
angle_threshold = 20  # 프레임 간 허용 각도 변화 범위 (단위: 도)


while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (640, 480))
    height, width, _ = img.shape

    # ROI 설정
    roi_y1, roi_y2 = 200, 450
    roi = img[roi_y1:roi_y2, :]
    
    cv2.rectangle(img, (0, roi_y1), (640, roi_y2), (0, 255, 0), 2)

    # HSV 색상 마스크
    lower_white = np.array([129, 120, 60])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(roi, lower_white, upper_white)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 150)

    color_roi = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 트랙 포인트 추출
    left_points, right_points = [], []
    h, w = edges.shape
    mid = w // 2

    for y in range(h - 50, -1, -2):
        row = edges[y]
        for x in range(mid - 1, -1, -1):
            if row[x] == 255:
                left_points.append((x, y))
                break
        for x in range(mid + 1, w):
            if row[x] == 255:
                right_points.append((x, y))
                break
        if len(left_points) >= 20 and len(right_points) >= 20:
            break
    # 추세선 및 교점 시각화
    if len(left_points) >= 20 and len(right_points) >= 20:
        left_points = filter_outliers(left_points, axis='x', threshold=10)
        right_points = filter_outliers(right_points, axis='x', threshold=10)

        for pt in left_points:
            cv2.circle(color_roi, tuple(pt), 3, (255, 255, 0), -1) # 연노랑점
            cv2.circle(img, (pt[0], pt[1] + roi_y1), 3, (255, 255, 0), -1)  # 연노랑점 (원본 위치 보정)
        for pt in right_points:
            cv2.circle(color_roi, tuple(pt), 3, (255, 0, 255), -1) # 분홍점
            cv2.circle(img, (pt[0], pt[1] + roi_y1), 3, (255, 0, 255), -1)  # 분홍점 (원본 위치 보정)

        if len(left_points) >= 2 and len(right_points) >= 2:
            left_line = get_tangent(left_points, 0, 500)
            right_line = get_tangent(right_points, 20, 639)

            # 추세선 시각화
            # ROI 좌표에서는 그대로 사용
            cv2.line(color_roi, left_line[0], left_line[1], (0, 255, 255), 2)
            cv2.line(color_roi, right_line[0], right_line[1], (0, 0, 255), 2)

            # 원본 좌표 보정해서 사용 (y좌표에 roi_y1 더함)
            left_line_img = [(left_line[0][0], left_line[0][1] + roi_y1), (left_line[1][0], left_line[1][1] + roi_y1)]
            right_line_img = [(right_line[0][0], right_line[0][1] + roi_y1), (right_line[1][0], right_line[1][1] + roi_y1)]

            cv2.line(img, left_line_img[0], left_line_img[1], (0, 255, 255), 2)
            cv2.line(img, right_line_img[0], right_line_img[1], (0, 0, 255), 2)


            # 교점 계산
            p_left = np.array(left_line[0])
            v_left = np.array(left_line[1]) - p_left
            p_right = np.array(right_line[0])
            v_right = np.array(right_line[1]) - p_right
            intersection = get_intersection(p_left, v_left, p_right, v_right)
            
            # 중심점
            center = (w // 2, 150)
            # ROI 이미지에는 그대로
            cv2.circle(color_roi, center, 7, (255, 0, 0), -1)

            # 원본 이미지에 그릴 때는 Y 좌표 보정
            center_img = (center[0], center[1] + roi_y1)
            cv2.circle(img, center_img, 7, (255, 0, 0), -1)

            if intersection:
                # ROI 이미지 기준
                cv2.circle(color_roi, intersection, 7, (0, 255, 0), -1)
                cv2.line(color_roi, center, intersection, (255, 255, 255), 2)

                top_of_vertical = (center[0], center[1] - 80)
                cv2.line(color_roi, center, top_of_vertical, (0, 255, 255), 1)

                # 원본 이미지 기준으로 좌표 보정
                intersection_img = (intersection[0], intersection[1] + roi_y1)
                center_img = (center[0], center[1] + roi_y1)
                top_of_vertical_img = (top_of_vertical[0], top_of_vertical[1] + roi_y1)

                cv2.circle(img, intersection_img, 7, (0, 255, 0), -1)
                cv2.line(img, center_img, intersection_img, (255, 255, 255), 2)
                cv2.line(img, center_img, top_of_vertical_img, (0, 255, 255), 1)


                vec_to_inter = np.array(intersection) - np.array(center)
                vertical_vec = np.array([0, -1])
                angle = get_angle(vec_to_inter, vertical_vec)
                adjusted_angle = 90 + angle if vec_to_inter[0] > 0 else 90 - angle
                adjusted_angle = np.clip(adjusted_angle, 0, 180)
                
                if prev_angle is None or abs(adjusted_angle - prev_angle) < angle_threshold:
                    prev_angle = adjusted_angle  # 업데이트
                    cv2.putText(color_roi, f"Angle: {adjusted_angle:.2f} deg", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    # 너무 튀는 값이면 무시하고 이전 값 유지
                    cv2.putText(color_roi, f"Angle: {prev_angle:.2f} deg (stable)", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)

                cv2.putText(color_roi, f"Angle: {adjusted_angle:.2f} deg", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow('Track Detection (Video)', color_roi)
    cv2.imshow('Original video', img)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
