import cv2
import numpy as np
import time
import afb  # 너가 만든 afb 패키지: camera.py, gpio.py 포함

# ----------------------------
# 초기화
# ----------------------------
afb.gpio.init()
afb.camera.init(640, 480, 30)

# ----------------------------
# 유틸리티
# ----------------------------

def get_angle(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)

def get_tangent(points, x1, x2):
    x_pts = [x for x, y in points]
    y_pts = [y for x, y in points]
    
    if max(x_pts) - min(x_pts) < 1e-6:
        # x 값이 전부 같음 → 수직선으로 처리
        x = x_pts[0]
        return [(x, 0), (x, max(y_pts))]
    
    coeffs = np.polyfit(x_pts, y_pts, 1)
    a, b = coeffs
    return [(x1, int(a * x1 + b)), (x2, int(a * x2 + b))]

def filter_outliers(points, axis='x', threshold=50):
    if not points:
        return []
    # points를 행렬로 변환
    coords = np.array(points)
    center = np.median(coords[:, 0] if axis == 'x' else coords[:, 1])
    diffs = np.abs((coords[:, 0] if axis == 'x' else coords[:, 1]) - center)
    return coords[diffs < threshold].tolist()

def get_intersection(p1, v1, p2, v2):
    # 두 개의 벡터방정식으로 교점 찾기
    A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    if np.linalg.matrix_rank(A) < 2:
        return None
    t = np.linalg.solve(A, b)
    intersection = p1 + t[0] * v1
    return tuple(intersection.astype(int))

def get_nearest_point(points, target_y, max_diff=5):
    candidates = [(x, y) for (x, y) in points if abs(y - target_y) <= max_diff]
    print(f"candidates입니다. \n{candidates}\n")
    if not candidates:
        return None
    return min(candidates, key=lambda pt: abs(pt[1] - target_y))

def get_center_points_by_nearest(left_pts, right_pts, y_samples = list(range(465, 339, -12)), max_diff=5):
    center_points = []
    return center_points

# ----------------------------
# 조향각 계산
# ----------------------------

def compute_steering_angle(frame):
    roi_y1, roi_y2 = 330, 465
    roi = frame[roi_y1:roi_y2, :]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    mask = cv2.inRange(roi_hsv, lower_white, upper_white)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 150)

    h, w = edges.shape
    mid = w // 2
    left_pts, right_pts = [], []

    for y in range(h - 1, -1, -2):
        row = edges[y]
        for x in range(mid - 1, -1, -1):
            if row[x] == 255:
                left_pts.append((x, y))
                break
        for x in range(mid + 1, w):
            if row[x] == 255:
                right_pts.append((x, y))
                break
        if len(left_pts) >= 50 and len(right_pts) >= 50:
            break

    if len(left_pts) < 2 or len(right_pts) < 2:
        return 90  # 차선 부족 → 직진 유지

    left_line = get_tangent(left_pts[:10], 0, 500)
    right_line = get_tangent(right_pts[:10], 20, 639)

    p_left = np.array(left_line[0])
    v_left = np.array(left_line[1]) - p_left
    p_right = np.array(right_line[0])
    v_right = np.array(right_line[1]) - p_right

    intersection = get_intersection(p_left, v_left, p_right, v_right)
    center = np.array([w // 2, 240])
    if intersection is not None:
        vec = intersection - center
        angle = get_angle(vec, np.array([0, -1]))
        adjusted = 90 + angle if vec[0] > 0 else 90 - angle
        return int(np.clip(adjusted, 40, 140))
    return 90


# ----------------------------
# 주행 루프
# ----------------------------
def main_loop():
    try:
        while True:
            frame = afb.camera.get_image()
            angle = compute_steering_angle(frame)
            print(f"[INFO] 조향각: {angle:.2f}°")

            afb.gpio.servo(angle)
            afb.gpio.motor(100, 1, 1)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("🛑 주행 중지")
        afb.gpio.motor(0, 1, 1)
        afb.gpio.servo(90)
    
    
# ----------------------------
# 실행
# ----------------------------
if __name__ == "__main__":
    main_loop()