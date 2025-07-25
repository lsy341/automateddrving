import cv2
import numpy as np
import os
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

def get_center_points_by_nearest(left_pts, right_pts, max_diff=5):
    center_points = []
    l_y = min(left_pts, key=lambda pt: pt[1])
    r_y = min(right_pts, key=lambda pt: pt[1])
    min_y = min(l_y[1], r_y[1])
    
    y_samples = list(range(150, min_y, -((150 - min_y)//5)))
    
    for y in y_samples:
        left_pt = get_nearest_point(left_pts, y, max_diff)
        right_pt = get_nearest_point(right_pts, y, max_diff)
        print(f"left_pt = {left_pt}")
        print(f"right_pt = {right_pt}")
        if left_pt and right_pt:
            center_x = (left_pt[0] + right_pt[0]) // 2
            center_points.append((center_x, y))
    print(min_y)
    print("===========================")
    return center_points

def is_curve(points, threshold=0.0009):
    # 주어진 (x, y) 점들로부터 곡선 여부를 판단합니다.
    # 2차항 계수 a의 절댓값이 threshold 이상이면 곡선으로 간주합니다.
    if len(points) < 3:
        return False  # 점이 너무 적으면 곡선 판단 불가

    x_pts = [x for x, y in points]
    y_pts = [y for x, y in points]

    try:
        coeffs = np.polyfit(y_pts, x_pts, 2)  # x = ay² + by + c
        a = coeffs[0]
        print(a)
        return abs(a) > threshold
    except Exception:
        return False

def process_frame(img):
    img = cv2.resize(img, (640, 480))

    roi_y1, roi_y2 = 240, 390
    roi = img[roi_y1:roi_y2, :]
    cv2.rectangle(img, (0, roi_y1), (640, roi_y2), (0, 255, 0), 2)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([85, 40, 150])       # 전체 hue, 낮은 채도, 밝은 명도
    upper_white = np.array([105, 255, 255])
    mask = cv2.inRange(hsv_roi, lower_white, upper_white)

    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 150)
    
    # 애매한 흰색들 255로 설정
    _, edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)  # 완전한 이진화
    color_roi = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    left_points, right_points = [], []
    h, w = edges.shape
    mid = w // 2

    # ROI에서 y는 아래에서 위로 탐색
    for y in range(h - 1, -1, -2):
        row = edges[y]
        
        # 왼쪽 안쪽 차선 탐색
        for x in range(mid - 1, -1, -1):
            if row[x] == 255:
                left_points.append((x, y))
                break
            
        # 오른쪽 안쪽 차선 탐색
        for x in range(mid + 1, w):
            if row[x] == 255:
                right_points.append((x, y))
                break
        if len(left_points) >= 30 and len(right_points) >= 30:
            break
    print(len(left_points), len(right_points))    
    all_points = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for pt in left_points:
        cv2.circle(all_points, tuple(pt), 3, (255, 255, 0), -1)
    for pt in right_points:
        cv2.circle(all_points, tuple(pt), 3, (255, 0, 255), -1)
        
    # # 곡선 여부에 따라 그릴지 결정
    # if is_curve(left_points):
    #     print("왼쪽 차선은 곡선입니다.")
    # else:
    #     print("왼쪽 차선은 곡선이 아닙니다.")

    # if is_curve(right_points):
    #     print("오른쪽 차선은 곡선입니다.")
    # else:
    #     print("오른쪽 차선은 곡선이 아닙니다.")
        
    
    if len(left_points) >= 5 and len(right_points) >= 5:
        all_left_points = left_points
        all_right_points = right_points
        left_points = filter_outliers(left_points, axis='x', threshold=30)
        # print(f"left_points = {len(left_points)}")
        right_points = filter_outliers(right_points, axis='x', threshold=30)
        # print(f"right_points = {len(right_points)}")
        # print(right_points)
        for pt in left_points:
            cv2.circle(color_roi, tuple(pt), 3, (255, 255, 0), -1)
        for pt in right_points:
            cv2.circle(color_roi, tuple(pt), 3, (255, 0, 255), -1)

        if len(left_points) >= 2 and len(right_points) >= 2:
            left_poly_points = left_points[:10]
            right_poly_points = right_points[:10]
            
            for pt in left_poly_points:
                cv2.circle(color_roi, pt, 5, (0, 255, 0), -1)
            for pt in right_poly_points:
                cv2.circle(color_roi, pt, 5, (0, 255, 0), -1)
                
            left_line = get_tangent(left_poly_points, 0, 500)
            right_line = get_tangent(right_poly_points, 20, 639)

            cv2.line(color_roi, left_line[0], left_line[1], (0, 255, 255), 2)
            cv2.line(color_roi, right_line[0], right_line[1], (0, 0, 255), 2)

            p_left = np.array(left_line[0])
            v_left = np.array(left_line[1]) - p_left
            p_right = np.array(right_line[0])
            v_right = np.array(right_line[1]) - p_right
            intersection = get_intersection(p_left, v_left, p_right, v_right)

            center = (w // 2, 240)
            cv2.circle(color_roi, center, 7, (255, 0, 0), -1)

            if intersection:
                # 교점 존재 시 시각화
                cv2.circle(color_roi, intersection, 7, (0, 255, 0), -1)
                cv2.line(color_roi, center, intersection, (255, 255, 255), 2)

                # 중심 → 수직 위쪽으로 기준선 그리기
                top_of_vertical = (center[0], center[1] - 80)
                cv2.line(color_roi, center, top_of_vertical, (0, 255, 255), 1)

                # 교점 방향 벡터 계산
                vec_to_inter = np.array(intersection) - np.array(center)
                vertical_vec = np.array([0, -1])
                angle = get_angle(vec_to_inter, vertical_vec)
                adjusted_angle = 90 + angle if vec_to_inter[0] > 0 else 90 - angle
                adjusted_angle = np.clip(adjusted_angle, 0, 180)

                cv2.putText(color_roi, f"Angle: {adjusted_angle:.2f} deg", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 중심선 점 + 선 시각화 추가
            center_points = get_center_points_by_nearest(all_left_points, all_right_points)
            center_points = filter_outliers(center_points, 'x', threshold=10)
            for pt in center_points:
                cv2.circle(color_roi, pt, 3, (0, 255, 0), -1)  # 초록 점

            for i in range(len(center_points) - 1):
                cv2.line(color_roi, center_points[i], center_points[i + 1], (0, 255, 0), 2)  # 초록 선

    return img, color_roi, all_points, edges


# 비디오 경로 설정
video_path = r"C:\makingprogram\AFB\automateddrving\vod_test.mp4"  # 여기에 실험할 비디오 파일 경로 입력
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Video", frame)

    img, color_roi, all_points, edges = process_frame(frame)

    cv2.imshow('original img', img)
    cv2.imshow('color roi', color_roi)
    cv2.imshow('all points', all_points)
    cv2.imshow('edges', edges)
    
    key = cv2.waitKey(30) & 0xFF

    
    if key == ord('q'):
        break
    elif key == ord(' '):  # 스페이스바로 일시정지/재생 토글
        paused = not paused
cap.release()
cv2.destroyAllWindows()
