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

def get_center_points_by_nearest(left_pts, right_pts, y_samples = list(range(465, 339, -12)), max_diff=5):
    center_points = []

    for y in y_samples:
        left_pt = get_nearest_point(left_pts, y, max_diff)
        right_pt = get_nearest_point(right_pts, y, max_diff)
        print(f"left_pt = {left_pt}")
        print(f"right_pt = {right_pt}")
        if left_pt and right_pt:
            center_x = (left_pt[0] + right_pt[0]) // 2
            center_points.append((center_x, y))
    print("===========================")
    return center_points

# 이미지 폴더 경로 설정
image_folder = 'captures'
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))])
total_images = len(image_files)

def process_image(index):
    file_path = os.path.join(image_folder, image_files[index])
    img = cv2.imread(file_path)
    img = cv2.resize(img, (640, 480))

    roi_y1, roi_y2 = 330, 465
    roi = img[roi_y1:roi_y2, :]
    cv2.rectangle(img, (0, roi_y1), (640, roi_y2), (0, 255, 0), 2)

    lower_white = np.array([151, 166, 164])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(roi, lower_white, upper_white)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 150)
    
    # 애매한 흰색들 255로 설정
    edges = cv2.resize(edges, (640, 480), interpolation=cv2.INTER_NEAREST)
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
        if len(left_points) >= 80 and len(right_points) >= 80:
            break
        
    all_points = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for pt in left_points:
        cv2.circle(all_points, tuple(pt), 3, (255, 255, 0), -1)
    for pt in right_points:
        cv2.circle(all_points, tuple(pt), 3, (255, 0, 255), -1)
    
    if len(left_points) >= 80 and len(right_points) >= 80:
        left_points = filter_outliers(left_points, axis='x', threshold=50)
        # print(f"left_points = {len(left_points)}")
        right_points = filter_outliers(right_points, axis='x', threshold=50)
        # print(f"right_points = {len(right_points)}")

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
            center_points = get_center_points_by_nearest(left_points, right_points)
            for pt in center_points:
                cv2.circle(color_roi, pt, 3, (0, 255, 0), -1)  # 초록 점

            for i in range(len(center_points) - 1):
                cv2.line(color_roi, center_points[i], center_points[i + 1], (0, 255, 0), 2)  # 초록 선

    return img, color_roi, all_points, edges

def on_trackbar(val):
    img, color_roi, all_points, edges = process_image(val)
    combined = cv2.hconcat([img, cv2.resize(color_roi, (640, 480))])
    cv2.imshow('Image Viewer', combined)
    cv2.imshow('all points', all_points)
    cv2.imshow('edges', edges)

cv2.namedWindow('Image Viewer')
cv2.createTrackbar('Index', 'Image Viewer', 0, total_images - 1, on_trackbar)

# 초기 이미지 표시
on_trackbar(0)

# ESC 종료
while True:
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
