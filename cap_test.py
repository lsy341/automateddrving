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
    coeffs = np.polyfit(x_pts, y_pts, 1)
    a, b = coeffs
    return [(x1, int(a * x1 + b)), (x2, int(a * x2 + b))]

def filter_outliers(points, axis='x', threshold=50):
    if not points:
        return []
    coords = np.array(points)
    center = np.median(coords[:, 0] if axis == 'x' else coords[:, 1])
    diffs = np.abs((coords[:, 0] if axis == 'x' else coords[:, 1]) - center)
    return coords[diffs < threshold].tolist()

def get_intersection(p1, v1, p2, v2):
    A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    if np.linalg.matrix_rank(A) < 2:
        return None
    t = np.linalg.solve(A, b)
    intersection = p1 + t[0] * v1
    return tuple(intersection.astype(int))

# 이미지 폴더 경로 설정
image_folder = 'captures'
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))])
total_images = len(image_files)

def process_image(index):
    file_path = os.path.join(image_folder, image_files[index])
    img = cv2.imread(file_path)
    img = cv2.resize(img, (640, 480))

    roi_y1, roi_y2 = 200, 450
    roi = img[roi_y1:roi_y2, :]

    lower_white = np.array([129, 120, 60])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(roi, lower_white, upper_white)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 150)
    color_roi = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

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

    if len(left_points) >= 20 and len(right_points) >= 20:
        left_points = filter_outliers(left_points, axis='x', threshold=10)
        right_points = filter_outliers(right_points, axis='x', threshold=10)

        for pt in left_points:
            cv2.circle(color_roi, tuple(pt), 3, (255, 255, 0), -1)
        for pt in right_points:
            cv2.circle(color_roi, tuple(pt), 3, (255, 0, 255), -1)

        if len(left_points) >= 2 and len(right_points) >= 2:
            left_line = get_tangent(left_points, 0, 500)
            right_line = get_tangent(right_points, 20, 639)

            cv2.line(color_roi, left_line[0], left_line[1], (0, 255, 255), 2)
            cv2.line(color_roi, right_line[0], right_line[1], (0, 0, 255), 2)

            p_left = np.array(left_line[0])
            v_left = np.array(left_line[1]) - p_left
            p_right = np.array(right_line[0])
            v_right = np.array(right_line[1]) - p_right
            intersection = get_intersection(p_left, v_left, p_right, v_right)

            center = (w // 2, 150)
            cv2.circle(color_roi, center, 7, (255, 0, 0), -1)

            if intersection:
                cv2.circle(color_roi, intersection, 7, (0, 255, 0), -1)
                cv2.line(color_roi, center, intersection, (255, 255, 255), 2)

                top_of_vertical = (center[0], center[1] - 80)
                cv2.line(color_roi, center, top_of_vertical, (0, 255, 255), 1)

                vec_to_inter = np.array(intersection) - np.array(center)
                vertical_vec = np.array([0, -1])
                angle = get_angle(vec_to_inter, vertical_vec)
                adjusted_angle = 90 + angle if vec_to_inter[0] > 0 else 90 - angle
                adjusted_angle = np.clip(adjusted_angle, 0, 180)

                cv2.putText(color_roi, f"Angle: {adjusted_angle:.2f} deg", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img, color_roi

def on_trackbar(val):
    img, color_roi = process_image(val)
    combined = cv2.hconcat([img, cv2.resize(color_roi, (640, 480))])
    cv2.imshow('Image Viewer', combined)

cv2.namedWindow('Image Viewer')
cv2.createTrackbar('Index', 'Image Viewer', 0, total_images - 1, on_trackbar)

# 초기 이미지 표시
on_trackbar(0)

# ESC 종료
while True:
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
