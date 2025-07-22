import os, shutil, cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image, ImageFilter
import numpy as np

# ★★★ 1) 경로 설정 ★★★
BASE = r"C:\makingprogram\AFB\automateddrving\증강"

IN_IMG  = os.path.join(BASE, "train/images")
IN_LBL  = os.path.join(BASE, "train/labels")
OUT_IMG = os.path.join(BASE, "aug/images")
OUT_LBL = os.path.join(BASE, "aug/labels")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

# 밝기 조절 증강기
bright_gen = ImageDataGenerator(brightness_range=[0.5, 1.5])

# 중앙 Crop 라벨 변환
def adjust_yolo_label_for_center_crop(label_lines, crop_ratio=0.8):
    new_lines = []
    crop_offset = (1 - crop_ratio) / 2  # = 0.1
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)

        # 절대 좌표에서 crop 후 상대 좌표 재계산
        new_x = (x - crop_offset) / crop_ratio
        new_y = (y - crop_offset) / crop_ratio
        new_w = w / crop_ratio
        new_h = h / crop_ratio

        # 0~1 벗어나면 이미지 밖이므로 제외
        if 0 <= new_x <= 1 and 0 <= new_y <= 1:
            new_line = f"{cls} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}"
            new_lines.append(new_line)
    return new_lines

# Crop + 라벨 조정 + 저장
def apply_center_crop_with_label(img_path, label_path, stem, num=5):
    img = Image.open(img_path)
    w, h = img.size

    left   = int(0.1 * w)
    top    = int(0.1 * h)
    right  = int(0.9 * w)
    bottom = int(0.9 * h)

    with open(label_path, 'r') as f:
        label_lines = f.readlines()

    new_labels = adjust_yolo_label_for_center_crop(label_lines)

    for i in range(num):
        cropped = img.crop((left, top, right, bottom)).resize((w, h))
        out_img_name = f"crop_{stem}_{i}.jpg"
        out_lbl_name = f"crop_{stem}_{i}.txt"

        cropped.save(os.path.join(OUT_IMG, out_img_name))
        with open(os.path.join(OUT_LBL, out_lbl_name), 'w') as f:
            f.write("\n".join(new_labels))

# Blur 증강
def apply_blur(img_path, label_path, stem, num=5):
    img = Image.open(img_path)
    for i in range(num):
        blurred = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        out_img_name = f"blur_{stem}_{i}.jpg"
        out_lbl_name = f"blur_{stem}_{i}.txt"
        blurred.save(os.path.join(OUT_IMG, out_img_name))
        shutil.copy(label_path, os.path.join(OUT_LBL, out_lbl_name))

# 밝기, crop, blur 증강 실행
for img_name in sorted(os.listdir(IN_IMG)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    stem       = os.path.splitext(img_name)[0]
    img_path   = os.path.join(IN_IMG,  f"{stem}.jpg")
    label_path = os.path.join(IN_LBL, f"{stem}.txt")
    if not os.path.exists(label_path):
        print(f"[WARN] 라벨 없음: {stem}.txt → 스킵")
        continue

    # 밝기 증강
    x = img_to_array(load_img(img_path)).reshape((1,) + img_to_array(load_img(img_path)).shape)
    for i, batch in enumerate(bright_gen.flow(x, batch_size=1, shuffle=False)):
        aug_img = Image.fromarray(batch[0].astype(np.uint8))
        out_img_name = f"bright_{stem}_{i}.jpg"
        out_lbl_name = f"bright_{stem}_{i}.txt"
        aug_img.save(os.path.join(OUT_IMG, out_img_name))
        shutil.copy(label_path, os.path.join(OUT_LBL, out_lbl_name))
        if i == 4:
            break

    # 중앙 crop + 라벨 변환
    apply_center_crop_with_label(img_path, label_path, stem)

    # 블러 증강
    apply_blur(img_path, label_path, stem)

print("✅ 밝기 + 중앙 crop(라벨 포함) + 블러 증강 완료!")
