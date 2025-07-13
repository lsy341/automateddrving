"""
Siberian Husky YOLO 데이터셋
밝기(brightness 0.5~1.5배) 증강 5장씩 생성 + 라벨(txt) 복사
╰─ 입력 :  <base>/train/images/*.jpg  /train/labels/*.txt
╰─ 출력 : <base>/aug/images/         /aug/labels/
"""

import os, shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
import numpy as np

# ★★★ 1) 데이터셋 압축 해제 후 여기에 맞춰 주세요 ★★★
BASE = r"C:\makingprogram\AFB\증강"   # = 압축을 푼 최상위 폴더

IN_IMG  = os.path.join(BASE, "train/images")
IN_LBL  = os.path.join(BASE, "train/labels")
OUT_IMG = os.path.join(BASE, "aug/images")
OUT_LBL = os.path.join(BASE, "aug/labels")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

# 밝기만 조절하는 증강기
datagen = ImageDataGenerator(brightness_range=[0.5, 1.5])

for img_name in sorted(os.listdir(IN_IMG)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    stem       = os.path.splitext(img_name)[0]          # 예: husky_0123
    img_path   = os.path.join(IN_IMG,  f"{stem}.jpg")
    label_path = os.path.join(IN_LBL, f"{stem}.txt")
    if not os.path.exists(label_path):
        print(f"[WARN] 라벨 없음: {stem}.txt → 스킵")
        continue

    # 이미지 로드 후 (1,H,W,C) 형태로 변환
    x = img_to_array(load_img(img_path)).reshape((1,)+tuple(img_to_array(load_img(img_path)).shape))

    # 5장 증강
    for i, batch in enumerate(datagen.flow(x, batch_size=1, shuffle=False)):
        aug_img = Image.fromarray(batch[0].astype(np.uint8))
        out_img_name = f"bright_{stem}_{i}.jpg"
        out_lbl_name = f"bright_{stem}_{i}.txt"

        aug_img.save(os.path.join(OUT_IMG, out_img_name))
        shutil.copy(label_path, os.path.join(OUT_LBL, out_lbl_name))

        if i == 4:       # 이미지당 5장
            break

print("✅ 증강 완료!")
