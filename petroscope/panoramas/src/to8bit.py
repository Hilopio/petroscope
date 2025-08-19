import os
import numpy as np
from PIL import Image
from glob import glob

# Папка с 16-битными TIFF-изображениями
input_folder = '/home/g.nikolaev/data/MIST1'
output_folder = '/home/g.nikolaev/data/MIST1_normalized'
os.makedirs(output_folder, exist_ok=True)

# --- 1. Найти глобальный минимум и максимум ---
all_mins = []
all_maxs = []
tif_files = glob(os.path.join(input_folder, '*.tif'))

for filepath in tif_files:
    img = Image.open(filepath)
    arr = np.array(img)
    all_mins.append(arr.min())
    all_maxs.append(arr.max())

global_min = min(all_mins)
global_max = max(all_maxs)

print(f'Глобальный минимум: {global_min}, максимум: {global_max}')

# --- 2. Нормализация и сохранение ---
for filepath in tif_files:
    img = Image.open(filepath)
    arr = np.array(img)
    norm_arr = ((arr - global_min) / (global_max - global_min) * 255).clip(0, 255).astype(np.uint8)
    out_img = Image.fromarray(norm_arr)
    out_path = os.path.join(output_folder, os.path.basename(filepath))
    out_img.save(out_path)

print(f'Готово! 8-битные изображения сохранены в {output_folder}')
