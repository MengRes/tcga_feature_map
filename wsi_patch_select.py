import os
import numpy as np
import pandas as pd
from collections import Counter
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import openslide
from tqdm import tqdm

import random
random.seed(42)
random_seed = 42


coord_dir = "/raid/mengliang/wsi_process/tcga-brca_patch/patches/"
wsi_dir = "/home/mxz3935/dataset_folder/tcga-brca/"
output_dir = "tcga-brca_selected_patch"
patch_size = 256
num_sampled_patches = 100



df_label = pd.read_csv("files/tcga-brca_label.csv")
df_label["source"] = df_label["filename"].str.extract(r"TCGA-([A-Z0-9]{2})-")

selected_source = ['A2', 'AR', 'D8']

df_selected = df_label[df_label["source"].isin(selected_source)]

samples_per_source = 5
selected_rows = []

for source in selected_source:
    group = df_label[df_label["source"] == source]
    if len(group) <= samples_per_source:
        selected_rows.append(group)
    else:
        selected_rows.append(group.sample(n=samples_per_source, random_state=42))

# 合并选中的样本
df_selected = pd.concat(selected_rows).reset_index(drop=True)

# 添加 coord_path 和 wsi_path 列
df_selected["coord_path"] = df_selected["filename"].apply(lambda f: os.path.join(coord_dir, f"{f}.h5"))
df_selected["wsi_path"] = df_selected["filename"].apply(lambda f: os.path.join(wsi_dir, f"{f}.svs"))

# 保存为新的 CSV 文件
output_csv = "files/tcga-brca_selected_sources.csv"
df_selected.to_csv(output_csv, index=False)

print(f"已保存: {output_csv}")

df = pd.read_csv("files/tcga-brca_selected_sources.csv")
# 保存每个 patch 记录
patch_rows = []

# 遍历每一条 WSI 记录
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting patches"):
    filename, coord_path, wsi_path = row["filename"], row["coord_path"], row["wsi_path"]
    slide_id = os.path.splitext(filename)[0]
    slide_output_dir = os.path.join(output_dir, slide_id)
    os.makedirs(slide_output_dir, exist_ok=True)

    if not os.path.isfile(coord_path) or not os.path.isfile(wsi_path):
        print(f"[Skip] Missing file for {slide_id}")
        continue

    with h5py.File(coord_path, "r") as f:
        coords = f["coords"][:]

    if len(coords) == 0:
        print(f"[Skip] No coords in {slide_id}")
        continue

    sampled_coords = coords if len(coords) <= num_sampled_patches else random.sample(list(coords), num_sampled_patches)

    slide = openslide.OpenSlide(wsi_path)

    for coord in sampled_coords:
        x, y = map(int, coord)
        patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
        patch_np = np.array(patch)

        patch_name = f"{x}_{y}.npy"
        npy_path = os.path.join(slide_output_dir, patch_name)
        np.save(npy_path, patch_np)

        # 构建当前 patch 的信息行，包含原始 WSI 行的所有列
        patch_row = row.to_dict()
        patch_row["patch_x"] = x
        patch_row["patch_y"] = y
        patch_row["npy_path"] = npy_path
        patch_rows.append(patch_row)

# 构建 DataFrame：每 patch 一行，包含原始列 + patch_x/y + npy_path
patch_df = pd.DataFrame(patch_rows)

# 保存
patch_df.to_csv("files/tcga-brca_selected_patches.csv", index=False)
print("Saved patch_df to files/tcga-brca_selected_patches.csv")