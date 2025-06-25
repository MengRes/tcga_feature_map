import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import random
import matplotlib.pyplot as plt
import h5py
import openslide
from tqdm import tqdm

# ===================== 参数设置 =====================
# 可根据需要修改以下参数
accept_age_groups = ["60-69", "70-79", "80-89"]  # 只接受这些年龄段
accept_sex = ["female"]                  # 只接受这些性别
accept_race = ["white"]                  # 只接受这些种族
accept_label = ["IDC", "ILC"]            # 需要均分的label
accept_hosp_list = ["A2", "AR", "D8"]   # 只接受这些hosp source
n_per_hosp = 10                          # 每个hosp选取的WSI数量
num_sampled_patches = 100                  # 每个WSI最多采样多少patch
patch_size = 256

# 路径配置
coord_dir = "/raid/mengliang/wsi_process/tcga-brca_patch/patches/"
wsi_dir = "/home/mxz3935/dataset_folder/tcga-brca/"
output_dir = "tcga-brca_selected_patch"

# 读取标签数据
df_label = pd.read_csv("files/tcga-brca_label.csv")
df_label["source"] = df_label["filename"].str.extract(r"TCGA-([A-Z0-9]{2})-")

def age_group(age):
    try:
        age = int(age)
        return f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
    except:
        return "unknown"
df_label["age_group"] = df_label["age"].apply(age_group)

# ===================== 多条件筛选 =====================
mask = (
    df_label["age_group"].isin(accept_age_groups) &
    df_label["gender"].isin(accept_sex) &
    df_label["race"].isin(accept_race) &
    df_label["label"].isin(accept_label) &
    df_label["source"].isin(accept_hosp_list)
)
df_filtered = df_label[mask].copy()

print(f"\n=== 数据筛选结果 ===")
print(f"筛选条件:")
print(f"  年龄段: {accept_age_groups}")
print(f"  性别: {accept_sex}")
print(f"  种族: {accept_race}")
print(f"  Label: {accept_label}")
print(f"  Hosp sources: {accept_hosp_list}")
print(f"筛选后数据量: {len(df_filtered)} 个WSI")

# ===================== 按hosp分组并均衡采样 =====================
print(f"\n=== 开始按hosp均衡采样 ===")
print(f"每个hosp目标采样数: {n_per_hosp}")
print(f"需要均分的label: {accept_label}")

selected_rows = []
total_selected = 0

for hosp in accept_hosp_list:
    print(f"\n--- 处理hosp: {hosp} ---")
    hosp_data = df_filtered[df_filtered["source"] == hosp].copy()
    
    if len(hosp_data) == 0:
        print(f"  ✗ {hosp}: 无可用数据")
        continue
    
    # 统计每个label的可用数量
    label_counts = hosp_data["label"].value_counts()
    print(f"  {hosp} 可用数据分布:")
    for label, count in label_counts.items():
        print(f"    {label}: {count} 个")
    
    # 计算每个label的目标数量（尽量均分）
    n_labels = len(accept_label)
    target_per_label = n_per_hosp // n_labels
    remainder = n_per_hosp % n_labels
    
    print(f"  目标分配: 每个label {target_per_label} 个，剩余 {remainder} 个")
    
    hosp_selected = []
    
    # 对每个label进行采样
    for i, label in enumerate(accept_label):
        label_data = hosp_data[hosp_data["label"] == label].copy()
        available_count = len(label_data)
        
        # 计算当前label的目标数量
        current_target = target_per_label
        if i < remainder:  # 前几个label多分配一个
            current_target += 1
        
        # 实际采样数量
        actual_count = min(available_count, current_target)
        
        print(f"    {label}: 可用{available_count}个，目标{current_target}个，实际采样{actual_count}个")
        
        if actual_count > 0:
            if actual_count == available_count:
                hosp_selected.append(label_data)
            else:
                hosp_selected.append(label_data.sample(n=actual_count, random_state=42))
    
    # 合并当前hosp的所有采样结果
    if hosp_selected:
        hosp_df = pd.concat(hosp_selected, ignore_index=True)
        selected_rows.append(hosp_df)
        total_selected += len(hosp_df)
        print(f"  ✓ {hosp}: 成功采样 {len(hosp_df)} 个WSI")
    else:
        print(f"  ✗ {hosp}: 无有效采样")

print(f"\n=== 采样完成 ===")
print(f"总计采样: {total_selected} 个WSI")

if not selected_rows:
    print("No matched WSI sets found with the given criteria.")
    exit()

df_selected = pd.concat(selected_rows, ignore_index=True)

# 统计每个source选择的WSI数量
source_counts = df_selected["source"].value_counts()
print(f"\n=== 各source选择的WSI数量 ===")
for source, count in source_counts.items():
    print(f"{source}: {count} 个WSI")

# 统计每个label的分布
label_counts = df_selected["label"].value_counts()
print(f"\n=== 各label的分布 ===")
for label, count in label_counts.items():
    print(f"{label}: {count} 个WSI")

# 统计每个hosp内label的分布
print(f"\n=== 各hosp内label分布 ===")
for hosp in accept_hosp_list:
    hosp_data = df_selected[df_selected["source"] == hosp].copy()
    if len(hosp_data) > 0:
        hosp_label_counts = hosp_data["label"].value_counts()
        print(f"{hosp}: {dict(hosp_label_counts)}")

df_selected["coord_path"] = df_selected["filename"].apply(lambda f: os.path.join(coord_dir, f"{f}.h5"))
df_selected["wsi_path"] = df_selected["filename"].apply(lambda f: os.path.join(wsi_dir, f"{f}.svs"))

output_csv = "files/tcga-brca_selected_sources.csv"
df_selected.to_csv(output_csv, index=False)
print(f"已保存: {output_csv}")

# ===================== Patch采样与保存 =====================
print(f"\n=== 开始提取patches ===")
print(f"每个WSI最多采样: {num_sampled_patches} 个patches")
print(f"Patch大小: {patch_size}x{patch_size}")

# 清空输出目录
if os.path.exists(output_dir):
    import shutil
    shutil.rmtree(output_dir)
    print(f"已清空输出目录: {output_dir}")
os.makedirs(output_dir, exist_ok=True)
print(f"已创建输出目录: {output_dir}")

df = pd.read_csv("files/tcga-brca_selected_sources.csv")
patch_rows = []
successful_wsi = 0
failed_wsi = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting patches"):
    filename = str(row["filename"])
    coord_path = str(row["coord_path"])
    wsi_path = str(row["wsi_path"])
    slide_id = os.path.splitext(filename)[0]
    slide_output_dir = os.path.join(output_dir, slide_id)
    os.makedirs(slide_output_dir, exist_ok=True)
    
    if not os.path.isfile(coord_path) or not os.path.isfile(wsi_path):
        print(f"[Skip] Missing file for {slide_id}")
        failed_wsi += 1
        continue
        
    try:
        with h5py.File(coord_path, "r") as f:
            if "coords" in f:
                coords_dataset = f["coords"]
                if isinstance(coords_dataset, h5py.Dataset):
                    coords = np.array(coords_dataset[:])
                else:
                    print(f"[Skip] coords is not a dataset in {slide_id}")
                    failed_wsi += 1
                    continue
            else:
                print(f"[Skip] No coords dataset found in {slide_id}")
                failed_wsi += 1
                continue
    except Exception as e:
        print(f"[Skip] Error reading coords for {slide_id}: {e}")
        failed_wsi += 1
        continue
        
    if len(coords) == 0:
        print(f"[Skip] No coords in {slide_id}")
        failed_wsi += 1
        continue
        
    coords_list = coords.tolist() if hasattr(coords, 'tolist') else list(coords)
    sampled_coords = coords_list if len(coords_list) <= num_sampled_patches else random.sample(coords_list, num_sampled_patches)
    
    try:
        slide = openslide.OpenSlide(wsi_path)
    except Exception as e:
        print(f"[Skip] Error opening slide for {slide_id}: {e}")
        failed_wsi += 1
        continue
        
    successful_patches = 0
    for coord in sampled_coords:
        try:
            x, y = map(int, coord)
            patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            patch_np = np.array(patch)
            patch_name = f"{x}_{y}.npy"
            npy_path = os.path.join(slide_output_dir, patch_name)
            np.save(npy_path, patch_np)
            patch_row = row.to_dict()
            patch_row["patch_x"] = x
            patch_row["patch_y"] = y
            patch_row["npy_path"] = npy_path
            patch_rows.append(patch_row)
            successful_patches += 1
        except Exception as e:
            print(f"[Skip] Error processing patch {coord} for {slide_id}: {e}")
            continue
    slide.close()
    
    successful_wsi += 1
    print(f"  {slide_id}: 成功提取 {successful_patches}/{len(sampled_coords)} 个patches")

patch_df = pd.DataFrame(patch_rows)
patch_df.to_csv("files/tcga-brca_selected_patches.csv", index=False)

print(f"\n=== Patch提取完成 ===")
print(f"成功处理WSI: {successful_wsi} 个")
print(f"失败WSI: {failed_wsi} 个")
print(f"总计提取patches: {len(patch_df)} 个")
print("Saved patch_df to files/tcga-brca_selected_patches.csv") 