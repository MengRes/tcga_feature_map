import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from CONCH.conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

# === 初始化模型 ===
model_cfg = 'conch_ViT-B-16'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_path = './checkpoints/conch/pytorch_model.bin'
model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)
model.eval()

# === 配置路径 ===
csv_path = "files/tcga-brca_patch_conch.csv"  # 包含 label 与 patch_label 的完整表格
save_dir = "tcga-brca_selected_patch"
os.makedirs(save_dir, exist_ok=True)

# === 读取并筛选 df ===
df = pd.read_csv(csv_path)
df_filtered = df[df["label"] == df["patch_label"]].copy()
print(f"匹配 label 的 patch 数量: {len(df_filtered)}")

# === 按 WSI 分组 ===
grouped = df_filtered.groupby("filename")

# === 遍历每个 WSI ===
for wsi_id, group in tqdm(grouped, desc="Processing matched patches"):

    all_feats = []
    all_coords = []

    for _, row in group.iterrows():
        npy_path = row["npy_path"]
        x, y = int(row["patch_x"]), int(row["patch_y"])

        if not os.path.exists(npy_path):
            continue

        img_array = np.load(npy_path)
        img = Image.fromarray(img_array)

        # 特征提取
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            feat = model.encode_image(img_tensor)

        all_feats.append(feat)
        all_coords.append([x, y])

    # 保存为 .pt
    if all_feats:
        features = torch.cat(all_feats, dim=0)
        coords = torch.tensor(all_coords)

        save_path = os.path.join(save_dir, f"{wsi_id}_filtered_features.pt")
        torch.save({
            "features": features,
            "coords": coords,
            "wsi_id": wsi_id
        }, save_path)
        print(f"[Saved] {save_path}")