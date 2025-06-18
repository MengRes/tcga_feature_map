import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from CONCH.conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
import torch
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# === 初始化模型 ===
model_cfg = 'conch_ViT-B-16'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_path = './checkpoints/conch/pytorch_model.bin'
model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)
_ = model.eval()

# 配置路径
patch_root = "tcga-brca_selected_patch"
feature_save_root = "tcga-brca_patch_feature"
os.makedirs(feature_save_root, exist_ok=True)


# 遍历每个 WSI 的子文件夹
for wsi_id in tqdm(os.listdir(patch_root), desc="WSI"):
    wsi_patch_dir = os.path.join(patch_root, wsi_id)
    if not os.path.isdir(wsi_patch_dir):
        continue

    save_path = os.path.join(feature_save_root, f"{wsi_id}_features.pt")
    if os.path.exists(save_path):
        print(f"[Skip] {wsi_id} already processed.")
        continue

    all_feats = []
    all_coords = []

    for fname in os.listdir(wsi_patch_dir):
        if not fname.endswith(".npy"):
            continue

        # 读取 patch
        npy_path = os.path.join(wsi_patch_dir, fname)
        img_array = np.load(npy_path)
        img = Image.fromarray(img_array)

        # 坐标提取
        x, y = map(int, os.path.splitext(fname)[0].split("_"))
        all_coords.append([x, y])

        # 特征提取
        img_tensor = preprocess(img).unsqueeze(0).to(device)  # [1, 3, H, W]
        with torch.inference_mode():
            feat = model.encode_image(img_tensor)
        all_feats.append(feat)

    # 聚合并保存
    if all_feats:
        features = torch.cat(all_feats, dim=0)  # [N, C]
        coords = torch.tensor(all_coords)      # [N, 2]
        torch.save({
            "features": features,
            "coords": coords,
            "wsi_id": wsi_id
        }, save_path)
        print(f"[Saved] {save_path}")