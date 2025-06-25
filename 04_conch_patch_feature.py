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
filtered_csv_path = "files/tcga-brca_patch_conch_filtered.csv"  # 筛选后的patch信息
save_dir = "tcga-brca_filtered_features"  # 保存筛选后特征的目录
os.makedirs(save_dir, exist_ok=True)

# === 读取筛选后的patch信息 ===
print("读取筛选后的patches...")
df_filtered = pd.read_csv(filtered_csv_path)
print(f"筛选后的patch数量: {len(df_filtered)}")

# === 按 WSI 分组 ===
grouped = df_filtered.groupby("filename")
print(f"WSI数量: {len(grouped)}")

# === 遍历每个 WSI ===
for wsi_id, group in tqdm(grouped, desc="Processing WSI patches"):
    print(f"\n处理WSI: {wsi_id}, patch数量: {len(group)}")
    
    all_feats = []
    all_coords = []
    all_patch_info = []

    for _, row in group.iterrows():
        npy_path = str(row["npy_path"])  # 确保是字符串
        x, y = int(row["patch_x"]), int(row["patch_y"])
        patch_label = str(row["patch_label"])  # 确保是字符串
        patch_probability = float(row["patch_probability"])  # 确保是浮点数

        if not os.path.exists(npy_path):
            print(f"[Warning] Patch文件不存在: {npy_path}")
            continue

        try:
            img_array = np.load(npy_path)
            img = Image.fromarray(img_array.astype(np.uint8))

            # 特征提取
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.inference_mode():
                feat = model.encode_image(img_tensor)

            all_feats.append(feat)
            all_coords.append([x, y])
            all_patch_info.append({
                'patch_x': x,
                'patch_y': y,
                'patch_label': patch_label,
                'patch_probability': patch_probability,
                'npy_path': npy_path
            })
            
        except Exception as e:
            print(f"[Error] 处理patch失败 {npy_path}: {e}")
            continue

    # 保存为 .pt
    if all_feats:
        features = torch.cat(all_feats, dim=0)
        coords = torch.tensor(all_coords)
        
        # 创建patch信息DataFrame
        patch_info_df = pd.DataFrame(all_patch_info)

        save_path = os.path.join(save_dir, f"{wsi_id}_features.pt")
        torch.save({
            "features": features,
            "coords": coords,
            "wsi_id": wsi_id,
            "patch_info": patch_info_df
        }, save_path)
        
        # 保存patch信息为CSV
        csv_save_path = os.path.join(save_dir, f"{wsi_id}_patch_info.csv")
        patch_info_df.to_csv(csv_save_path, index=False)
        
        print(f"[Saved] {save_path} - 特征维度: {features.shape}, patch数量: {len(all_feats)}")
    else:
        print(f"[Warning] WSI {wsi_id} 没有有效的patches")

print(f"\n特征提取完成！结果保存在: {save_dir}")
print(f"处理的WSI数量: {len(grouped)}")