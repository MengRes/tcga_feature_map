import os
import numpy as np
import pandas as pd
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ==== 参数配置 ====
patch_info_csv = "files/tcga-brca_selected_patches.csv"  # 你之前生成的patch info csv文件
output_csv = "files/tcga-brca_patch_conch.csv"
filtered_output_csv = "files/tcga-brca_patch_conch_filtered.csv"  # 筛选后的patch信息

checkpoint_path = './checkpoints/conch/pytorch_model.bin'
model_cfg = 'conch_ViT-B-16'
probability_threshold = 0.8  # 概率阈值

# ==== 初始化模型 ====
model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)
model.eval()

tokenizer = get_tokenizer()
classes = ['invasive ductal carcinoma', 'invasive lobular carcinoma']
prompts = ['an H&E image of invasive ductal carcinoma', 'an H&E image of invasive lobular carcinoma']
tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)

label_map = {
    'invasive ductal carcinoma': 'IDC',
    'invasive lobular carcinoma': 'ILC'
}

# ==== 读取patch信息 ====
df = pd.read_csv(patch_info_csv)
print(f"Total patches to process: {len(df)}")

labels = []
probabilities = []
failed_indices = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying patches"):
    npy_path = str(row['npy_path'])  # 确保是字符串类型
    if not os.path.isfile(npy_path):
        print(f"[Warning] Patch file not found: {npy_path}")
        labels.append("Missing")
        probabilities.append(0.0)
        failed_indices.append(idx)
        continue

    try:
        patch_array = np.load(npy_path)
        patch_image = Image.fromarray(patch_array.astype(np.uint8))
        image_tensor = preprocess(patch_image).unsqueeze(0).to(device)

        with torch.inference_mode():
            image_embeddings = model.encode_image(image_tensor)
            text_embeddings = model.encode_text(tokenized_prompts)
            sim_scores = (image_embeddings @ text_embeddings.T * model.logit_scale.exp()).softmax(dim=-1).cpu().numpy()

        pred_idx = sim_scores.argmax()
        pred_label = classes[pred_idx]
        pred_prob = float(sim_scores[0][pred_idx])  # 确保是float类型
        
        labels.append(label_map.get(pred_label, pred_label))
        probabilities.append(pred_prob)

    except Exception as e:
        print(f"[Error] Failed to process {npy_path}: {e}")
        labels.append("Error")
        probabilities.append(0.0)
        failed_indices.append(idx)

# ==== 添加分类结果列 ====
df['patch_label'] = labels
df['patch_probability'] = probabilities

# ==== 保存所有分类结果 ====
df.to_csv(output_csv, index=False)
print(f"Saved all labeled patches info to: {output_csv}")

# ==== 筛选条件：patch label 与 WSI label 一致 且 概率大于阈值 ====
print(f"\nFiltering patches with probability threshold: {probability_threshold}")
print(f"Label consistency check: patch_label == label")

# 创建筛选条件
label_consistent = df['patch_label'] == df['label']
high_probability = df['patch_probability'] >= probability_threshold
valid_patches = label_consistent & high_probability

# 筛选数据
df_filtered = df[valid_patches].copy()

print(f"Total patches: {len(df)}")
print(f"Label consistent patches: {label_consistent.sum()}")
print(f"High probability patches (>= {probability_threshold}): {high_probability.sum()}")
print(f"Filtered patches (consistent + high prob): {len(df_filtered)}")

# ==== 保存筛选后的patch信息 ====
df_filtered.to_csv(filtered_output_csv, index=False)
print(f"Saved filtered patches info to: {filtered_output_csv}")

# ==== 统计信息 ====
if len(df_filtered) > 0:
    print(f"\nFiltered patches statistics:")
    print(f"By WSI label:")
    print(df_filtered['label'].value_counts())
    print(f"\nBy patch label:")
    print(df_filtered['patch_label'].value_counts())
    print(f"\nProbability distribution:")
    print(f"Min: {df_filtered['patch_probability'].min():.3f}")
    print(f"Max: {df_filtered['patch_probability'].max():.3f}")
    print(f"Mean: {df_filtered['patch_probability'].mean():.3f}")
    print(f"Std: {df_filtered['patch_probability'].std():.3f}")
else:
    print("No patches passed the filtering criteria!")