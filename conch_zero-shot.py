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

checkpoint_path = './checkpoints/conch/pytorch_model.bin'
model_cfg = 'conch_ViT-B-16'

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

labels = []
failed_indices = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying patches"):
    npy_path = row['npy_path']
    if not os.path.isfile(npy_path):
        print(f"[Warning] Patch file not found: {npy_path}")
        labels.append("Missing")
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
        labels.append(label_map.get(pred_label, pred_label))

    except Exception as e:
        print(f"[Error] Failed to process {npy_path}: {e}")
        labels.append("Error")
        failed_indices.append(idx)

# ==== 添加分类结果列 ====
df['patch_label'] = labels

# ==== 保存到新CSV ====
df.to_csv(output_csv, index=False)
print(f"Saved labeled patches info to: {output_csv}")