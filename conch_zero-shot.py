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

# === 初始化模型 ===
model_cfg = 'conch_ViT-B-16'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_path = './checkpoints/conch/pytorch_model.bin'
model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)
_ = model.eval()

# === 定义类别文本描述 ===
tokenizer = get_tokenizer()
classes = ['invasive ductal carcinoma', 
           'invasive lobular carcinoma']
prompts = ['an H&E image of invasive ductal carcinoma', 
           'an H&E image of invasive lobular carcinoma']

tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)
print(tokenized_prompts.shape)


# === 读取 patch 图像 ===
npy_path = "tcga-brca_selected_patch/TCGA-A2-A0EM-01Z-00-DX1/14305_23395.npy"  # 替换成你的 patch 路径
patch_array = np.load(npy_path)  # shape: (H, W, 3)
patch_image = Image.fromarray(patch_array.astype(np.uint8))  # 转为 PIL.Image

image_tensor = preprocess(patch_image).unsqueeze(0).to(device)



with torch.inference_mode():
    image_embedings = model.encode_image(image_tensor)
    text_embedings = model.encode_text(tokenized_prompts)
    sim_scores = (image_embedings @ text_embedings.T * model.logit_scale.exp()).softmax(dim=-1).cpu().numpy()

print("Predicted class:", classes[sim_scores.argmax()])
print("Normalized similarity scores:", [f"{cls}: {score:.3f}" for cls, score in zip(classes, sim_scores[0])])