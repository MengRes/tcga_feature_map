# Gender Analysis Scripts

This folder contains scripts for gender-based feature extraction and t-SNE visualization on TCGA-LUAD dataset.

## Scripts Overview

### 1. wsi_patch_filter_luad.py
- **Purpose**: Filter TCGA-LUAD WSIs and patches by gender, hospital, age, and race, and export patch metadata for downstream feature extraction and visualization.
- **Function**: Ensures balanced selection of WSIs by gender and hospital, and outputs filtered patch and WSI metadata CSVs for later use.

### 2. uni_feature_tsne_gender.py
- **Purpose**: Extract features using UNI model and create t-SNE visualization colored by gender
- **Model**: UNI/UNI2-H from MahmoodLab
- **Features**: 1536-dimensional features for UNI2-H, 1024-dimensional for UNI

### 3. resnet50_feature_tsne_gender.py  
- **Purpose**: Extract features using ResNet50 model and create t-SNE visualization colored by gender
- **Model**: Pretrained ResNet50 (ImageNet weights)
- **Features**: 2048-dimensional features

### 4. conch_feature_tsne_gender.py
- **Purpose**: Extract features using CONCH model and create t-SNE visualization colored by gender
- **Model**: CONCH vision-language model
- **Features**: Model-dependent feature dimensions

## Usage

All scripts follow the same command-line interface pattern:

```bash
python gender_analysis/[script_name].py \
    --input_dir results/wsi_patch_filter_luad_YYYYMMDD_HHMMSS \
    --wsi_dir /path/to/wsi/images \
    [model_specific_parameters]
```

### Example Commands

#### Patch Filtering:
```bash
python gender_analysis/wsi_patch_filter_luad.py \
    --wsi_dir /path/to/tcga/luad/svs/files \
    --meta_csv /path/to/TCGA-LUAD.metadata.tsv \
    --output_dir results/wsi_patch_filter_luad_YYYYMMDD_HHMMSS
```

#### UNI Model:
```bash
python gender_analysis/uni_feature_tsne_gender.py \
    --input_dir results/wsi_patch_filter_luad_20250701_231340 \
    --wsi_dir /path/to/tcga/luad/svs/files \
    --uni_model_name uni2-h
```

#### ResNet50 Model:
```bash
python gender_analysis/resnet50_feature_tsne_gender.py \
    --input_dir results/wsi_patch_filter_luad_20250701_231340 \
    --wsi_dir /path/to/tcga/luad/svs/files
```

#### CONCH Model:
```bash
python gender_analysis/conch_feature_tsne_gender.py \
    --input_dir results/wsi_patch_filter_luad_20250701_231340 \
    --wsi_dir /path/to/tcga/luad/svs/files \
    --conch_ckpt ./checkpoints/conch/pytorch_model.bin \
    --conch_cfg conch_ViT-B-16
```

## Input Requirements

- **Patch CSV**: `conch_filtered_patches.csv` from WSI patch filtering
- **WSI Metadata**: `selected_wsis.csv` containing gender information
- **WSI Images**: SVS files in the specified WSI directory

## Output Structure

Each script generates:
- `features_[model_name]/`: Extracted features for each WSI
- `visualizations/`: t-SNE plots and results
  - `tsne_by_gender_[model_name].png`: t-SNE visualization
  - `tsne_gender_result_[model_name].csv`: t-SNE coordinates and metadata

## Dependencies

- torch, torchvision
- timm (for UNI models)
- openslide-python
- scikit-learn
- pandas, numpy
- PIL, matplotlib
- tqdm

## Notes

- All scripts use English comments for internationalization
- Features are extracted from patches filtered by CONCH zero-shot classification
- Gender information is derived from TCGA metadata
- t-SNE parameters can be customized via command-line arguments 