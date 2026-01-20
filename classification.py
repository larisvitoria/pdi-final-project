# 0. IMPORTS & ENVIRONMENT SETUP

import os
import cv2
import random
import numpy as np
import pandas as pd
import warnings
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp # FIX: Importação mais segura para compatibilidade
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    accuracy_score, recall_score, f1_score
)

from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

# System Configuration
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")


# 1. GLOBAL CONFIGURATION

class Config:
    """
    Configuration for Comparative Study of 3 Architectures.
    Strategy: Differential Learning Rate + Weighted Ensemble.
    """

    # 1.1 Paths
    CACHE_DIR = "./data"
    PROCESSED_DIR = "./processed_mass_highres_final_v3"
    CHECKPOINT_DIR = "./checkpoints_comparative"
    MANIFEST_FILE = "manifest.csv"

    # 1.2 Image Processing
    CROP_SIZE_ORIGINAL = 600
    IMG_SIZE = 512                 

    # 1.3 Hyperparameters
    SEED = 42
    BATCH_SIZE = 16                 
    EPOCHS = 100                  
    
    # Differential Learning Rates
    LR_BACKBONE = 1e-5              # Preserve ImageNet knowledge
    LR_HEAD = 1e-3                  # Learn Cancer features fast
    
    WEIGHT_DECAY = 1e-2
    DROPOUT_RATE = 0.5
    POS_WEIGHT = 1.2                # Balanced penalty
    PATIENCE = 20                   # Early Stopping
    
    DECISION_THRESHOLD = 0.55       # Slightly conservative

    # 1.4 System
    NUM_WORKERS = 2 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1.5 Architecture Comparison 
    MODELS_TO_TRAIN = [
        "densenet121", 
        "efficientnet_v2_s", 
        "resnext50_32x4d"
        ]


# 2. UTILITIES & REPRODUCIBILITY

def set_reproducibility(seed: int = Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    print(f"[SETUP] Reproducibility enabled (SEED: {seed})")

def print_header(title: str):
    print(f"\n{'='*60}\n {title}\n{'='*60}")

def print_step(step: str, msg: str):
    print(f"[{step}] {msg}")


# 3. DATA PROCESSING PIPELINE

class ImageProcessor:
    @staticmethod
    def perform_smart_crop(task_args):
        row, file_index, output_dir, crop_size, final_size = task_args
        filename = f"{row['label']}_{row['patient_id']}_{row['laterality']}_{row['view']}_{row.name}.png"
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            return row.name, output_path, True

        try:
            image_key = str(row["image_file_path"]).split("/")[-2]
            mask_key = str(row["roi_mask_file_path"]).split("/")[-2]
            image_path = file_index.get(image_key, None)
            mask_path = file_index.get(mask_key, None)
            
            # Fallback search
            if not image_path:
                for k, v in file_index.items():
                    if image_key in k: image_path = v; break
            if not mask_path:
                for k, v in file_index.items():
                    if mask_key in k: mask_path = v; break

            if not image_path or not mask_path: return row.name, None, False

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None: return row.name, None, False
            if image.shape != mask.shape:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return row.name, None, False

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x, center_y = x + w // 2, y + h // 2

            max_dim = max(w, h)
            dynamic_size = int(max_dim * 1.5)
            crop_dim = max(crop_size, dynamic_size)
            half_crop = crop_dim // 2
            
            img_h, img_w = image.shape
            x1 = max(0, center_x - half_crop)
            y1 = max(0, center_y - half_crop)
            x2 = min(img_w, center_x + half_crop)
            y2 = min(img_h, center_y + half_crop)

            roi = image[y1:y2, x1:x2]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            roi = clahe.apply(roi)
            roi_resized = cv2.resize(roi, (final_size, final_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, roi_resized)
            return row.name, output_path, True
        except Exception:
            return row.name, None, False

class DataManager:
    def __init__(self):
        self.base_path = Config.CACHE_DIR
        os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
        self.manifest_path = os.path.join(Config.PROCESSED_DIR, Config.MANIFEST_FILE)

    def _index_files(self) -> dict:
        print_step("INFO", f"Indexing files in {self.base_path}...")
        file_index = {}
        for root, _, files in os.walk(self.base_path):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_index[os.path.basename(os.path.dirname(os.path.join(root, filename)))] = os.path.join(root, filename)
        return file_index

    def prepare_dataset(self) -> pd.DataFrame:
        if os.path.exists(self.manifest_path):
            print_step("INFO", "Loading existing manifest file.")
            # WARNING: If you changed preprocessing logic, delete 'manifest.csv' manually!
            return pd.read_csv(self.manifest_path)

        print_step("INIT", "Processing raw dataset...")
        csv_dir = os.path.join(self.base_path, "csv")
        df_train = pd.read_csv(os.path.join(csv_dir, "mass_case_description_train_set.csv"))
        df_test = pd.read_csv(os.path.join(csv_dir, "mass_case_description_test_set.csv"))
        full_df = pd.concat([df_train, df_test], ignore_index=True)

        full_df.columns = [c.replace(" ", "_").lower() for c in full_df.columns]
        full_df = full_df.rename(columns={"image_view": "view", "left_or_right_breast": "laterality", "pathology": "pathology_class"})
        full_df["label"] = (full_df["pathology_class"] == "MALIGNANT").astype(int)
        
        file_index = self._index_files()
        tasks = []
        for idx, row in full_df.iterrows():
            row.name = idx
            tasks.append((row, file_index, Config.PROCESSED_DIR, Config.CROP_SIZE_ORIGINAL, Config.IMG_SIZE))

        print_step("PROC", f"Starting High-Res Processing on {len(tasks)} images...")
        processed_map = {}
        with ProcessPoolExecutor(max_workers=2) as executor:
            results = list(tqdm(executor.map(ImageProcessor.perform_smart_crop, tasks), total=len(tasks)))
        
        for idx, path, success in results:
            if success: processed_map[idx] = path

        full_df["processed_path"] = full_df.index.map(processed_map)
        clean_df = full_df.dropna(subset=["processed_path"]).reset_index(drop=True)

        print_step("SPLIT", "Performing Patient-Aware Splitting...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.SEED)
        train_idx, test_idx = next(gss.split(clean_df, groups=clean_df["patient_id"]))
        clean_df.loc[train_idx, "split"] = "train"
        clean_df.loc[test_idx, "split"] = "test"

        df_train = clean_df[clean_df["split"] == "train"]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=Config.SEED)
        _, val_idx = next(gss_val.split(df_train, groups=df_train["patient_id"]))
        val_patients = df_train.iloc[val_idx]["patient_id"].values
        clean_df.loc[clean_df["patient_id"].isin(val_patients), "split"] = "val"

        clean_df.to_csv(self.manifest_path, index=False)
        return clean_df


# 4. DATASET FACTORY

def get_transforms(split: str):
    # FIX: Using ImageNet Statistics for better Transfer Learning performance
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    
    if split == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
            A.Affine(shear=(-10, 10), scale=(0.85, 1.15), rotate=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.CoarseDropout(max_holes=6, max_height=20, max_width=20, fill_value=0, p=0.1),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2()
        ])
    return A.Compose([
        A.Normalize(mean=imagenet_mean, std=imagenet_std), # FIX: Removed Stray comma
        ToTensorV2()
    ])

class CancerDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['processed_path'], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transforms: img = self.transforms(image=img)['image']
        return img, torch.tensor(row['label'], dtype=torch.float32)


# 5. MODEL FACTORY

def build_model(model_name: str) -> nn.Module:
    """
    Model Factory: Initializes the architecture and modifies the head for binary classification.
    """
    
    print_step("BUILD", f"Initializing Architecture: {model_name.upper()}")
    
    try:
        model = None

        # GROUP A: ResNet Family (ResNet, ResNeXt)
        if model_name == 'resnet50':
            model = models.resnet50(weights='DEFAULT')
        elif model_name == 'resnext50_32x4d':
            model = models.resnext50_32x4d(weights='DEFAULT')

        if model_name in ['resnet50', 'resnext50_32x4d']:
            model.fc = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE),
                nn.Linear(model.fc.in_features, 1)
            )
            return model
      
        # GROUP B: DenseNet Family
        if model_name == 'densenet121':
            model = models.densenet121(weights='DEFAULT')
        elif model_name == 'densenet169':
            model = models.densenet169(weights='DEFAULT')

        if model_name in ['densenet121', 'densenet169']:
            model.classifier = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE),
                nn.Linear(model.classifier.in_features, 1)
            )
            return model

        # GROUP C: EfficientNet Family
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='DEFAULT')
        elif model_name == 'efficientnet_v2_s':
            model = models.efficientnet_v2_s(weights='DEFAULT')

        if model_name in ['efficientnet_b0', 'efficientnet_v2_s']:
            # EfficientNet classifier is a Sequential block: [0]=Dropout, [1]=Linear
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE),
                nn.Linear(in_features, 1)
            )
            return model

        # ERROR HANDLING
        raise ValueError(f"Architecture '{model_name}' is not supported.")

    except Exception as e:
        print(f"[ERROR] Failed to build model {model_name}: {e}")
        raise e


# 6. TRAINING ENGINE

def train_engine(model_name: str, loaders: dict):
    print_header(f"TRAINING PROTOCOL: {model_name.upper()}")
    
    model = build_model(model_name).to(Config.DEVICE)
    save_path = os.path.join(Config.CHECKPOINT_DIR, f"{model_name}_best.pth")
    
    for param in model.parameters(): param.requires_grad = True

    # Identify Backbones vs Heads
    if 'resnet' in model_name or 'resnext' in model_name:
        head_params = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    elif 'densenet' in model_name:
        head_params = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    elif 'efficientnet' in model_name:
        head_params = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': Config.LR_BACKBONE},
        {'params': head_params, 'lr': Config.LR_HEAD}
    ], weight_decay=Config.WEIGHT_DECAY)
    
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # FIX: Safer AMP initialization
    scaler = amp.GradScaler(enabled=(Config.DEVICE.type == 'cuda'))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train
        model.train()
        train_loss = 0
        loop = tqdm(loaders['train'], desc=f"Ep {epoch+1:02d}", leave=False)
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(Config.DEVICE), lbls.to(Config.DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            
            # FIX: Safer AMP autocast
            with amp.autocast(enabled=(Config.DEVICE.type == 'cuda')):
                loss = criterion(model(imgs), lbls)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, lbls in loaders['val']:
                # Inference uses standard precision or amp (optional), here standard is fine
                logits = model(imgs.to(Config.DEVICE))
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(lbls.numpy())
        
        try: val_auc = roc_auc_score(val_targets, val_preds)
        except: val_auc = 0.5
        
        scheduler.step(val_auc)
        avg_loss = train_loss / len(loaders['train'])
        
        print(f"Ep {epoch+1:02d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(" >>> SAVED BEST")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping.")
                break

    # Load Best for Final Inference
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    return model, best_auc 


# 7. MAIN ORCHESTRATOR

if __name__ == "__main__":
    
    print_header("SYSTEM INITIALIZATION")
    set_reproducibility()
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    manager = DataManager()
    df_data = manager.prepare_dataset()
    print_step("INFO", f"Dataset Ready: {len(df_data)} images")
    
    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = CancerDataset(df_data[df_data['split'] == split], transforms=get_transforms(split))
        loaders[split] = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=(split == 'train'), 
                                    num_workers=Config.NUM_WORKERS, pin_memory=True)

    # D. COMPARATIVE TRAINING LOOP
    ensemble_weights = []
    ensemble_preds = []
    y_test_true = []
    
    # Get True Test Labels once
    for _, lbls in loaders['test']: y_test_true.extend(lbls.numpy())
    y_test_true = np.array(y_test_true)

    print_header("STARTING COMPARATIVE STUDY")

    for model_name in Config.MODELS_TO_TRAIN:
        # 1. Train
        trained_model, val_auc = train_engine(model_name, loaders)
        
        # 2. Inference on Test Set
        trained_model.eval()
        preds = []
        with torch.no_grad():
            for imgs, _ in loaders['test']:
                preds.extend(torch.sigmoid(trained_model(imgs.to(Config.DEVICE))).cpu().numpy())
        preds = np.array(preds).flatten()
        
        # 3. Individual Metrics
        y_bin = (preds > Config.DECISION_THRESHOLD).astype(int)
        
        print(f"\n■ REPORT: {model_name.upper()}")
        print(f"  ├─ Validation AUC: {val_auc:.4f} (Weight in Ensemble)")
        print(f"  ├─ Test AUC:       {roc_auc_score(y_test_true, preds):.4f}")
        print(f"  ├─ Test Accuracy:  {accuracy_score(y_test_true, y_bin):.4f}")
        print(f"  └─ F1 Score:       {f1_score(y_test_true, y_bin):.4f}\n")
        
        # 4. Store for Ensemble
        # Weight = (Val AUC - 0.5). If model is random (0.5), weight is 0.
        weight = max(0, val_auc - 0.5) 
        ensemble_weights.append(weight)
        ensemble_preds.append(preds)

    # E. WEIGHTED ENSEMBLE AGGREGATION
    print_header("ENSEMBLE FINAL EVALUATION")
    
    # Normalize weights
    total_weight = sum(ensemble_weights)
    if total_weight == 0: norm_weights = [1/3]*3 # Fallback to equal mean if all failed
    else: norm_weights = [w / total_weight for w in ensemble_weights]
    
    print(f"Ensemble Weights: {dict(zip(Config.MODELS_TO_TRAIN, np.round(norm_weights, 3)))}")

    final_prob = np.zeros_like(ensemble_preds[0])
    for i, preds in enumerate(ensemble_preds):
        final_prob += preds * norm_weights[i]
        
    final_bin = (final_prob > Config.DECISION_THRESHOLD).astype(int)
    
    print("*** FINAL COMPARATIVE RESULTS ***")
    print(f"Ensemble AUC:      {roc_auc_score(y_test_true, final_prob):.4f}")
    print(f"Ensemble Accuracy: {accuracy_score(y_test_true, final_bin):.4f}")
    
    print("\nDetailed Report (Ensemble):")
    print(classification_report(y_test_true, final_bin, target_names=['Benign', 'Malignant']))