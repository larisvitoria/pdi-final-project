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
from torch.cuda.amp import GradScaler, autocast
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
    Central configuration for the Breast Cancer Classification Pipeline.
    Designed for stability and reproducibility.
    """

    # 1.1 Paths
    CACHE_DIR = "./data"
    PROCESSED_DIR = "./processed_mass_highres_final_v2"
    CHECKPOINT_DIR = "./checkpoints"
    MANIFEST_FILE = "manifest.csv"

    # 1.2 Image Processing
    CROP_SIZE_ORIGINAL = 600        # Initial context extraction
    IMG_SIZE = 384                  # CNN Input Size

    # 1.3 Hyperparameters
    SEED = 42
    BATCH_SIZE = 16            
    EPOCHS = 100
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-2
    DROPOUT_RATE = 0.7
    PATIENCE = 15                   # Early Stopping Patience

    # 1.4 System
    NUM_WORKERS = 2 if os.name != 'nt' else 0  # 0 is safer on Windows for debugging
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1.5 Architecture
    MODELS_TO_TRAIN = [
        "resnet50",
        "densenet121",
        "efficientnet_b0"
    ]


# 2. UTILITIES & REPRODUCIBILITY

def set_reproducibility(seed: int = Config.SEED):
    """ Ensures deterministic behavior across runs. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[SETUP] Reproducibility enabled (SEED: {seed})")

def print_header(title: str):
    print(f"\n{'='*60}\n {title}\n{'='*60}")

def print_step(step: str, msg: str):
    print(f"[{step}] {msg}")


# 3. DATA PROCESSING PIPELINE

class ImageProcessor:
    """ Handles ROI extraction and Image Enhancement. """
    
    @staticmethod
    def perform_smart_crop(task_args):
        # Unpack arguments
        row, file_index, output_dir, crop_size, final_size = task_args
        
        # Metadata
        filename = f"{row['label']}_{row['patient_id']}_{row['laterality']}_{row['view']}_{row.name}.png"
        output_path = os.path.join(output_dir, filename)

        # 3.A Check Cache
        if os.path.exists(output_path):
            return row.name, output_path, True

        try:
            # 3.B Locate Files
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

            # 3.C Process (Load -> Crop -> Enhance -> Resize)
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

            # Dynamic Crop
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
            
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            roi = clahe.apply(roi)
            
            roi_resized = cv2.resize(roi, (final_size, final_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, roi_resized)
            
            return row.name, output_path, True

        except Exception:
            return row.name, None, False

class DataManager:
    """ Orchestrates File Indexing and Dataset Creation. """
    
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
            return pd.read_csv(self.manifest_path)

        print_step("INIT", "Processing raw dataset (First Run)...")
        
        # 3.1 Load CSVs
        csv_dir = os.path.join(self.base_path, "csv")
        df_train = pd.read_csv(os.path.join(csv_dir, "mass_case_description_train_set.csv"))
        df_test = pd.read_csv(os.path.join(csv_dir, "mass_case_description_test_set.csv"))
        full_df = pd.concat([df_train, df_test], ignore_index=True)

        # 3.2 Cleanup Metadata
        full_df.columns = [c.replace(" ", "_").lower() for c in full_df.columns]
        full_df = full_df.rename(columns={"image_view": "view", "left_or_right_breast": "laterality", "pathology": "pathology_class"})
        full_df["label"] = (full_df["pathology_class"] == "MALIGNANT").astype(int)
        
        file_index = self._index_files()

        # 3.3 Parallel Processing
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

        # 3.4 Patient-Aware Split
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
    if split == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=25, p=0.6),
            A.Affine(shear=(-10, 10), scale=(0.85, 1.15), p=0.6),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

class CancerDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
    
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['processed_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms: img = self.transforms(image=img)['image']
        return img, torch.tensor(row['label'], dtype=torch.float32)


# 5. MODEL FACTORY

def build_model(model_name: str):
    print_step("BUILD", f"Initializing {model_name.upper()}...")
    if model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Sequential(nn.Dropout(Config.DROPOUT_RATE), nn.Linear(model.fc.in_features, 1))

    elif model_name == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        model.classifier = nn.Sequential(nn.Dropout(Config.DROPOUT_RATE), nn.Linear(model.classifier.in_features, 1))

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Sequential(nn.Dropout(Config.DROPOUT_RATE), nn.Linear(model.classifier[1].in_features, 1))

    else:
        raise ValueError("Model not supported")
    
    return model


# 6. TRAINING ENGINE

def train_engine(model_name: str, loaders: dict):
    print_header(f"TRAINING PROTOCOL: {model_name.upper()}")
    
    model = build_model(model_name).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # 6.1 Weighted Loss (Prioritizing Sensitivity)
    pos_weight = torch.tensor([3.0]).to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler()
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    best_auc = 0.0
    patience_counter = 0
    save_path = os.path.join(Config.CHECKPOINT_DIR, f"{model_name}_best.pth")

    for epoch in range(Config.EPOCHS):
        # A. Training Step
        model.train()
        train_loss = 0
        loop = tqdm(loaders['train'], desc=f"Ep {epoch+1:02d}", leave=False)
        
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(Config.DEVICE), lbls.to(Config.DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0] # Só para monitorar (opcional)

        # B. Validation Step
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, lbls in loaders['val']:
                imgs = imgs.to(Config.DEVICE)
                val_preds.extend(torch.sigmoid(model(imgs)).cpu().numpy())
                val_targets.extend(lbls.numpy())
        
        # C. Metrics & Checkpointing
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except: val_auc = 0.5
        
        avg_loss = train_loss / len(loaders['train'])
        scheduler.step(val_auc)

        log_msg = f"Ep {epoch+1:02d} | LR: {current_lr:.1e} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}"
        # log_msg = f"Ep {epoch+1:02d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}"
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(f"{log_msg} | >>> SAVED BEST")
            
        else:
            patience_counter += 1
            print(f"{log_msg} | Patience: {patience_counter}/{Config.PATIENCE}")
            
        if patience_counter >= Config.PATIENCE:
            print(f"\n[STOP] Early stopping triggered.")
            break

    # D. Final Test Inference
    print_step("TEST", "Loading best model for inference...")
    if os.path.exists(save_path): model.load_state_dict(torch.load(save_path))
    model.eval()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for imgs, lbls in loaders['test']:
            test_preds.extend(torch.sigmoid(model(imgs.to(Config.DEVICE))).cpu().numpy())
            test_targets.extend(lbls.numpy())

    return np.array(test_preds).flatten(), np.array(test_targets).flatten()


# 7. MAIN ORCHESTRATOR

if __name__ == "__main__":
    
    # A. Setup
    print_header("SYSTEM INITIALIZATION")
    set_reproducibility()
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # B. Data Prep
    manager = DataManager()
    df_data = manager.prepare_dataset()
    print_step("INFO", f"Dataset Ready: {len(df_data)} images")
    print(df_data['split'].value_counts())

    # C. DataLoaders
    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = CancerDataset(df_data[df_data['split'] == split], transforms=get_transforms(split))
        loaders[split] = DataLoader(
            ds, batch_size=Config.BATCH_SIZE, shuffle=(split == 'train'), 
            num_workers=Config.NUM_WORKERS, pin_memory=True
        )

    # D. Ensemble Training Loop
    results_preds = {}
    y_test_true = None

    for model_name in Config.MODELS_TO_TRAIN:
        # 1. Train & Predict
        y_prob, y_true = train_engine(model_name, loaders)
        results_preds[model_name] = y_prob
        y_test_true = y_true
        
        # 2. Evaluate Individual Model
        y_bin = (y_prob > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
        
        print(f"\n■ REPORT: {model_name.upper()}")
        print(f"  ├─ AUC: {roc_auc_score(y_true, y_prob):.4f} | Acc: {accuracy_score(y_true, y_bin):.4f}")
        print(f"  ├─ Recall (Sens): {recall_score(y_true, y_bin):.4f} | Spec: {tn/(tn+fp):.4f}")
        print(f"  └─ Matrix: TP={tp} | FN={fn} (Miss) || FP={fp} | TN={tn}\n")

    # E. Ensemble Aggregation
    print_header("ENSEMBLE FINAL EVALUATION")
    
    ensemble_prob = np.mean(list(results_preds.values()), axis=0)
    ensemble_bin = (ensemble_prob > 0.5).astype(int)
    
    print("*** FINAL METRICS ***")
    print(f"AUC Score:      {roc_auc_score(y_test_true, ensemble_prob):.4f}")
    print(f"Accuracy:       {accuracy_score(y_test_true, ensemble_bin):.4f}")
    print(f"F1 Score:       {f1_score(y_test_true, ensemble_bin):.4f}")
    
    print("\nDetailed Report:")
    print(classification_report(y_test_true, ensemble_bin, target_names=['Benign', 'Malignant']))
    
    print_step("SUCCESS", "Pipeline execution completed.")