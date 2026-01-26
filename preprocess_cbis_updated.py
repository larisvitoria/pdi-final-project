import os
import random
from typing import Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


class CBISDDSM_Preprocessor:
    """
    CBIS-DDSM JPEG preprocessor designed for lesion segmentation on ROI masks.

    Key upgrades vs your current version:
    - Safer image/mask path mapping using directory key (avoids PatientID collisions).
    - Filters out samples with empty ROI masks (prevents training collapse to all-background).
    - Applies breast masking + (optional) pectoral suppression consistently to BOTH image and lesion mask.
    - Uses more realistic augmentation (removes extreme 175° rotations + heavy elastic warps).
    - Patch extraction centers on lesion bbox (more robust than centroid when masks are elongated).
    """

    def __init__(self, base_path: Optional[str] = None, output_path: Optional[str] = None):
        default_kaggle = "/kaggle/input/cbis-ddsm-breast-cancer-image-dataset"
        default_local = "cbis-ddsm-breast-cancer-image-dataset"

        if base_path is None:
            candidate_paths = [
                os.path.join(os.getcwd(), default_local),
                default_local,
                default_kaggle,
            ]
            for path in candidate_paths:
                if os.path.isfile(os.path.join(path, "csv", "calc_case_description_train_set.csv")):
                    base_path = path
                    break
            else:
                base_path = default_local

        if output_path is None:
            output_path = "/kaggle/working" if os.path.isdir("/kaggle/working") else "preprocessed_output"

        self.base_path = base_path
        self.output_path = output_path
        self.manifest: Optional[pd.DataFrame] = None

    # ----------------------------
    # 1) Manifest creation
    # ----------------------------
    def load_and_filter_manifest(self, drop_empty_masks: bool = True) -> Optional[pd.DataFrame]:
        print(">> Loading CSVs and building manifest...")

        try:
            calc_df = pd.read_csv(f"{self.base_path}/csv/calc_case_description_train_set.csv")
            mass_df = pd.read_csv(f"{self.base_path}/csv/mass_case_description_train_set.csv")
            dicom_info = pd.read_csv(f"{self.base_path}/csv/dicom_info.csv")
        except FileNotFoundError:
            print(f"ERROR: Files not found under {self.base_path}")
            return None

        full_df = pd.concat([calc_df, mass_df], ignore_index=True)

        # Clean dicom_info paths (Kaggle format)
        dicom_info["image_path_clean"] = dicom_info["image_path"].str.replace("CBIS-DDSM/", "", regex=False)

        # Map by dir_key or PatientID; choose the better match.
        dicom_info["dir_key"] = dicom_info["image_path_clean"].astype(str).str.split("/").str[0]

        full_mamo_info = dicom_info[dicom_info["SeriesDescription"] == "full mammogram images"].copy()
        roi_mask_info = dicom_info[dicom_info["SeriesDescription"] == "ROI mask images"].copy()

        full_mamo_dir = full_mamo_info.drop_duplicates(subset=["dir_key"], keep="first")
        roi_mask_dir = roi_mask_info.drop_duplicates(subset=["dir_key"], keep="first")

        img_map_dir = dict(zip(full_mamo_dir["dir_key"], full_mamo_dir["image_path_clean"]))
        mask_map_dir = dict(zip(roi_mask_dir["dir_key"], roi_mask_dir["image_path_clean"]))

        full_mamo_pid = full_mamo_info.drop_duplicates(subset=["PatientID"], keep="first")
        roi_mask_pid = roi_mask_info.drop_duplicates(subset=["PatientID"], keep="first")

        img_map_pid = dict(zip(full_mamo_pid["PatientID"], full_mamo_pid["image_path_clean"]))
        mask_map_pid = dict(zip(roi_mask_pid["PatientID"], roi_mask_pid["image_path_clean"]))

        # Extract dir keys from description CSVs
        full_df["img_key"] = full_df["image file path"].astype(str).str.split("/").str[0]
        full_df["mask_key"] = full_df["ROI mask file path"].astype(str).str.split("/").str[0]

        image_path_dir = full_df["img_key"].map(img_map_dir)
        mask_path_dir = full_df["mask_key"].map(mask_map_dir)
        image_path_pid = full_df["img_key"].map(img_map_pid)
        mask_path_pid = full_df["mask_key"].map(mask_map_pid)

        dir_matches = (image_path_dir.notna() & mask_path_dir.notna()).sum()
        pid_matches = (image_path_pid.notna() & mask_path_pid.notna()).sum()

        if pid_matches >= dir_matches:
            full_df["image_path"] = image_path_pid
            full_df["mask_path"] = mask_path_pid
            print(">> Using PatientID mapping for image/mask paths.")
        else:
            full_df["image_path"] = image_path_dir
            full_df["mask_path"] = mask_path_dir
            print(">> Using dir_key mapping for image/mask paths.")

        # Filter pathology + labels
        full_df = full_df[full_df["pathology"] != "BENIGN_WITHOUT_CALLBACK"].copy()
        full_df["label"] = full_df["pathology"].apply(lambda x: 1 if x == "MALIGNANT" else 0)

        # Participant grouping (to avoid leakage)
        full_df["participant_id"] = full_df["patient_id"].astype(str).str.extract(r"(P_\d+)", expand=False)
        full_df["participant_id"] = full_df["participant_id"].fillna(full_df["patient_id"].astype(str))

        initial_count = len(full_df)
        full_df = full_df.dropna(subset=["image_path", "mask_path"]).copy()
        print(f">> Path mapping OK: {len(full_df)} samples (from {initial_count})")

        # Select final columns
        full_df = full_df[[
            "patient_id", "participant_id", "image_path", "mask_path",
            "image view", "label", "abnormality type"
        ]].rename(columns={"image view": "view", "abnormality type": "abnormality_type"}).reset_index(drop=True)

        # Optionally drop empty masks (strongly recommended for lesion-centric segmentation)
        if drop_empty_masks:
            full_df = self._drop_empty_masks(full_df)

        self.manifest = full_df.reset_index(drop=True)

        os.makedirs(self.output_path, exist_ok=True)
        save_loc = os.path.join(self.output_path, "manifest.csv")
        self.manifest.to_csv(save_loc, index=False)

        self.print_dataset_statistics()
        return self.manifest

    def _drop_empty_masks(self, df: pd.DataFrame) -> pd.DataFrame:
        print(">> Checking for empty ROI masks (this runs once; can take a bit)...")
        keep_rows = []
        empty = 0
        for i, row in df.iterrows():
            m = self.read_mask(row["mask_path"])
            if m is None:
                empty += 1
                continue
            if float(m.sum()) <= 0.0:
                empty += 1
                continue
            keep_rows.append(i)

        filtered = df.iloc[keep_rows].copy().reset_index(drop=True)
        print(f">> Dropped {empty} samples with empty/unreadable masks. Remaining: {len(filtered)}")
        return filtered

    def print_dataset_statistics(self):
        if self.manifest is None:
            return
        print("\n" + "=" * 50)
        print("Dataset Statistics")
        print("=" * 50)
        print(f"Samples: {len(self.manifest)}")
        print("\nClass distribution:\n", self.manifest["label"].value_counts())
        print("\nAbnormality types:\n", self.manifest["abnormality_type"].value_counts())
        print("=" * 50 + "\n")

    def create_group_split(self, test_size: float = 0.15, val_size: float = 0.15, group_col: str = "participant_id"):
        if self.manifest is None:
            raise RuntimeError("manifest is None. Run load_and_filter_manifest() first.")
        if len(self.manifest) == 0:
            raise ValueError("manifest is empty. Check path mapping and mask filtering.")
        if group_col not in self.manifest.columns:
            raise ValueError(f"Missing group column: {group_col}")

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_val_idx, test_idx = next(gss.split(self.manifest, groups=self.manifest[group_col]))
        train_val_df = self.manifest.iloc[train_val_idx].reset_index(drop=True)
        test_df = self.manifest.iloc[test_idx].reset_index(drop=True)

        val_ratio = val_size / (1 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[group_col]))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        return train_df, val_df, test_df

    # ----------------------------
    # 2) I/O
    # ----------------------------
    def read_and_normalize_image(self, image_rel_path: str) -> Optional[np.ndarray]:
        full_path = f"{self.base_path}/{image_rel_path}"
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return img.astype(np.float32) / 255.0

    def read_mask(self, mask_rel_path: str) -> Optional[np.ndarray]:
        full_path = f"{self.base_path}/{mask_rel_path}"
        mask = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return (mask_binary / 255.0).astype(np.float32)

    # ----------------------------
    # 3) Breast masking (Otsu)
    # ----------------------------
    def breast_mask_otsu(self, image: np.ndarray) -> np.ndarray:
        # image: float32 [0,1]
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean small blobs + fill small holes
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones_like(img_uint8, dtype=np.uint8) * 255  # fallback: keep everything

        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(img_uint8, dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        return mask  # uint8 0/255

    def apply_breast_mask(self, image: np.ndarray, lesion_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        bm = self.breast_mask_otsu(image)
        img_uint8 = (image * 255).astype(np.uint8)
        img_masked = cv2.bitwise_and(img_uint8, img_uint8, mask=bm).astype(np.float32) / 255.0

        # Ensure lesion mask does not leak outside breast (safety)
        lm_uint8 = (lesion_mask * 255).astype(np.uint8)
        lm_masked = cv2.bitwise_and(lm_uint8, lm_uint8, mask=bm).astype(np.float32) / 255.0
        return img_masked, (lm_masked > 0.5).astype(np.float32)

    # ----------------------------
    # 4) Pectoral suppression (MLO)
    # ----------------------------
    def _is_breast_on_left(self, img_uint8: np.ndarray) -> bool:
        h, w = img_uint8.shape
        return img_uint8[:, : w // 2].sum() > img_uint8[:, w // 2 :].sum()

    def pectoral_mask_hough(self, image: np.ndarray, view: str) -> Optional[np.ndarray]:
        if view != "MLO":
            return None

        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        h, w = img_uint8.shape
        is_left = self._is_breast_on_left(img_uint8)

        edges = cv2.Canny(img_uint8, 30, 120)

        roi = np.zeros_like(edges, dtype=np.uint8)
        if is_left:
            roi[0 : h // 2, 0 : w // 2] = 255
        else:
            roi[0 : h // 2, w // 2 : w] = 255
        edges = cv2.bitwise_and(edges, edges, mask=roi)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=25,
            minLineLength=30,
            maxLineGap=30,
        )
        if lines is None:
            return None

        best_line = None
        best_len = 0.0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = float(np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
            if 20 < angle < 85 and length > best_len:
                best_len = length
                best_line = (x1, y1, x2, y2)

        if best_line is None:
            return None

        x1, y1, x2, y2 = best_line
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        mask = np.zeros((h, w), dtype=np.uint8)

        if is_left:
            x_int = int(-b / m) if m != 0 else 0
            y_int = int(b)
            pts = np.array([[[0, 0], [0, np.clip(y_int, 0, h - 1)], [np.clip(x_int, 0, w - 1), 0]]], dtype=np.int32)
        else:
            x_int = int(-b / m) if m != 0 else w - 1
            y_int = int(m * (w - 1) + b)
            pts = np.array([[[w - 1, 0], [w - 1, np.clip(y_int, 0, h - 1)], [np.clip(x_int, 0, w - 1), 0]]], dtype=np.int32)

        cv2.fillPoly(mask, pts, 255)
        return mask  # uint8 0/255

    def apply_pectoral_suppression(self, image: np.ndarray, lesion_mask: np.ndarray, view: str) -> Tuple[np.ndarray, np.ndarray]:
        pm = self.pectoral_mask_hough(image, view)
        if pm is None:
            return image, lesion_mask

        img_uint8 = (image * 255).astype(np.uint8)
        lm_uint8 = (lesion_mask * 255).astype(np.uint8)

        inv = cv2.bitwise_not(pm)
        img_out = cv2.bitwise_and(img_uint8, img_uint8, mask=inv).astype(np.float32) / 255.0
        lm_out = cv2.bitwise_and(lm_uint8, lm_uint8, mask=inv).astype(np.float32) / 255.0
        return img_out, (lm_out > 0.5).astype(np.float32)

    # ----------------------------
    # 5) CLAHE
    # ----------------------------
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(img_uint8).astype(np.float32) / 255.0
        return out

    # ----------------------------
    # 6) Patch extraction
    # ----------------------------
    def extract_lesion_patch(self, image: np.ndarray, mask: np.ndarray, patch_size: int = 598, margin: int = 32):
        """
        - If the lesion bbox (plus margin) fits inside patch_size: crop patch_size at native scale centered on bbox center.
        - If bbox+margin is larger than patch_size: crop a larger window around the bbox and resize down to patch_size.
        """
        mask_bin = (mask > 0.5).astype(np.uint8)
        ys, xs = np.where(mask_bin > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Empty mask (no lesion pixels).")

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        cx = int((x0 + x1) / 2)
        cy = int((y0 + y1) / 2)

        bbox_w = (x1 - x0 + 1) + 2 * margin
        bbox_h = (y1 - y0 + 1) + 2 * margin
        needed = int(max(bbox_w, bbox_h))

        crop_size = patch_size if needed <= patch_size else needed

        half = crop_size // 2
        pad = half + 2
        img_p = np.pad(image, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
        msk_p = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

        cx_p = cx + pad
        cy_p = cy + pad

        y_start = cy_p - half
        y_end = y_start + crop_size
        x_start = cx_p - half
        x_end = x_start + crop_size

        img_crop = img_p[y_start:y_end, x_start:x_end]
        msk_crop = msk_p[y_start:y_end, x_start:x_end]

        if img_crop.shape != (crop_size, crop_size):
            # very rare boundary issues
            img_crop = cv2.resize(img_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            msk_crop = cv2.resize(msk_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

        if crop_size != patch_size:
            img_crop = cv2.resize(img_crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            msk_crop = cv2.resize(msk_crop, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

        msk_crop = (msk_crop > 0.5).astype(np.float32)
        return img_crop.astype(np.float32), msk_crop

    # ----------------------------
    # 8) Augmentation (tuned)
    # ----------------------------
    def get_augmentation_pipeline(self):
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0.05,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.7,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.12, p=0.3),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ]
        )

    def apply_augmentation(self, image: np.ndarray, mask: np.ndarray):
        aug = self.get_augmentation_pipeline()
        out = aug(image=image, mask=mask)
        img_out = out["image"].astype(np.float32)
        mask_out = (out["mask"] > 0.5).astype(np.float32)
        return img_out, mask_out

    # ----------------------------
    # Full pair preprocessing: 2..6 (+8 optional)
    # ----------------------------
    def preprocess_pair(self, row: pd.Series, patch_size: int = 598, do_aug: bool = False):
        img = self.read_and_normalize_image(row["image_path"])
        msk = self.read_mask(row["mask_path"])
        if img is None or msk is None:
            raise ValueError("Failed to read image/mask.")

        # Align shapes (rare)
        if msk.shape != img.shape:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Step 3: breast mask (applied to both)
        img, msk = self.apply_breast_mask(img, msk)

        # Step 4: pectoral suppression (applied to both)
        img, msk = self.apply_pectoral_suppression(img, msk, row["view"])

        # Step 5: CLAHE
        img = self.apply_clahe(img)

        # Step 6: lesion patch
        img_patch, msk_patch = self.extract_lesion_patch(img, msk, patch_size=patch_size)

        # Step 8: augmentation (optional)
        if do_aug:
            img_patch, msk_patch = self.apply_augmentation(img_patch, msk_patch)

        # Final safety: ensure mask is binary and non-empty
        msk_patch = (msk_patch > 0.5).astype(np.float32)
        if float(msk_patch.sum()) <= 0.0:
            raise ValueError("Empty mask after preprocessing (check mapping / filtering).")

        return img_patch.astype(np.float32), msk_patch.astype(np.float32)


def run_pipeline():
    proc = CBISDDSM_Preprocessor()
    df = proc.load_and_filter_manifest(drop_empty_masks=True)

    if df is None:
        return

    train_df, val_df, test_df = proc.create_group_split()
    print(f">> Group split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # Save splits
    os.makedirs(proc.output_path, exist_ok=True)
    train_df.to_csv(os.path.join(proc.output_path, "train_manifest.csv"), index=False)
    val_df.to_csv(os.path.join(proc.output_path, "val_manifest.csv"), index=False)
    test_df.to_csv(os.path.join(proc.output_path, "test_manifest.csv"), index=False)

    print(">> Done. Manifests saved in:", proc.output_path)


if __name__ == "__main__":
    run_pipeline()
