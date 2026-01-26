import os
import re
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A


class CBISDDSM_Preprocessor:
    """
    Mass-only preprocessing pipeline for the Kaggle CBIS-DDSM JPEG dataset.

    Key goals:
      - Build a manifest with correct JPEG paths for FULL mammograms + ROI masks.
      - Filter to MASS cases only (no calcifications).
      - Apply breast isolation (Otsu + largest CC) and pectoral muscle suppression for MLO.
      - Apply CLAHE.
      - Extract a fixed-size patch centered at the lesion (mask centroid), with zero padding.
      - Provide group-aware splits by participant_id to avoid leakage.
    """

    def __init__(self, base_path: Optional[str] = None, output_path: Optional[str] = None, seed: int = 42):
        default_kaggle = "/kaggle/input/cbis-ddsm-breast-cancer-image-dataset"
        default_local = "cbis-ddsm-breast-cancer-image-dataset"

        if base_path is None:
            candidate_paths = [
                os.path.join(os.getcwd(), default_local),
                default_local,
                default_kaggle,
            ]
            for path in candidate_paths:
                if os.path.isfile(os.path.join(path, "csv", "mass_case_description_train_set.csv")):
                    base_path = path
                    break
            else:
                base_path = default_local

        if output_path is None:
            output_path = "/kaggle/working" if os.path.isdir("/kaggle/working") else "preprocessed_output"

        self.base_path = base_path
        self.output_path = output_path
        self.seed = seed
        self.manifest: Optional[pd.DataFrame] = None

        os.makedirs(self.output_path, exist_ok=True)

    # ----------------------------
    # 1) Manifest (MASS-only)
    # ----------------------------
    def load_and_filter_manifest(self) -> Optional[pd.DataFrame]:
        """
        Loads MASS CSV(s), joins with dicom_info to get correct JPEG paths for
        full mammograms and ROI masks, filters BENIGN_WITHOUT_CALLBACK, and writes manifest_mass.csv.

        Returns:
            pd.DataFrame or None
        """
        print(">> Building MASS-only manifest...")

        csv_dir = os.path.join(self.base_path, "csv")
        mass_train = os.path.join(csv_dir, "mass_case_description_train_set.csv")
        mass_test = os.path.join(csv_dir, "mass_case_description_test_set.csv")
        dicom_info_path = os.path.join(csv_dir, "dicom_info.csv")

        if not os.path.isfile(mass_train):
            print(f"ERROR: Missing file: {mass_train}")
            return None
        if not os.path.isfile(dicom_info_path):
            print(f"ERROR: Missing file: {dicom_info_path}")
            return None

        mass_dfs = [pd.read_csv(mass_train)]
        if os.path.isfile(mass_test):
            mass_dfs.append(pd.read_csv(mass_test))
            print(">> Found and loaded mass_case_description_test_set.csv (will merge).")

        full_df = pd.concat(mass_dfs, ignore_index=True)

        # Ensure abnormality_type is MASS (some versions already are)
        if "abnormality type" in full_df.columns:
            full_df = full_df[full_df["abnormality type"].astype(str).str.lower().eq("mass")].copy()

        dicom_info = pd.read_csv(dicom_info_path)
        dicom_info["image_path_clean"] = dicom_info["image_path"].astype(str).str.replace("CBIS-DDSM/", "", regex=False)

        full_mamo_info = dicom_info[dicom_info["SeriesDescription"] == "full mammogram images"].copy()
        roi_mask_info = dicom_info[dicom_info["SeriesDescription"] == "ROI mask images"].copy()

        # Build dir_key and PatientID maps; choose the one with more valid matches.
        full_mamo_info["dir_key"] = full_mamo_info["image_path_clean"].str.split("/").str[0]
        roi_mask_info["dir_key"] = roi_mask_info["image_path_clean"].str.split("/").str[0]

        full_mamo_dir = full_mamo_info.drop_duplicates(subset=["dir_key"], keep="first")
        roi_mask_dir = roi_mask_info.drop_duplicates(subset=["dir_key"], keep="first")

        img_map_dir = dict(zip(full_mamo_dir["dir_key"], full_mamo_dir["image_path_clean"]))
        mask_map_dir = dict(zip(roi_mask_dir["dir_key"], roi_mask_dir["image_path_clean"]))

        full_mamo_pid = full_mamo_info.drop_duplicates(subset=["PatientID"], keep="first")
        roi_mask_pid = roi_mask_info.drop_duplicates(subset=["PatientID"], keep="first")

        img_map_pid = dict(zip(full_mamo_pid["PatientID"], full_mamo_pid["image_path_clean"]))
        mask_map_pid = dict(zip(roi_mask_pid["PatientID"], roi_mask_pid["image_path_clean"]))

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

        full_df = full_df[full_df["pathology"] != "BENIGN_WITHOUT_CALLBACK"].copy()
        full_df["label"] = full_df["pathology"].apply(lambda x: 1 if str(x).upper() == "MALIGNANT" else 0)

        full_df["participant_id"] = full_df["patient_id"].astype(str).str.extract(r"(P_\d+)", expand=False)
        full_df["participant_id"] = full_df["participant_id"].fillna(full_df["patient_id"].astype(str))

        before = len(full_df)
        full_df = full_df.dropna(subset=["image_path", "mask_path"]).copy()
        after = len(full_df)
        print(f">> Path mapping: {after} valid MASS samples (from {before} after label filtering).")

        cols = ["patient_id", "participant_id", "image_path", "mask_path", "image view", "label"]
        if "abnormality type" in full_df.columns:
            cols.append("abnormality type")

        self.manifest = full_df[cols].rename(columns={"image view": "view", "abnormality type": "abnormality_type"})
        if "abnormality_type" not in self.manifest.columns:
            self.manifest["abnormality_type"] = "mass"

        self.manifest = self.manifest.reset_index(drop=True)
        out_csv = os.path.join(self.output_path, "manifest_mass.csv")
        self.manifest.to_csv(out_csv, index=False)
        print(f">> Saved: {out_csv}")

        self.print_dataset_statistics()
        return self.manifest

    def print_dataset_statistics(self):
        if self.manifest is None:
            return
        print("\n" + "=" * 60)
        print("MASS-ONLY DATASET STATS")
        print("=" * 60)
        print(f"Samples: {len(self.manifest)}")
        print("\nLabel distribution:")
        print(self.manifest["label"].value_counts())
        print("\nViews:")
        print(self.manifest["view"].value_counts())
        print("=" * 60 + "\n")

    # ----------------------------
    # 2) Read + normalize
    # ----------------------------
    def read_and_normalize_image(self, image_rel_path: str, view: str) -> Optional[np.ndarray]:
        full_path = os.path.join(self.base_path, image_rel_path)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        img = img.astype(np.float32) / 255.0
        img = self.segment_breast_otsu(img)
        img = self.suppress_pectoral_muscle(img, view)
        return img

    def read_mask(self, mask_rel_path: str) -> Optional[np.ndarray]:
        full_path = os.path.join(self.base_path, mask_rel_path)
        mask = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return (mask_bin / 255.0).astype(np.float32)

    # ----------------------------
    # 3) Breast isolation (Otsu + largest CC)
    # ----------------------------
    def segment_breast_otsu(self, image: np.ndarray) -> np.ndarray:
        img_uint8 = (image * 255).astype(np.uint8)
        _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        largest = max(contours, key=cv2.contourArea)
        mask_clean = np.zeros_like(img_uint8)
        cv2.drawContours(mask_clean, [largest], -1, 255, thickness=cv2.FILLED)

        masked = cv2.bitwise_and(img_uint8, img_uint8, mask=mask_clean)
        return masked.astype(np.float32) / 255.0

    def _is_breast_on_left(self, image_uint8: np.ndarray) -> bool:
        h, w = image_uint8.shape
        return np.sum(image_uint8[:, : w // 2]) > np.sum(image_uint8[:, w // 2 :])

    # ----------------------------
    # 4) Pectoral suppression (MLO only)
    # ----------------------------
    def suppress_pectoral_muscle(self, image: np.ndarray, view: str) -> np.ndarray:
        if view != "MLO":
            return image

        img_uint8 = (image * 255).astype(np.uint8)
        is_left = self._is_breast_on_left(img_uint8)
        h, w = img_uint8.shape

        edges = cv2.Canny(img_uint8, 30, 100)
        roi_mask = np.zeros_like(edges)

        if is_left:
            roi_mask[0 : h // 2, 0 : w // 2] = 255
        else:
            roi_mask[0 : h // 2, w // 2 : w] = 255

        edges_roi = cv2.bitwise_and(edges, edges, mask=roi_mask)

        lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=25, minLineLength=30, maxLineGap=30)
        if lines is None:
            return image

        best = None
        best_len = 0.0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 20 < angle < 85 and length > best_len:
                best_len = length
                best = (x1, y1, x2, y2)

        if best is None:
            return image

        x1, y1, x2, y2 = best
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        mask = np.zeros_like(img_uint8)

        if is_left:
            x_int = int(-b / m) if m != 0 else 0
            y_int = int(b)
            pts = np.array([[[0, 0], [0, min(y_int, h)], [min(x_int, w), 0]]], dtype=np.int32)
        else:
            x_int = int(-b / m) if m != 0 else w
            y_int = int(m * w + b)
            pts = np.array([[[w, 0], [w, min(y_int, h)], [min(x_int, w), 0]]], dtype=np.int32)

        cv2.fillPoly(mask, pts, 255)
        img_no_muscle = cv2.bitwise_and(img_uint8, img_uint8, mask=cv2.bitwise_not(mask))
        return img_no_muscle.astype(np.float32) / 255.0

    # ----------------------------
    # 5) CLAHE
    # ----------------------------
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        img_uint8 = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(img_uint8)
        return out.astype(np.float32) / 255.0

    # ----------------------------
    # 6) Patch extraction around lesion
    # ----------------------------
    def extract_lesion_patch(self, image: np.ndarray, mask: np.ndarray, patch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        mask_uint8 = (mask * 255).astype(np.uint8)
        moments = cv2.moments(mask_uint8)
        if moments["m00"] == 0:
            cy, cx = image.shape[0] // 2, image.shape[1] // 2
        else:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

        half = patch_size // 2
        pad = ((half, half), (half, half))
        img_p = np.pad(image, pad, mode="constant", constant_values=0)
        msk_p = np.pad(mask, pad, mode="constant", constant_values=0)

        pcx, pcy = cx + half, cy + half
        sx, ex = pcx - half, pcx - half + patch_size
        sy, ey = pcy - half, pcy - half + patch_size

        img_patch = img_p[sy:ey, sx:ex]
        msk_patch = msk_p[sy:ey, sx:ex]

        if img_patch.shape != (patch_size, patch_size):
            img_patch = cv2.resize(img_patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            msk_patch = cv2.resize(msk_patch, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

        return img_patch.astype(np.float32), (msk_patch > 0.5).astype(np.float32)

    # ----------------------------
    # 7) Class weights (classification label, optional)
    # ----------------------------
    def compute_class_weights(self, train_df: pd.DataFrame) -> dict:
        y = train_df["label"].values
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
        weight_dict = dict(zip(np.unique(y), weights))
        print(f">> Class weights (label): {weight_dict}")
        return weight_dict

    # ----------------------------
    # 8) Augmentation (mass-friendly)
    # ----------------------------
    def get_augmentation_pipeline(self):
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, border_mode=0, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.15, border_mode=0, p=0.25),
                A.ElasticTransform(alpha=30, sigma=5, border_mode=0, p=0.15),
            ]
        )

    def apply_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        aug = self.get_augmentation_pipeline()
        out = aug(image=image, mask=mask)
        img = out["image"].astype(np.float32)
        msk = (out["mask"] > 0.5).astype(np.float32)
        return img, msk

    # ----------------------------
    # Group split (leakage-safe)
    # ----------------------------
    def create_group_split(self, test_size: float = 0.15, val_size: float = 0.15, group_col: str = "participant_id"):
        if self.manifest is None:
            raise ValueError("manifest is None. Call load_and_filter_manifest first.")
        if len(self.manifest) == 0:
            raise ValueError("manifest is empty. Check path mapping and mask filtering.")
        if group_col not in self.manifest.columns:
            raise ValueError(f"Missing group column: {group_col}")

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
        train_val_idx, test_idx = next(gss.split(self.manifest, groups=self.manifest[group_col]))
        train_val = self.manifest.iloc[train_val_idx].reset_index(drop=True)
        test = self.manifest.iloc[test_idx].reset_index(drop=True)

        val_ratio = val_size / (1 - test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=self.seed)
        train_idx, val_idx = next(gss2.split(train_val, groups=train_val[group_col]))
        train = train_val.iloc[train_idx].reset_index(drop=True)
        val = train_val.iloc[val_idx].reset_index(drop=True)

        return train, val, test


def main():
    proc = CBISDDSM_Preprocessor()
    df = proc.load_and_filter_manifest()
    if df is None:
        return
    train, val, test = proc.create_group_split()
    print(f">> Split: train={len(train)} val={len(val)} test={len(test)}")
    proc.compute_class_weights(train)


if __name__ == "__main__":
    main()
