import os
import random
from typing import Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from sklearn.model_selection import GroupShuffleSplit


class CBISDDSM_Preprocessor:
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

    def load_and_filter_manifest(self) -> Optional[pd.DataFrame]:
        print(">> Loading CSV files and consolidating manifest...")
        try:
            calc_df = pd.read_csv(os.path.join(self.base_path, "csv", "calc_case_description_train_set.csv"))
            mass_df = pd.read_csv(os.path.join(self.base_path, "csv", "mass_case_description_train_set.csv"))
            dicom_info = pd.read_csv(os.path.join(self.base_path, "csv", "dicom_info.csv"))
        except FileNotFoundError:
            print(f"ERROR: Files not found in {self.base_path}")
            return None

        full_df = pd.concat([calc_df, mass_df], ignore_index=True)
        dicom_info["image_path_clean"] = dicom_info["image_path"].str.replace("CBIS-DDSM/", "", regex=False)

        full_mamo_info = dicom_info[dicom_info["SeriesDescription"] == "full mammogram images"]
        roi_mask_info = dicom_info[dicom_info["SeriesDescription"] == "ROI mask images"]

        img_map = dict(zip(full_mamo_info["PatientID"], full_mamo_info["image_path_clean"]))
        mask_map = dict(zip(roi_mask_info["PatientID"], roi_mask_info["image_path_clean"]))

        full_df["img_key"] = full_df["image file path"].str.split("/").str[0]
        full_df["mask_key"] = full_df["ROI mask file path"].str.split("/").str[0]

        full_df["image_path"] = full_df["img_key"].map(img_map)
        full_df["mask_path"] = full_df["mask_key"].map(mask_map)

        full_df = full_df[full_df["pathology"] != "BENIGN_WITHOUT_CALLBACK"].copy()
        full_df["label"] = full_df["pathology"].apply(lambda x: 1 if x == "MALIGNANT" else 0)
        full_df["participant_id"] = full_df["patient_id"].astype(str).str.extract(r"(P_\d+)", expand=False)
        full_df["participant_id"] = full_df["participant_id"].fillna(full_df["patient_id"].astype(str))

        self.manifest = full_df.dropna(subset=["image_path", "mask_path"]).copy()

        self.manifest = self.manifest[
            [
                "patient_id",
                "participant_id",
                "image_path",
                "mask_path",
                "image view",
                "label",
                "abnormality type",
            ]
        ].rename(columns={"image view": "view", "abnormality type": "abnormality_type"})

        os.makedirs(self.output_path, exist_ok=True)
        save_loc = os.path.join(self.output_path, "manifest.csv")
        self.manifest.to_csv(save_loc, index=False)
        return self.manifest

    def read_mask(self, mask_rel_path: str) -> Optional[np.ndarray]:
        full_path = os.path.join(self.base_path, mask_rel_path)
        mask = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return (mask_binary / 255.0).astype(np.float32)

    def segment_breast_otsu(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image

        _, binary_mask = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image.astype(np.float32)

        largest_contour = max(contours, key=cv2.contourArea)
        mask_clean = np.zeros_like(img_uint8)
        cv2.drawContours(mask_clean, [largest_contour], -1, 255, thickness=cv2.FILLED)
        img_masked = cv2.bitwise_and(img_uint8, img_uint8, mask=mask_clean)
        return img_masked.astype(np.float32) / 255.0

    def _is_breast_on_left(self, image: np.ndarray) -> bool:
        h, w = image.shape
        left_sum = np.sum(image[:, : w // 2])
        right_sum = np.sum(image[:, w // 2 :])
        return left_sum > right_sum

    def suppress_pectoral_muscle(self, image: np.ndarray, view: str) -> np.ndarray:
        if view != "MLO":
            return image

        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image

        is_left = self._is_breast_on_left(img_uint8)
        h, w = img_uint8.shape

        edges = cv2.Canny(img_uint8, 30, 100)
        roi_mask = np.zeros_like(edges)
        if is_left:
            roi_mask[0 : h // 2, 0 : w // 2] = 255
        else:
            roi_mask[0 : h // 2, w // 2 : w] = 255
        edges_roi = cv2.bitwise_and(edges, edges, mask=roi_mask)

        lines = cv2.HoughLinesP(
            edges_roi,
            rho=1,
            theta=np.pi / 180,
            threshold=25,
            minLineLength=30,
            maxLineGap=30,
        )
        if lines is None:
            return image

        best_line = None
        max_len = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 20 < angle < 85 and length > max_len:
                max_len = length
                best_line = (x1, y1, x2, y2)

        if best_line is None:
            return image

        x1, y1, x2, y2 = best_line
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

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        img_uint8 = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_uint8).astype(np.float32) / 255.0

    def extract_lesion_patch(self, image: np.ndarray, mask: np.ndarray, patch_size: int = 598) -> Tuple[np.ndarray, np.ndarray]:
        mask_uint8 = (mask * 255).astype(np.uint8)
        moments = cv2.moments(mask_uint8)
        if moments["m00"] == 0:
            cy, cx = image.shape[0] // 2, image.shape[1] // 2
        else:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

        half_size = patch_size // 2
        pad_width = ((half_size, half_size), (half_size, half_size))
        img_padded = np.pad(image, pad_width, mode="constant", constant_values=0)
        mask_padded = np.pad(mask, pad_width, mode="constant", constant_values=0)

        pad_cx = cx + half_size
        pad_cy = cy + half_size
        start_x = pad_cx - half_size
        end_x = start_x + patch_size
        start_y = pad_cy - half_size
        end_y = start_y + patch_size

        img_patch = img_padded[start_y:end_y, start_x:end_x]
        mask_patch = mask_padded[start_y:end_y, start_x:end_x]

        if img_patch.shape != (patch_size, patch_size):
            img_patch = cv2.resize(img_patch, (patch_size, patch_size))
            mask_patch = cv2.resize(mask_patch, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

        return img_patch, mask_patch

    def get_augmentation_pipeline(self):
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=175, border_mode=0, p=0.5),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, border_mode=0, p=0.5),
            ]
        )

    def apply_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        aug_pipeline = self.get_augmentation_pipeline()
        augmented = aug_pipeline(image=image, mask=mask)
        img_out = augmented["image"].astype(np.float32)
        mask_out = (augmented["mask"] > 0.5).astype(np.float32)
        return img_out, mask_out

    def preprocess_pair(self, row: pd.Series, do_aug: bool = False, patch_size: int = 598) -> Tuple[np.ndarray, np.ndarray]:
        full_img_path = os.path.join(self.base_path, row["image_path"])
        img_raw = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            raise FileNotFoundError(f"Image not found: {full_img_path}")
        img = img_raw.astype(np.float32) / 255.0

        mask = self.read_mask(row["mask_path"])
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {row['mask_path']}")
        if mask.shape != img.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)

        img = self.segment_breast_otsu(img)
        img = self.suppress_pectoral_muscle(img, row["view"])
        img = self.apply_clahe(img)
        img_patch, mask_patch = self.extract_lesion_patch(img, mask, patch_size=patch_size)

        if do_aug:
            img_patch, mask_patch = self.apply_augmentation(img_patch, mask_patch)

        img_patch = np.clip(img_patch, 0.0, 1.0).astype(np.float32)
        mask_patch = (mask_patch > 0.5).astype(np.float32)

        if img_patch.shape != (patch_size, patch_size):
            img_patch = cv2.resize(img_patch, (patch_size, patch_size))
        if mask_patch.shape != (patch_size, patch_size):
            mask_patch = cv2.resize(mask_patch, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
        mask_patch = (mask_patch > 0.5).astype(np.float32)

        return img_patch.astype(np.float32), mask_patch.astype(np.float32)

    def create_group_split(self, test_size: float = 0.15, val_size: float = 0.15, group_col: str = "participant_id"):
        if self.manifest is None:
            raise ValueError("Manifest not loaded. Call load_and_filter_manifest() first.")
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

    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        os.makedirs(self.output_path, exist_ok=True)
        train_df.to_csv(os.path.join(self.output_path, "train_manifest.csv"), index=False)
        val_df.to_csv(os.path.join(self.output_path, "val_manifest.csv"), index=False)
        test_df.to_csv(os.path.join(self.output_path, "test_manifest.csv"), index=False)

    def sanity_check(self, df: pd.DataFrame, n: int = 200):
        if df.empty:
            print("Sanity check skipped: empty DataFrame.")
            return
        sample_df = df.sample(n=min(n, len(df)), random_state=42)
        failures = 0
        empty_masks = 0
        area_ratios = []

        for _, row in sample_df.iterrows():
            try:
                _, mask = self.preprocess_pair(row, do_aug=False)
            except Exception:
                failures += 1
                continue

            area_ratio = float(mask.sum() / mask.size)
            area_ratios.append(area_ratio)
            if mask.sum() == 0:
                empty_masks += 1

        if area_ratios:
            mean_ratio = float(np.mean(area_ratios))
            min_ratio = float(np.min(area_ratios))
            max_ratio = float(np.max(area_ratios))
        else:
            mean_ratio = min_ratio = max_ratio = 0.0

        print(
            f"Sanity check: failures={failures}, empty_masks={empty_masks}, "
            f"mean_area_ratio={mean_ratio:.6f}, min_area_ratio={min_ratio:.6f}, max_area_ratio={max_ratio:.6f}"
        )

    def _save_qa_overlays(self, df: pd.DataFrame, n: int = 20):
        if df.empty:
            return
        qa_dir = os.path.join(self.output_path, "qa_overlays")
        os.makedirs(qa_dir, exist_ok=True)
        indices = df.index.tolist()
        random.Random(42).shuffle(indices)
        picked = indices[: min(n, len(indices))]

        for i, idx in enumerate(picked):
            row = df.loc[idx]
            try:
                img, mask = self.preprocess_pair(row, do_aug=False)
            except Exception:
                continue

            img_u8 = (img * 255).astype(np.uint8)
            bgr = np.stack([img_u8, img_u8, img_u8], axis=-1)
            mask_bool = mask > 0.5
            bgr[mask_bool, 2] = 255
            out_path = os.path.join(qa_dir, f"qa_{i:02d}_{row['patient_id']}.png")
            cv2.imwrite(out_path, bgr)


def run_pipeline():
    processor = CBISDDSM_Preprocessor()
    manifest = processor.load_and_filter_manifest()
    if manifest is None:
        return

    train_df, val_df, test_df = processor.create_group_split()
    processor.save_splits(train_df, val_df, test_df)

    print(
        f"Split concluded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )

    print("\n>> Sanity check: train")
    processor.sanity_check(train_df)
    print(">> Sanity check: val")
    processor.sanity_check(val_df)
    print(">> Sanity check: test")
    processor.sanity_check(test_df)

    print("\n>> Generating QA overlays...")
    processor._save_qa_overlays(train_df, n=20)

    print("\n>> Preprocessing finished.")


if __name__ == "__main__":
    run_pipeline()
