#!/usr/bin/env python3
"""
Trình Lấy Mẫu và Sinh Bệnh Nhân Ngẫu Nhiên (Random Patient Sampler & Biological Generator).

Cung cấp 2 chế độ:
(a) Real Test Cohort Sampler: Lấy ngẫu nhiên N bệnh nhân thực tế từ tập Test độc lập
    (kèm nhãn bệnh lý chuẩn vàng để người dùng đối chiếu tức thời xem mô hình đoán đúng hay sai).
(b) Biological Synthetic Generator: Sinh ngẫu nhiên N hồ sơ biểu hiện gen giả lập
    theo phân phối thống kê sinh học Pan-Cancer (log2 RSEM counts >= 0, biểu hiện marker đặc trưng).
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .cancer_types import (
    TCGA_33_CANCER_INFO,
    get_cancer_code_by_id,
    get_cancer_name_by_id,
)


class PatientSampler:
    """Bộ lấy mẫu bệnh nhân thực nghiệm và sinh bệnh nhân giả lập."""

    def __init__(
        self,
        demo_dir: Optional[Union[str, Path]] = None,
        preprocessors_dir: Optional[Union[str, Path]] = None,
        demo_data_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        resolved_demo = demo_dir if demo_dir is not None else demo_data_dir
        self.demo_dir = Path(resolved_demo) if resolved_demo is not None else base_dir / "demo_data"
        self.preprocessors_dir = (
            Path(preprocessors_dir)
            if preprocessors_dir is not None
            else base_dir / "models_pretrained" / "preprocessors"
        )

        # 1. Nạp kho mẫu thực tế 100 bệnh nhân
        demo_npz_path = self.demo_dir / "demo_patients_100.npz"
        demo_meta_path = self.demo_dir / "demo_patients_100_metadata.tsv"

        if demo_npz_path.exists():
            demo_data = np.load(demo_npz_path, allow_pickle=False)
            self._real_X: np.ndarray = demo_data["X_raw"]
            self._real_y: np.ndarray = demo_data["y_true"]
        else:
            self._real_X = np.empty((0, 16335), dtype=np.float64)
            self._real_y = np.empty((0,), dtype=np.int64)

        if demo_meta_path.exists():
            self._real_meta_df = pd.read_csv(demo_meta_path, sep="\t")
        else:
            self._real_meta_df = pd.DataFrame()

        # 2. Nạp thống kê sinh học để sinh dữ liệu giả lập
        prep_npz_path = self.preprocessors_dir / "preprocessors.npz"
        if prep_npz_path.exists():
            prep_npz = np.load(prep_npz_path)
            self.centering: np.ndarray = prep_npz["centering"].astype(np.float64).ravel()
            self.scalefactor: np.ndarray = prep_npz["scalefactor"].astype(np.float64).ravel()
        else:
            self.centering = np.zeros(16335, dtype=np.float64)
            self.scalefactor = np.ones(16335, dtype=np.float64)

        priors_json_path = self.preprocessors_dir / "class_priors.json"
        if priors_json_path.exists():
            priors_data = json.loads(priors_json_path.read_text(encoding="utf-8"))
            self.class_priors: np.ndarray = np.asarray(priors_data["priors"], dtype=np.float64).ravel()
        else:
            self.class_priors = np.ones(33, dtype=np.float64) / 33.0

        # Nạp danh sách gen
        gene_json_path = self.preprocessors_dir / "gene_names.json"
        if gene_json_path.exists():
            g_data = json.loads(gene_json_path.read_text(encoding="utf-8"))
            self.gene_names: List[str] = list(g_data["genes"] if isinstance(g_data, dict) else g_data)
        else:
            self.gene_names = [f"gene_{i}" for i in range(16335)]

    def sample_real_patients(
        self,
        n: int = 5,
        seed: Optional[int] = None,
        cancer_type: Optional[Union[str, int]] = None,
        replace: bool = False,
    ) -> Dict[str, Any]:
        """
        Lấy ngẫu nhiên N bệnh nhân từ tập Test thực tế (có sẵn nhãn thật 1..33).

        Args:
            n: Số lượng bệnh nhân muốn lấy.
            seed: Seed ngẫu nhiên để tái lập kết quả.
            cancer_type: Lọc theo loại ung thư (mã viết tắt ví dụ 'LUAD', 'BRCA' hoặc id số 1..33).
            replace: Cho phép lấy lặp lại mẫu hay không.

        Returns:
            Dict chứa X_raw (N, 16335), patient_ids, true_labels, true_acronyms, true_diseases, metadata.
        """
        if len(self._real_X) == 0:
            raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu thực tế tại {self.demo_dir}")

        rng = np.random.default_rng(seed)
        total_avail = len(self._real_X)

        # Lọc theo loại ung thư nếu có
        candidate_indices = np.arange(total_avail)
        if cancer_type is not None:
            if isinstance(cancer_type, str):
                code_upper = cancer_type.strip().upper()
                target_ids = [cid for cid, info in TCGA_33_CANCER_INFO.items() if info["code"] == code_upper]
                if not target_ids:
                    raise ValueError(f"Mã ung thư không tồn tại: '{cancer_type}'")
                target_id = target_ids[0]
            else:
                target_id = int(cancer_type)
            candidate_indices = np.where(self._real_y == target_id)[0]
            if len(candidate_indices) == 0:
                raise ValueError(f"Không tìm thấy bệnh nhân nào thuộc loại ung thư {cancer_type} trong tập demo.")

        num_avail = len(candidate_indices)
        if n > num_avail and not replace:
            actual_n = num_avail
        else:
            actual_n = max(1, int(n))

        selected = rng.choice(candidate_indices, size=actual_n, replace=replace or (actual_n > num_avail))

        X_sel = self._real_X[selected, :]
        y_sel = self._real_y[selected]

        patients_info = []
        for idx_in_demo in selected:
            if not self._real_meta_df.empty and idx_in_demo < len(self._real_meta_df):
                row = self._real_meta_df.iloc[idx_in_demo].to_dict()
                pid = str(row.get("patient_id", f"DEMO_PATIENT_{idx_in_demo:03d}"))
                sample_type = str(row.get("sample_type", "Primary Tumor"))
            else:
                pid = f"DEMO_PATIENT_{idx_in_demo:03d}"
                sample_type = "Primary Tumor"
            cid = int(self._real_y[idx_in_demo])
            patients_info.append(
                {
                    "patient_id": pid,
                    "true_label_id": cid,
                    "true_acronym": get_cancer_code_by_id(cid),
                    "true_disease": get_cancer_name_by_id(cid),
                    "sample_type": sample_type,
                    "source": "REAL_TEST_COHORT",
                }
            )

        return {
            "mode": "REAL_TEST_PATIENTS",
            "num_patients": actual_n,
            "X_raw": X_sel,
            "true_labels": y_sel.tolist(),
            "true_acronyms": [p["true_acronym"] for p in patients_info],
            "true_diseases": [p["true_disease"] for p in patients_info],
            "patient_ids": [p["patient_id"] for p in patients_info],
            "patients_metadata": patients_info,
        }

    def generate_synthetic_patients(
        self,
        n: int = 5,
        seed: Optional[int] = None,
        cancer_type: Optional[Union[str, int]] = None,
        biological_variation: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Sinh ngẫu nhiên N hồ sơ biểu hiện gen giả lập theo phân phối sinh học Pan-Cancer.

        Giá trị sinh ra tuân thủ miền giá trị sinh học của RNA-Seq log2(norm_count + 1) >= 0.

        Args:
            n: Số lượng hồ sơ muốn sinh.
            seed: Seed ngẫu nhiên.
            cancer_type: Chọn đích danh 1 loại ung thư hoặc để None để lấy ngẫu nhiên theo tỷ lệ phổ biến.
            biological_variation: Hệ số biến thiên sinh học quanh kỳ vọng (mặc định 1.0).

        Returns:
            Dict chứa X_raw (N, 16335), patient_ids, simulated_target_labels, v.v.
        """
        rng = np.random.default_rng(seed)
        n_gen = max(1, int(n))
        D = len(self.centering)

        # Xác định nhãn bệnh mục tiêu cho từng bệnh nhân sinh ra
        if cancer_type is not None:
            if isinstance(cancer_type, str):
                code_upper = cancer_type.strip().upper()
                cids = [cid for cid, info in TCGA_33_CANCER_INFO.items() if info["code"] == code_upper]
                if not cids:
                    raise ValueError(f"Mã ung thư không hợp lệ: '{cancer_type}'")
                target_ids = np.full(n_gen, cids[0], dtype=np.int64)
            else:
                target_ids = np.full(n_gen, int(cancer_type), dtype=np.int64)
        else:
            # Lấy mẫu theo tỷ lệ tiên nghiệm class_priors của 33 loại ung thư
            target_ids = rng.choice(np.arange(1, 34), size=n_gen, p=self.class_priors)

        # Tính profile biểu hiện sinh học thực nghiệm cho từng lớp từ kho mẫu thực tế
        has_real_classes = len(self._real_X) > 0 and len(self._real_y) > 0
        class_profiles: Dict[int, np.ndarray] = {}
        if has_real_classes:
            for cid in range(1, 34):
                mask_c = (self._real_y == cid)
                if np.any(mask_c):
                    class_profiles[cid] = np.mean(self._real_X[mask_c], axis=0)

        # Nạp trọng số M2 SVM đa lớp để hỗ trợ hướng vector sinh học nếu thiếu profile thực
        weights_m2_path = self.preprocessors_dir.parent / "M2_msvm" / "model_weights.npz"
        has_m2_weights = False
        W_sig: Optional[np.ndarray] = None
        if weights_m2_path.exists():
            w_npz = np.load(weights_m2_path)
            if "W" in w_npz.files:
                W_sig = w_npz["W"][:D, :]  # (16335, 33) phủ đủ 33 lớp
                has_m2_weights = True

        X_synth = np.zeros((n_gen, D), dtype=np.float64)
        patients_info = []

        for i in range(n_gen):
            target_cid = int(target_ids[i])
            col_idx = target_cid - 1

            if target_cid in class_profiles:
                # Sinh quanh trung tâm phân phối biểu hiện gen thực tế của loại ung thư đó
                base_profile = class_profiles[target_cid]
                noise = rng.normal(0.0, float(biological_variation) * 0.25, size=D) * self.scalefactor
                x_raw = np.maximum(0.0, base_profile + noise)
            else:
                # Fallback: dùng centering + hướng vector hyperplane phân tách của SVM
                z = rng.normal(0.0, float(biological_variation), size=D)
                if has_m2_weights and W_sig is not None:
                    class_direction = W_sig[:, col_idx]
                    norm_dir = np.linalg.norm(class_direction) + 1e-12
                    z += (class_direction / norm_dir) * rng.uniform(3.5, 5.0)
                x_raw = np.maximum(0.0, self.centering + z * self.scalefactor)

            X_synth[i, :] = x_raw

            pid = f"SIM-TCGA-2026-{target_cid:02d}-{i+1:04d}"
            patients_info.append(
                {
                    "patient_id": pid,
                    "simulated_target_label_id": target_cid,
                    "simulated_target_acronym": get_cancer_code_by_id(target_cid),
                    "simulated_target_disease": get_cancer_name_by_id(target_cid),
                    "source": "BIOLOGICAL_SYNTHETIC_SIMULATOR",
                }
            )

        return {
            "mode": "BIOLOGICAL_SYNTHETIC",
            "num_patients": n_gen,
            "X_raw": X_synth,
            "simulated_target_labels": target_ids.tolist(),
            "simulated_target_acronyms": [p["simulated_target_acronym"] for p in patients_info],
            "simulated_target_diseases": [p["simulated_target_disease"] for p in patients_info],
            "patient_ids": [p["patient_id"] for p in patients_info],
            "patients_metadata": patients_info,
        }

    def load_patient_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Nạp tệp biểu hiện gen của 1 hoặc nhiều bệnh nhân (.json, .csv, .tsv, .gz, .npz),
        tự động trích xuất mã bệnh nhân (patient_id) và nhãn đối chiếu thực tế (nếu có).

        Returns:
            Dict chứa X_raw, patient_ids, ground_truth_map, patients_metadata, description.
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp đầu vào: {p}")
        name_lower = p.name.lower()

        patient_ids: List[str] = []
        ground_truth_map: Dict[str, Dict[str, Any]] = {}
        patients_metadata: List[Dict[str, Any]] = []
        X_raw: Any = None

        if name_lower.endswith((".json.gz", ".json")):
            import gzip
            if name_lower.endswith(".json.gz"):
                raw_data = json.loads(gzip.decompress(p.read_bytes()).decode("utf-8"))
            else:
                raw_data = json.loads(p.read_text(encoding="utf-8"))

            if isinstance(raw_data, list):
                dfs = []
                for idx, item in enumerate(raw_data):
                    meta = item.get("metadata", {}) if isinstance(item, dict) else {}
                    pid = str(meta.get("patient_id") or meta.get("sample_id") or item.get("patient_id") or f"PATIENT_{idx:04d}")
                    patient_ids.append(pid)
                    stype = str(meta.get("sample_type", "Clinical Sample"))
                    t_lid = meta.get("true_label_id_1based") or meta.get("true_label_id") or meta.get("label_id")
                    t_dis = meta.get("_primary_disease") or meta.get("disease")
                    t_acr = get_cancer_code_by_id(int(t_lid)) if t_lid is not None else None
                    if t_acr is None and t_dis:
                        for cid, inf in TCGA_33_CANCER_INFO.items():
                            if inf["name"].lower() == str(t_dis).lower() or inf["code"].lower() == str(t_dis).lower():
                                t_lid, t_acr, t_dis = cid, inf["code"], inf["name"]
                                break
                    if t_acr:
                        ground_truth_map[pid] = {
                            "true_label_id": t_lid,
                            "true_acronym": t_acr,
                            "true_disease": t_dis or get_cancer_name_by_id(int(t_lid)),
                            "sample_type": stype,
                        }
                    vals = item.get("expression_values", item)
                    if isinstance(item, dict) and "gene_names" in item:
                        dfs.append(pd.Series(vals, index=item["gene_names"], name=pid))
                    elif isinstance(vals, dict):
                        gene_dict = {k: v for k, v in vals.items() if k != "metadata"}
                        dfs.append(pd.Series(gene_dict, name=pid))
                    else:
                        dfs.append(pd.Series(vals, name=pid))
                X_raw = pd.DataFrame(dfs)
            elif isinstance(raw_data, dict):
                meta = raw_data.get("metadata", {})
                pid = str(meta.get("patient_id") or meta.get("sample_id") or raw_data.get("patient_id") or "PATIENT_0000")
                patient_ids.append(pid)
                stype = str(meta.get("sample_type", "Primary Tumor" if meta else "Clinical Sample"))
                t_lid = meta.get("true_label_id_1based") or meta.get("true_label_id") or meta.get("label_id")
                t_dis = meta.get("_primary_disease") or meta.get("disease")
                t_acr = get_cancer_code_by_id(int(t_lid)) if t_lid is not None else None
                if t_acr is None and t_dis:
                    for cid, inf in TCGA_33_CANCER_INFO.items():
                        if inf["name"].lower() == str(t_dis).lower() or inf["code"].lower() == str(t_dis).lower():
                            t_lid, t_acr, t_dis = cid, inf["code"], inf["name"]
                            break
                if t_acr:
                    ground_truth_map[pid] = {
                        "true_label_id": t_lid,
                        "true_acronym": t_acr,
                        "true_disease": t_dis or get_cancer_name_by_id(int(t_lid)),
                        "sample_type": stype,
                    }
                vals = raw_data.get("expression_values", raw_data)
                if "gene_names" in raw_data:
                    X_raw = pd.DataFrame([vals], columns=list(raw_data["gene_names"]), index=[pid])
                elif isinstance(vals, dict):
                    gene_dict = {k: v for k, v in vals.items() if k != "metadata"}
                    X_raw = pd.DataFrame([gene_dict], index=[pid])
                else:
                    X_raw = np.asarray(vals, dtype=np.float64).reshape(1, -1)

        elif name_lower.endswith(".npz"):
            npz_data = np.load(p, allow_pickle=False)
            X_raw = npz_data["X_raw"] if "X_raw" in npz_data else npz_data[npz_data.files[0]]
            N = X_raw.shape[0]

            meta_tsv = p.with_name(p.stem + "_metadata.tsv")
            meta_csv = p.with_name(p.stem + "_metadata.csv")
            meta_df = pd.DataFrame()
            if meta_tsv.exists():
                meta_df = pd.read_csv(meta_tsv, sep="\t")
            elif meta_csv.exists():
                meta_df = pd.read_csv(meta_csv)

            raw_sids = npz_data["sample_ids"] if "sample_ids" in npz_data else None
            raw_pids = npz_data["patient_ids"] if "patient_ids" in npz_data else None
            raw_y = npz_data["y_true"] if "y_true" in npz_data else None
            raw_dis = npz_data["diseases"] if "diseases" in npz_data else None

            for i in range(N):
                pid = ""
                if not meta_df.empty and i < len(meta_df):
                    row = meta_df.iloc[i].to_dict()
                    pid = str(row.get("patient_id") or row.get("sample_id") or f"PATIENT_{i:04d}")
                    stype = str(row.get("sample_type", "Primary Tumor"))
                    cid = int(row.get("label_id") or (raw_y[i] if raw_y is not None else 0))
                    dis = str(row.get("_primary_disease") or (raw_dis[i] if raw_dis is not None else get_cancer_name_by_id(cid)))
                else:
                    if raw_pids is not None:
                        pid = str(raw_pids[i])
                    elif raw_sids is not None:
                        pid = str(raw_sids[i])
                    else:
                        pid = f"PATIENT_{i:04d}"
                    stype = "Primary Tumor" if raw_y is not None else "Clinical Sample"
                    cid = int(raw_y[i]) if raw_y is not None else 0
                    dis = str(raw_dis[i]) if raw_dis is not None else get_cancer_name_by_id(cid)

                patient_ids.append(pid)
                if cid > 0:
                    ground_truth_map[pid] = {
                        "true_label_id": cid,
                        "true_acronym": get_cancer_code_by_id(cid),
                        "true_disease": dis,
                        "sample_type": stype,
                    }

        elif name_lower.endswith((".csv.gz", ".tsv.gz", ".csv", ".tsv", ".gz")):
            sep = "\t" if (".tsv" in name_lower) else ","
            df = pd.read_csv(p, sep=sep)
            N = len(df)

            base_stem = p.name.split(".")[0]
            meta_tsv = p.with_name(f"{base_stem}_metadata.tsv")
            meta_csv = p.with_name(f"{base_stem}_metadata.csv")
            meta_json = p.with_name(f"{base_stem}_metadata.json")
            meta_df = pd.DataFrame()
            if meta_tsv.exists():
                meta_df = pd.read_csv(meta_tsv, sep="\t")
            elif meta_csv.exists():
                meta_df = pd.read_csv(meta_csv)
            elif meta_json.exists():
                try:
                    j_meta = json.loads(meta_json.read_text(encoding="utf-8"))
                    meta_df = pd.DataFrame([j_meta] if isinstance(j_meta, dict) else j_meta)
                except Exception:
                    pass

            id_cols = [c for c in df.columns if str(c).lower() in ("patient_id", "sample_id", "sample", "case_id", "bcr_patient_barcode")]
            if id_cols:
                target_col = id_cols[0]
                patient_ids = [str(x) for x in df[target_col]]
                df = df.drop(columns=[target_col])
            elif not meta_df.empty and len(meta_df) == N:
                patient_ids = [str(row.get("patient_id") or row.get("sample_id") or f"PATIENT_{i:04d}") for i, row in meta_df.iterrows()]
            else:
                patient_ids = [f"PATIENT_{i:04d}" for i in range(N)]

            label_cols = [c for c in df.columns if str(c).lower() in ("label_id", "true_label_id", "cancer_type", "_primary_disease", "disease")]
            if label_cols:
                lbl_col = label_cols[0]
                for idx, val in enumerate(df[lbl_col]):
                    pid = patient_ids[idx]
                    cid = None
                    try:
                        cid = int(val)
                    except ValueError:
                        for c_id, inf in TCGA_33_CANCER_INFO.items():
                            if inf["code"].lower() == str(val).lower() or inf["name"].lower() == str(val).lower():
                                cid = c_id
                                break
                    if cid and cid in TCGA_33_CANCER_INFO:
                        ground_truth_map[pid] = {
                            "true_label_id": cid,
                            "true_acronym": get_cancer_code_by_id(cid),
                            "true_disease": get_cancer_name_by_id(cid),
                            "sample_type": "Tumor Sample",
                        }
                df = df.drop(columns=label_cols)
            elif not meta_df.empty and len(meta_df) == N:
                for idx, row in meta_df.iterrows():
                    pid = patient_ids[idx]
                    cid = int(row.get("label_id", row.get("true_label_id_1based", 0)))
                    if cid > 0:
                        ground_truth_map[pid] = {
                            "true_label_id": cid,
                            "true_acronym": get_cancer_code_by_id(cid),
                            "true_disease": str(row.get("_primary_disease", get_cancer_name_by_id(cid))),
                            "sample_type": str(row.get("sample_type", "Primary Tumor")),
                        }

            X_raw = df

        sample_types = {pid: ground_truth_map.get(pid, {}).get("sample_type", "Primary Tumor") for pid in patient_ids}

        return {
            "X_raw": X_raw,
            "patient_ids": patient_ids,
            "num_patients": len(patient_ids),
            "sample_types": sample_types,
            "ground_truth_map": ground_truth_map,
            "description": f"Tệp dữ liệu: {p.name} ({len(patient_ids)} bệnh nhân)",
        }
