#!/usr/bin/env python3
"""
Động Cơ Suy Luận Độc Lập TCGA Pan-Cancer (Pretrained TCGA Live Inference Engine).

Hỗ trợ:
- Suy luận tức thời cho 1 bệnh nhân hoặc theo lô (batch) nhiều bệnh nhân.
- Hai chế độ thực thi:
  1) `native`: Tính toán vector hóa tốc độ cao bằng NumPy, khớp chuẩn xác 100% công thức SystemDS DML (độ trễ < 1ms/mẫu).
  2) `systemds`: Kích hoạt trực tiếp trình biên dịch và runtime của Apache SystemDS qua DMLScript.
- Trả về toàn diện: Bệnh ung thư dự đoán chính, bảng phân phối xác suất (%) của Top-K nguy cơ cao nhất
  và toàn bộ 33 loại bệnh ung thư, kèm thời gian suy luận (latency ms).
- Tích hợp kiểm soát khuyết gen (missing gene handling): bù trừ tỷ lệ phủ 1/rho, mô phỏng dropout và
  hệ thống cảnh báo an toàn sinh học đa tầng.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .cancer_types import (
    TCGA_33_CANCER_INFO,
    get_cancer_code_by_id,
    get_cancer_name_by_id,
)
from .dml_engine import SystemDSLiveInferenceRunner
from .missing_genes import (
    CoverageSafetyLevel,
    MissingGenePolicy,
    apply_missing_gene_policy,
    check_coverage_safety,
    simulate_gene_dropout,
)


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax ổn định số học tránh tràn số (Log-Sum-Exp shift)."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def apply_prior_adjustment(
    P_raw: np.ndarray,
    class_priors: np.ndarray,
    tau: float = 0.0,
) -> np.ndarray:
    """Khớp 100% `mp::apply_prior_adjustment` trong SystemDS DML."""
    if float(tau) == 0.0:
        return P_raw
    priors = np.asarray(class_priors, dtype=np.float64).reshape(1, -1)
    log_P = np.log(P_raw + 1e-15)
    log_priors = np.log(priors + 1e-15)
    adj_logits = log_P - float(tau) * log_priors
    return _stable_softmax(adj_logits)


_AUTO_DETECT = object()


class PretrainedTCGAInferenceEngine:
    """Động cơ suy luận chuẩn hóa Z-score/ANOVA cho 33 phân loại ung thư TCGA."""

    BASE_MODELS = (
        "M1_multiLogReg",
        "M2_msvm",
        "M2_msvm_reg100",
        "M3_mlp_lowrank",
        "M5_kernel_ridge_rbf",
        "M5_kernel_ridge_cosine",
    )

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        heavy_bundle_dir: Any = _AUTO_DETECT,
        dml_script_path: Optional[Union[str, Path]] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        self.models_dir = Path(models_dir) if models_dir is not None else base_dir / "models_pretrained"
        if heavy_bundle_dir is _AUTO_DETECT:
            cand1 = base_dir / "google_drive_heavy_bundle"
            cand2 = base_dir.parent / "google_drive_heavy_bundle"
            if cand1.exists():
                self.heavy_bundle_dir = cand1
            elif cand2.exists():
                self.heavy_bundle_dir = cand2
            else:
                self.heavy_bundle_dir = None
        elif heavy_bundle_dir is None:
            self.heavy_bundle_dir = None
        else:
            self.heavy_bundle_dir = Path(heavy_bundle_dir)
        self.dml_script_path = (
            Path(dml_script_path)
            if dml_script_path is not None
            else base_dir / "dml" / "predict_tcga.dml"
        )

        prep_dir = self.models_dir / "preprocessors"
        prep_npz = np.load(prep_dir / "preprocessors.npz")
        self.centering = prep_npz["centering"].astype(np.float64).reshape(1, -1)
        self.scalefactor = prep_npz["scalefactor"].astype(np.float64).reshape(1, -1)
        self.anova_indices = prep_npz["anova_selected_indices"].astype(np.int64).ravel()

        self.num_genes = int(self.centering.shape[1])
        self.num_classes = 33

        priors_json = json.loads((prep_dir / "class_priors.json").read_text(encoding="utf-8"))
        self.class_priors = np.asarray(priors_json["priors"], dtype=np.float64).reshape(1, -1)

        label_map_raw = json.loads((prep_dir / "label_mapping.json").read_text(encoding="utf-8"))
        self.id_to_disease: Dict[int, str] = {int(k): str(v) for k, v in label_map_raw.items()}

        gene_data = json.loads((prep_dir / "gene_names.json").read_text(encoding="utf-8"))
        self.gene_names: List[str] = list(gene_data["genes"] if isinstance(gene_data, dict) else gene_data)
        self._gene_name_set = set(self.gene_names)

        self._weights: Dict[str, Dict[str, np.ndarray]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        for mk in self.BASE_MODELS:
            mdir = self.models_dir / mk
            if (mdir / "model_weights.npz").exists() and (mdir / "model_config.json").exists():
                w_npz = np.load(mdir / "model_weights.npz")
                self._weights[mk] = {k: w_npz[k].astype(np.float64) for k in w_npz.files}
                self._configs[mk] = json.loads((mdir / "model_config.json").read_text(encoding="utf-8"))

        ens_cfg_path = self.models_dir / "M6_ensemble" / "ensemble_config.json"
        self.ensemble_config: Dict[str, Any] = (
            json.loads(ens_cfg_path.read_text(encoding="utf-8")) if ens_cfg_path.exists() else {}
        )
        self._m5_support_path = self._locate_m5_support()
        self._m5_support_mmap: Optional[np.ndarray] = None

        # Trình chạy trực tiếp Apache SystemDS DML
        self._dml_runner: Optional[SystemDSLiveInferenceRunner] = None

    def _locate_m5_support(self) -> Optional[Path]:
        if self.heavy_bundle_dir is None:
            return None
        cand = self.heavy_bundle_dir / "M5_support" / "Xn_train_l2norm.npy"
        if cand.exists():
            return cand
        return None

    def has_m5_support(self) -> bool:
        return self._m5_support_path is not None and self._m5_support_path.exists()

    def get_dml_runner(self) -> SystemDSLiveInferenceRunner:
        """Khởi tạo lười (lazy load) trình điều phối SystemDS DML."""
        if self._dml_runner is None:
            self._dml_runner = SystemDSLiveInferenceRunner(
                dml_script_path=self.dml_script_path,
                models_dir=self.models_dir,
            )
        return self._dml_runner

    def _get_m5_support_mmap(self, expected_rows: Optional[int] = None) -> np.ndarray:
        if not self.has_m5_support():
            raise FileNotFoundError(
                "Thiếu ma trận hỗ trợ M5 `Xn_train_l2norm.npy` (~944 MB). "
                "Vui lòng tải gói `google_drive_heavy_bundle/` theo hướng dẫn trong "
                "`google_drive_heavy_bundle/README.md`, hoặc dùng M1/M2/M3/`M6_lightweight_no_m5`."
            )
        if self._m5_support_mmap is None:
            mmap_arr = np.load(self._m5_support_path, mmap_mode="r")
            if mmap_arr.ndim != 2 or mmap_arr.shape[1] != self.num_genes:
                raise ValueError(
                    f"Kích thước `Xn_train_l2norm.npy` không hợp lệ: {mmap_arr.shape}, "
                    f"kỳ vọng (*, {self.num_genes})."
                )
            self._m5_support_mmap = mmap_arr
        if expected_rows is not None and self._m5_support_mmap.shape[0] != expected_rows:
            raise ValueError(
                f"Số hàng `Xn_train_l2norm.npy` ({self._m5_support_mmap.shape[0]}) "
                f"không khớp với Alpha ({expected_rows})."
            )
        return self._m5_support_mmap

    def available_models(self) -> Dict[str, bool]:
        avail: Dict[str, bool] = {}
        for mk in self.BASE_MODELS:
            is_m5 = mk.startswith("M5_")
            avail[mk] = (mk in self._weights) and (self.has_m5_support() if is_m5 else True)
        for ename, rec in self.ensemble_config.get("recipes", {}).items():
            members_ok = all(avail.get(m, False) for m in rec["members"])
            avail[ename] = members_ok
        target_auto = (
            self.ensemble_config.get("default_full_ensemble", "M6_low_ece_calibrated")
            if self.has_m5_support()
            else self.ensemble_config.get("default_lightweight_ensemble", "M6_lightweight_no_m5")
        )
        avail["M6_auto"] = bool(avail.get(target_auto, False))
        return avail

    def _require_model_loaded(self, model_key: str) -> None:
        if model_key not in self._weights or model_key not in self._configs:
            raise ValueError(
                f"Mô hình '{model_key}' chưa được nạp trọng số từ `{self.models_dir}`."
            )

    MODEL_ALIASES: Dict[str, str] = {
        "m1": "M1_multiLogReg",
        "m2": "M2_msvm",
        "m2_reg100": "M2_msvm_reg100",
        "m3": "M3_mlp_lowrank",
        "m5": "M5_kernel_ridge_rbf",
        "m5_rbf": "M5_kernel_ridge_rbf",
        "m5_cosine": "M5_kernel_ridge_cosine",
        "m6": "M6_auto",
    }

    def resolve_model_name(self, model_name: str) -> str:
        """Chuẩn hóa tên mô hình, ánh xạ alias và tự động giải quyết M6_auto."""
        cleaned = str(model_name).strip()
        lower = cleaned.lower()
        if lower in self.MODEL_ALIASES:
            cleaned = self.MODEL_ALIASES[lower]

        if cleaned in ("M6_auto", "M6"):
            return (
                self.ensemble_config.get("default_full_ensemble", "M6_low_ece_calibrated")
                if self.has_m5_support()
                else self.ensemble_config.get("default_lightweight_ensemble", "M6_lightweight_no_m5")
            )
        return cleaned

    def _resolve_dropped_gene_indices(
        self,
        dropped_genes: Optional[Union[List[Union[str, int]], np.ndarray, str]] = None,
    ) -> Optional[List[int]]:
        """Chuyển đổi danh sách tên gen hoặc chỉ số sang list các chỉ số cột."""
        if dropped_genes is None:
            return None
        if isinstance(dropped_genes, str):
            p = Path(dropped_genes)
            if p.exists() and p.is_file():
                txt = p.read_text(encoding="utf-8").strip()
                items = [item.strip() for item in re.split(r"[,\s\n]+", txt) if item.strip()]
            else:
                items = [item.strip() for item in dropped_genes.split(",") if item.strip()]
        elif isinstance(dropped_genes, np.ndarray):
            items = dropped_genes.tolist()
        else:
            items = list(dropped_genes)

        if not items:
            return None

        gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
        indices = []
        for it in items:
            if isinstance(it, (int, np.integer)):
                if 0 <= int(it) < self.num_genes:
                    indices.append(int(it))
            elif isinstance(it, str):
                if it in gene_to_idx:
                    indices.append(gene_to_idx[it])
                elif it.isdigit():
                    idx = int(it)
                    if 0 <= idx < self.num_genes:
                        indices.append(idx)
        return indices if indices else None

    def validate_and_align_input(
        self,
        X_raw: Any,
        allow_missing_genes: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kiểm tra và căn chỉnh dữ liệu đầu vào (hỗ trợ dict, Series, DataFrame, np.ndarray, CSV/JSON path).

        Returns:
            (arr_X, present_gene_mask)
        """
        # 1. Hỗ trợ đường dẫn tệp JSON, CSV, TSV
        if isinstance(X_raw, (str, Path)):
            p = Path(X_raw)
            if not p.exists():
                raise FileNotFoundError(f"Không tìm thấy tệp đầu vào: {p}")
            name_lower = p.name.lower()
            if name_lower.endswith(".json.gz"):
                import gzip
                data = json.loads(gzip.decompress(p.read_bytes()).decode("utf-8"))
                return self.validate_and_align_input(data, allow_missing_genes=allow_missing_genes)
            elif name_lower.endswith(".json"):
                data = json.loads(p.read_text(encoding="utf-8"))
                return self.validate_and_align_input(data, allow_missing_genes=allow_missing_genes)
            elif name_lower.endswith((".csv.gz", ".tsv.gz", ".csv", ".tsv", ".gz")):
                sep = "\t" if (".tsv" in name_lower) else ","
                df = pd.read_csv(p, sep=sep)
                return self.validate_and_align_input(df, allow_missing_genes=allow_missing_genes)
            elif name_lower.endswith(".npz"):
                npz_data = np.load(p)
                arr = npz_data["X_raw"] if "X_raw" in npz_data else npz_data[npz_data.files[0]]
                return self.validate_and_align_input(arr, allow_missing_genes=allow_missing_genes)

        present_mask = np.ones((self.num_genes,), dtype=bool)

        # 2. Xử lý Dict
        if isinstance(X_raw, dict):
            if "expression_values" in X_raw:
                vals = X_raw["expression_values"]
                if "gene_names" in X_raw:
                    df_tmp = pd.DataFrame([vals], columns=list(X_raw["gene_names"]))
                    return self.validate_and_align_input(df_tmp, allow_missing_genes=allow_missing_genes)
                X_raw = np.asarray(vals, dtype=np.float64).reshape(1, -1)
            else:
                df_tmp = pd.DataFrame([X_raw])
                return self.validate_and_align_input(df_tmp, allow_missing_genes=allow_missing_genes)

        # 3. Xử lý Series
        if isinstance(X_raw, pd.Series):
            X_raw = X_raw.to_frame().T

        # 4. Xử lý DataFrame
        if isinstance(X_raw, pd.DataFrame):
            if X_raw.shape[0] == 0:
                raise ValueError("Đầu vào rỗng (N=0 mẫu).")
            cols = [str(c) for c in X_raw.columns]
            is_positional_int = list(X_raw.columns) == list(range(X_raw.shape[1]))
            if is_positional_int:
                arr = X_raw.to_numpy(dtype=np.float64)
            else:
                col_set = set(cols)
                if len(cols) != len(col_set):
                    raise ValueError("DataFrame chứa tên cột gen bị trùng lặp.")
                if cols == self.gene_names:
                    arr = X_raw.to_numpy(dtype=np.float64)
                elif col_set == self._gene_name_set:
                    arr = X_raw.loc[:, self.gene_names].to_numpy(dtype=np.float64)
                else:
                    missing = [g for g in self.gene_names if g not in col_set]
                    if missing and not allow_missing_genes:
                        raise ValueError(
                            f"DataFrame thiếu {len(missing)}/{self.num_genes} gen yêu cầu "
                            f"(ví dụ: {missing[:5]}). Thiết lập allow_missing_genes=True nếu muốn suy luận khuyết gen."
                        )
                    # Vectorized căn chỉnh tốc độ cao (<2ms) tránh tràn bộ nhớ DataFrame
                    N = len(X_raw)
                    arr = np.repeat(self.centering, N, axis=0)
                    present_mask = np.zeros(self.num_genes, dtype=bool)
                    gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
                    for col in cols:
                        if col in gene_to_idx:
                            idx = gene_to_idx[col]
                            arr[:, idx] = X_raw[col].to_numpy(dtype=np.float64)
                            present_mask[idx] = True
        else:
            arr = np.asarray(X_raw, dtype=np.float64)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.num_genes:
            raise ValueError(
                f"Kỳ vọng đầu vào kích thước (N, {self.num_genes}), nhận được {arr.shape}."
            )
        if arr.shape[0] == 0:
            raise ValueError("Đầu vào rỗng (N=0 mẫu).")
        if not np.isfinite(arr).all():
            raise ValueError("Dữ liệu biểu hiện gen chứa giá trị NaN hoặc Inf.")

        return arr, present_mask

    def preprocess_f0(self, X_raw: np.ndarray, allow_missing_genes: bool = False) -> np.ndarray:
        X, _ = self.validate_and_align_input(X_raw, allow_missing_genes=allow_missing_genes)
        return (X - self.centering) / self.scalefactor

    def preprocess_f2_anova_5000(self, X_raw: np.ndarray) -> np.ndarray:
        return self.preprocess_f0(X_raw)[:, self.anova_indices]

    def _predict_m1(self, X_f0: np.ndarray, tau_override: Optional[float] = None) -> np.ndarray:
        self._require_model_loaded("M1_multiLogReg")
        w = self._weights["M1_multiLogReg"]
        cfg = self._configs["M1_multiLogReg"]
        B = w["B"]
        priors = w.get("class_priors", self.class_priors)
        D = X_f0.shape[1]
        beta = B[:D, :]
        if B.shape[0] > D:
            linear = X_f0 @ beta + B[D : D + 1, :]
        else:
            linear = X_f0 @ beta
        logits = np.hstack([linear, np.zeros((X_f0.shape[0], 1), dtype=np.float64)])
        P_raw = _stable_softmax(logits)
        tau = float(cfg["hyperparams"]["tau"] if tau_override is None else tau_override)
        return apply_prior_adjustment(P_raw, priors, tau)

    def _predict_m2(self, model_key: str, X_f0: np.ndarray, tau_override: Optional[float] = None) -> np.ndarray:
        self._require_model_loaded(model_key)
        w = self._weights[model_key]
        cfg = self._configs[model_key]
        W = w["W"]
        priors = w.get("class_priors", self.class_priors)
        D = X_f0.shape[1]
        if W.shape[0] > D:
            scores = X_f0 @ W[:D, :] + W[D : D + 1, :]
        else:
            scores = X_f0 @ W
        temp_T = max(float(cfg["hyperparams"].get("temp_T", 1.0)), 1e-12)
        P_raw = _stable_softmax(scores / temp_T)
        tau = float(cfg["hyperparams"]["tau"] if tau_override is None else tau_override)
        return apply_prior_adjustment(P_raw, priors, tau)

    def _predict_m3(self, X_f2: np.ndarray, tau_override: Optional[float] = None) -> np.ndarray:
        self._require_model_loaded("M3_mlp_lowrank")
        w = self._weights["M3_mlp_lowrank"]
        cfg = self._configs["M3_mlp_lowrank"]
        bn_enabled = str(cfg["hyperparams"].get("use_bn", "TRUE")).upper() == "TRUE"
        bn_eps = 1e-5

        o1 = (X_f2 @ w["U1"]) @ w["V1"] + w["b1"]
        if bn_enabled:
            bn1 = ((o1 - w["em1"]) / np.sqrt(w["ev1"] + bn_eps)) * w["g1"] + w["bt1"]
            a1 = np.maximum(0.0, bn1)
        else:
            a1 = np.maximum(0.0, o1)

        o2 = a1 @ w["W2"] + w["b2"]
        if bn_enabled:
            bn2 = ((o2 - w["em2"]) / np.sqrt(w["ev2"] + bn_eps)) * w["g2"] + w["bt2"]
            a2 = np.maximum(0.0, bn2)
        else:
            a2 = np.maximum(0.0, o2)

        logits = a2 @ w["W3"] + w["b3"]
        P_raw = _stable_softmax(logits)
        priors = w.get("class_priors", self.class_priors)
        tau = float(cfg["hyperparams"]["tau"] if tau_override is None else tau_override)
        return apply_prior_adjustment(P_raw, priors, tau)

    def _predict_m5(
        self,
        model_key: str,
        X_f0: np.ndarray,
        tau_override: Optional[float] = None,
        row_chunk_size: int = 2048,
    ) -> np.ndarray:
        if int(row_chunk_size) < 1:
            raise ValueError(f"`row_chunk_size` phải >= 1, nhận được {row_chunk_size}.")
        self._require_model_loaded(model_key)
        w = self._weights[model_key]
        cfg = self._configs[model_key]
        hp = cfg["hyperparams"]
        kernel_type = str(hp["kernel_type"])
        norm_rows = str(hp.get("normalize_rows", "TRUE")).upper() == "TRUE"
        temp_T = max(float(hp.get("temp_T", 0.1)), 1e-4)
        tau = float(hp["tau"] if tau_override is None else tau_override)

        if norm_rows:
            nrm = np.sqrt(np.sum(X_f0 * X_f0, axis=1, keepdims=True) + 1e-12)
            Xn = X_f0 / nrm
        else:
            Xn = X_f0 / np.sqrt(max(1.0, float(X_f0.shape[1])))

        Alpha = w["Alpha"]
        Xn_tr = self._get_m5_support_mmap(expected_rows=Alpha.shape[0])
        bias_offset = w["bias_offset"]
        priors = w.get("class_priors", self.class_priors)

        N = Xn.shape[0]
        N_tr = Xn_tr.shape[0]
        scores = np.repeat(bias_offset, N, axis=0)

        if kernel_type == "rbf":
            eff_gamma = float(w["eff_gamma"].ravel()[0])
            r2 = np.sum(Xn * Xn, axis=1, keepdims=True)
            r2_tr = w["r2_tr"].reshape(1, -1)
            for start in range(0, N_tr, row_chunk_size):
                end = min(N_tr, start + row_chunk_size)
                Xn_blk = np.asarray(Xn_tr[start:end], dtype=np.float64)
                S_blk = Xn @ Xn_blk.T
                D2_blk = np.maximum(0.0, (-2.0 * S_blk + r2) + r2_tr[:, start:end])
                K_blk = np.exp(-eff_gamma * D2_blk)
                scores += K_blk @ Alpha[start:end, :]
        else:
            for start in range(0, N_tr, row_chunk_size):
                end = min(N_tr, start + row_chunk_size)
                Xn_blk = np.asarray(Xn_tr[start:end], dtype=np.float64)
                S_blk = Xn @ Xn_blk.T
                scores += S_blk @ Alpha[start:end, :]

        P_raw = _stable_softmax(scores / temp_T)
        return apply_prior_adjustment(P_raw, priors, tau)

    def predict_proba(
        self,
        model_name: str,
        X_raw: Any,
        tau_override: Optional[float] = None,
        allow_missing_genes: bool = False,
        gene_coverage_rate: float = 1.0,
        gene_dropout_rate: float = 0.0,
        dropped_genes: Optional[Union[List[Union[str, int]], np.ndarray, str]] = None,
        missing_policy: Union[MissingGenePolicy, str] = MissingGenePolicy.LINEAR_SCALE,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Dự báo ma trận phân phối xác suất `(N, 33)`."""
        resolved_model = self.resolve_model_name(model_name)

        arr, mask = self.validate_and_align_input(X_raw, allow_missing_genes=allow_missing_genes)
        X_f0 = (arr - self.centering) / self.scalefactor

        drop_indices = self._resolve_dropped_gene_indices(dropped_genes)
        effective_coverage = float(gene_coverage_rate)
        if gene_dropout_rate > 0.0:
            effective_coverage = max(0.01, effective_coverage * (1.0 - float(gene_dropout_rate)))

        if effective_coverage < 1.0 or drop_indices is not None:
            X_f0, dropout_mask, _ = simulate_gene_dropout(
                X_f0,
                coverage_rate=effective_coverage,
                seed=seed,
                dropped_indices=drop_indices,
            )
            mask = mask & dropout_mask

        if not np.all(mask):
            X_f0 = apply_missing_gene_policy(X_f0, mask, policy=missing_policy)

        if resolved_model == "M1_multiLogReg":
            return self._predict_m1(X_f0, tau_override=tau_override)
        if resolved_model in ("M2_msvm", "M2_msvm_reg100"):
            return self._predict_m2(resolved_model, X_f0, tau_override=tau_override)
        if resolved_model == "M3_mlp_lowrank":
            return self._predict_m3(X_f0[:, self.anova_indices], tau_override=tau_override)
        if resolved_model in ("M5_kernel_ridge_rbf", "M5_kernel_ridge_cosine"):
            return self._predict_m5(resolved_model, X_f0, tau_override=tau_override)

        recipes = self.ensemble_config.get("recipes", {})
        if resolved_model in recipes:
            rec = recipes[resolved_model]
            members = rec["members"]
            weights = rec["weights"]
            tau_ens = float(rec.get("tau", 0.0) if tau_override is None else tau_override)
            P_blend = np.zeros((X_f0.shape[0], self.num_classes), dtype=np.float64)
            for mk, wt in zip(members, weights):
                if mk == "M1_multiLogReg":
                    Pm = self._predict_m1(X_f0)
                elif mk in ("M2_msvm", "M2_msvm_reg100"):
                    Pm = self._predict_m2(mk, X_f0)
                elif mk == "M3_mlp_lowrank":
                    Pm = self._predict_m3(X_f0[:, self.anova_indices])
                elif mk in ("M5_kernel_ridge_rbf", "M5_kernel_ridge_cosine"):
                    Pm = self._predict_m5(mk, X_f0)
                else:
                    raise ValueError(f"Unknown ensemble member: {mk}")
                P_blend += float(wt) * Pm
            P_blend = P_blend / np.maximum(np.sum(P_blend, axis=1, keepdims=True), 1e-12)
            return apply_prior_adjustment(P_blend, self.class_priors, tau_ens)

        raise ValueError(f"Mô hình không hợp lệ hoặc chưa nạp: '{resolved_model}'.")

    def predict(
        self,
        model_name: str,
        X_raw: Any,
        top_k: int = 3,
        tau_override: Optional[float] = None,
        execution_mode: str = "native",
        allow_missing_genes: bool = False,
        gene_coverage_rate: float = 1.0,
        gene_dropout_rate: float = 0.0,
        dropped_genes: Optional[Union[List[Union[str, int]], np.ndarray, str]] = None,
        missing_policy: Union[MissingGenePolicy, str] = MissingGenePolicy.LINEAR_SCALE,
        seed: Optional[int] = None,
        patient_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dự đoán toàn diện cho 1 hoặc nhiều bệnh nhân:
        Trả về:
        - Bệnh ung thư dự đoán chính (label, acronym, name).
        - Phân phối xác suất Top-K và toàn bộ 33 lớp (% tỷ lệ).
        - Thời gian suy luận (latency ms).
        - Cảnh báo an toàn dữ liệu nếu thiếu gen.
        """
        t0 = time.time()
        mode_used = execution_mode.lower()
        resolved_model = self.resolve_model_name(model_name)

        arr, mask = self.validate_and_align_input(X_raw, allow_missing_genes=allow_missing_genes)

        drop_indices = self._resolve_dropped_gene_indices(dropped_genes)
        effective_coverage = float(gene_coverage_rate)
        if gene_dropout_rate > 0.0:
            effective_coverage = max(0.01, effective_coverage * (1.0 - float(gene_dropout_rate)))

        if effective_coverage < 1.0 or drop_indices is not None:
            _, dropout_mask, _ = simulate_gene_dropout(
                arr,
                coverage_rate=effective_coverage,
                seed=seed,
                dropped_indices=drop_indices,
            )
            mask = mask & dropout_mask

        actual_cov = float(np.sum(mask)) / float(self.num_genes)
        safety_audit = check_coverage_safety(actual_cov)

        if mode_used == "systemds":
            runner = self.get_dml_runner()
            arr_dml = arr.copy()
            # Điền kỳ vọng centering tại các gen thiếu để (C - C) / S = 0.0
            arr_dml[:, ~mask] = self.centering[0, ~mask]

            scale_factor = 1.0
            pol_str = str(missing_policy).lower()
            if pol_str.startswith("missinggenepolicy."):
                pol_str = pol_str.split(".", 1)[1].lower()
            if actual_cov < 0.9999 and actual_cov > 0.0:
                if pol_str == "linear_scale":
                    scale_factor = 1.0 / actual_cov
                elif pol_str in ("sqrt_scale", "l2_renorm"):
                    scale_factor = 1.0 / np.sqrt(actual_cov)
                elif pol_str == "zero_fill":
                    scale_factor = 1.0

            tau_val = float(0.0 if tau_override is None else tau_override)
            P, _, raw_latency_ms = runner.run_inference_dml(
                arr_dml,
                model_name=resolved_model,
                tau=tau_val,
                scale_factor=scale_factor,
            )
            N_samples = P.shape[0]
            latency_ms = float(raw_latency_ms) / max(1, N_samples)
        else:
            P = self.predict_proba(
                model_name=resolved_model,
                X_raw=arr,
                tau_override=tau_override,
                allow_missing_genes=True,
                gene_coverage_rate=effective_coverage,
                gene_dropout_rate=0.0,
                dropped_genes=drop_indices,
                missing_policy=missing_policy,
                seed=seed,
            )
            N_samples = P.shape[0]
            latency_ms = ((time.time() - t0) * 1000.0) / max(1, N_samples)

        results: List[Dict[str, Any]] = []
        k_use = max(1, min(int(top_k), self.num_classes))
        N_samples = P.shape[0]

        for i in range(N_samples):
            probs = P[i]
            order = np.argsort(-probs)
            top_id = int(order[0]) + 1
            top_disease = self.id_to_disease.get(
                top_id, TCGA_33_CANCER_INFO.get(top_id, {}).get("name", f"Class_{top_id}")
            )
            acronym = get_cancer_code_by_id(top_id)

            # Phân phối Top-K
            top_list = []
            for rank_idx, j in enumerate(order[:k_use]):
                cid = int(j) + 1
                dis = self.id_to_disease.get(
                    cid, TCGA_33_CANCER_INFO.get(cid, {}).get("name", f"Class_{cid}")
                )
                acr = get_cancer_code_by_id(cid)
                prob_val = float(probs[j])
                top_list.append(
                    {
                        "rank": rank_idx + 1,
                        "label_id": cid,
                        "acronym": acr,
                        "disease": dis,
                        "probability": prob_val,
                        "percentage": round(prob_val * 100.0, 3),
                    }
                )

            # Toàn bộ 33 loại ung thư
            all_33 = []
            prob_map = {}
            for j in order:
                cid = int(j) + 1
                dis = self.id_to_disease.get(
                    cid, TCGA_33_CANCER_INFO.get(cid, {}).get("name", f"Class_{cid}")
                )
                acr = get_cancer_code_by_id(cid)
                prob_val = float(probs[j])
                pct_val = round(prob_val * 100.0, 3)
                prob_map[acr] = pct_val
                all_33.append(
                    {
                        "label_id": cid,
                        "acronym": acr,
                        "disease": dis,
                        "probability": prob_val,
                        "percentage": pct_val,
                    }
                )

            pid = None
            if patient_ids and i < len(patient_ids):
                pid = str(patient_ids[i])
            else:
                pid = f"PATIENT_{i:04d}"

            results.append(
                {
                    "sample_index": i,
                    "patient_id": pid,
                    "predicted_label_id": top_id,
                    "predicted_acronym": acronym,
                    "predicted_disease": top_disease,
                    "confidence": float(probs[order[0]]),
                    "confidence_pct": round(float(probs[order[0]]) * 100.0, 3),
                    "inference_time_ms": round(latency_ms / max(1, N_samples), 3),
                    "execution_mode": mode_used,
                    "top_k": top_list,
                    "all_33_distribution": all_33,
                    "probabilities": prob_map,
                    "coverage_audit": safety_audit,
                }
            )

        return results

    def predict_patient(
        self,
        model_name: str,
        patient_data: Any,
        top_k: int = 5,
        execution_mode: str = "native",
        allow_missing_genes: bool = True,
        gene_coverage_rate: float = 1.0,
        gene_dropout_rate: float = 0.0,
        dropped_genes: Optional[Union[List[Union[str, int]], np.ndarray, str]] = None,
        missing_policy: Union[MissingGenePolicy, str] = MissingGenePolicy.LINEAR_SCALE,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Dự đoán tức thời cho 1 bệnh nhân đơn lẻ."""
        preds = self.predict(
            model_name=model_name,
            X_raw=patient_data,
            top_k=top_k,
            execution_mode=execution_mode,
            allow_missing_genes=allow_missing_genes,
            gene_coverage_rate=gene_coverage_rate,
            gene_dropout_rate=gene_dropout_rate,
            dropped_genes=dropped_genes,
            missing_policy=missing_policy,
            seed=seed,
        )
        return preds[0]
