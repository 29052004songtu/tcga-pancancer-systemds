#!/usr/bin/env python3
"""
Mô-đun Xử lý và Khảo sát Thực nghiệm Tính năng Khuyết Gen (Missing Gene Imputation & Scaling).

Cung cấp các chiến lược bù trừ toán học cho dữ liệu biểu hiện gen khi thiếu một phần gen:
1. `zero_fill`: Điền 0.0 trong không gian chuẩn hóa Z-score (tương đương điền kỳ vọng mu_train trong không gian thô).
2. `linear_scale`: Bù trừ tỷ lệ phủ gen 1/rho. Giúp duy trì biên phân chia lớp của MSVM/Linear Model
   và duy trì phân phối đầu vào qua BatchNorm/ReLU của Neural Network M3.
3. `sqrt_scale`: Bù trừ 1/sqrt(rho) dựa trên phương sai ngẫu nhiên.
4. `l2_renorm`: Tái chuẩn hóa vector mẫu theo chuẩn L2 ban đầu.
Kèm hệ thống phân cấp cảnh báo an toàn sinh học (Safety Threshold Guard).
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("tcga_missing_genes")


class MissingGenePolicy(str, Enum):
    ZERO_FILL = "zero_fill"
    LINEAR_SCALE = "linear_scale"
    SQRT_SCALE = "sqrt_scale"
    L2_RENORM = "l2_renorm"


class CoverageSafetyLevel(str, Enum):
    GREEN = "GREEN"      # >= 50% coverage: Độ tin cậy tối ưu (>96.5% Acc, >95% F1)
    YELLOW = "YELLOW"    # 25% - <50%: Rất tốt với 1/rho (>96.0% Acc, >94.5% F1), cảnh báo nhẹ
    ORANGE = "ORANGE"    # 10% - <25%: Thận trọng (94-96% Acc), M3 bắt đầu suy giảm
    RED = "RED"          # < 10%: Suy giảm mạnh (<90% Acc), nguy cơ chẩn đoán sai cao


def check_coverage_safety(coverage_rate: float) -> Dict[str, Any]:
    """
    Đánh giá độ an toàn chẩn đoán dựa trên tỷ lệ gen hiện diện (coverage rate in (0, 1]).
    Trả về cấp độ an toàn, khuyến nghị lâm sàng và thông điệp cảnh báo minh bạch.
    """
    cov = float(coverage_rate)
    if cov >= 0.50:
        level = CoverageSafetyLevel.GREEN
        is_safe = True
        warning = None
        recommendation = (
            f"Tỷ lệ phủ gen đạt {cov*100:.1f}% (>= 50%). Hệ thống hoạt động trong vùng tối ưu. "
            f"Độ chính xác dự kiến > 96.5%, Macro-F1 > 95.0%."
        )
    elif cov >= 0.25:
        level = CoverageSafetyLevel.YELLOW
        is_safe = True
        warning = (
            f"CẢNH BÁO MỨC VÀNG (Dữ liệu khuyết vừa phải): Chỉ có {cov*100:.1f}% số gen được cung cấp. "
            f"Thuật toán kích hoạt bù trừ tuyến tính 1/rho để giữ độ chính xác ~96.3%."
        )
        recommendation = (
            "Kết quả dự đoán vẫn đáng tin cậy cao, nhưng khuyến nghị đối chiếu thêm với Top-3 "
            "hoặc bổ sung panel gen đầy đủ nếu nghi ngờ ca bệnh hiếm."
        )
    elif cov >= 0.10:
        level = CoverageSafetyLevel.ORANGE
        is_safe = False
        warning = (
            f"CẢNH BÁO MỨC CAM (Dữ liệu khuyết sâu): Chỉ có {cov*100:.1f}% số gen (< 25%). "
            f"Mô hình mạng nơ-ron M3 bắt đầu suy giảm. Độ chính xác kỳ vọng ~94-95%."
        )
        recommendation = (
            "Khuyến nghị sử dụng mô hình M1_multiLogReg hoặc M6_lightweight_no_m5 thay vì M3 đơn lẻ. "
            "Cần sự tham vấn của chuyên gia giải phẫu bệnh."
        )
    else:
        level = CoverageSafetyLevel.RED
        is_safe = False
        warning = (
            f"CẢNH BÁO NGUY HIỂM ĐỎ (Dữ liệu thiếu quá nghiêm trọng): Chỉ có {cov*100:.1f}% số gen (< 10%). "
            f"Độ chính xác sụt giảm nhanh xuống dưới 90%. Nguy cơ chẩn đoán sai lệch cao!"
        )
        recommendation = (
            "Dữ liệu biểu hiện gen quá phân mảnh để đưa ra kết luận chẩn đoán phân loại Pan-Cancer 33 lớp. "
            "Vui lòng xét nghiệm giải trình tự bổ sung."
        )

    return {
        "coverage_rate": cov,
        "safety_level": level.value,
        "is_clinically_reliable": is_safe,
        "warning_message": warning,
        "clinical_recommendation": recommendation,
    }


def simulate_gene_dropout(
    X_f0: np.ndarray,
    coverage_rate: float,
    seed: Optional[int] = None,
    dropped_indices: Optional[Union[List[int], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Mô phỏng hiện tượng khuyết gen ngẫu nhiên hoặc theo danh sách chỉ số gen chỉ định.
    
    Args:
        X_f0: Ma trận biểu hiện gen đã chuẩn hóa (N, D).
        coverage_rate: Tỷ lệ gen giữ lại (0.0 < rate <= 1.0).
        seed: Seed sinh số ngẫu nhiên để tái lập kết quả.
        dropped_indices: Danh sách chỉ số cột gen chủ động ẩn/bỏ đi.

    Returns:
        (X_masked, present_mask, actual_coverage_rate)
    """
    N, D = X_f0.shape
    present_mask = np.ones((D,), dtype=bool)

    if dropped_indices is not None and len(dropped_indices) > 0:
        drop_idx = np.asarray(dropped_indices, dtype=np.int64)
        valid_drop = drop_idx[(drop_idx >= 0) & (drop_idx < D)]
        present_mask[valid_drop] = False

    if coverage_rate < 1.0:
        cov = max(1.0 / D, min(1.0, float(coverage_rate)))
        k_keep = max(1, int(round(cov * D)))
        rng = np.random.default_rng(seed)
        avail_indices = np.where(present_mask)[0]
        if len(avail_indices) > k_keep:
            chosen_indices = rng.choice(avail_indices, size=k_keep, replace=False)
            present_mask = np.zeros((D,), dtype=bool)
            present_mask[chosen_indices] = True
        elif len(avail_indices) <= k_keep and dropped_indices is None:
            chosen_indices = rng.choice(D, size=k_keep, replace=False)
            present_mask = np.zeros((D,), dtype=bool)
            present_mask[chosen_indices] = True

    actual_cov = float(np.sum(present_mask)) / float(D)
    X_masked = X_f0.copy()
    X_masked[:, ~present_mask] = 0.0

    return X_masked, present_mask, actual_cov


def apply_missing_gene_policy(
    X_masked: np.ndarray,
    present_mask: np.ndarray,
    policy: Union[MissingGenePolicy, str] = MissingGenePolicy.LINEAR_SCALE,
    X_f0_original_norm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Áp dụng thuật toán bù trừ toán học lên dữ liệu khuyết gen trong không gian chuẩn hóa.

    - `zero_fill`: Giữ nguyên giá trị 0.0 của các gen thiếu.
    - `linear_scale`: Nhân với hệ số co giãn 1 / rho (rho = tỷ lệ gen hiện diện).
    - `sqrt_scale`: Nhân với hệ số co giãn 1 / sqrt(rho).
    - `l2_renorm`: Tái chuẩn hóa norm hàng theo norm kỳ vọng.
    """
    pol_str = str(policy).lower()
    if pol_str.startswith("missinggenepolicy."):
        pol_str = pol_str.split(".", 1)[1].lower()

    D = X_masked.shape[1]
    rho = float(np.sum(present_mask)) / float(D)
    if rho >= 0.9999 or rho <= 0.0:
        return X_masked

    if pol_str == "zero_fill":
        return X_masked

    if pol_str == "linear_scale":
        return X_masked / rho

    if pol_str == "sqrt_scale":
        return X_masked / np.sqrt(rho)

    if pol_str == "l2_renorm":
        curr_norms = np.sqrt(np.sum(X_masked * X_masked, axis=1, keepdims=True) + 1e-12)
        if X_f0_original_norm is not None:
            target_norms = X_f0_original_norm
        else:
            # Norm kỳ vọng của D biến chuẩn N(0,1) là sqrt(D)
            target_norms = np.sqrt(float(D))
        scale = target_norms / curr_norms
        return X_masked * scale

    raise ValueError(f"Chiến lược xử lý khuyết gen không hợp lệ: '{policy}'")
