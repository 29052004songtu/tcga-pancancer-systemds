#!/usr/bin/env python3
"""
TCGA Pan-Cancer Standalone Genomic Diagnostic & Research Web Application
========================================================================
Giao diện Web tương tác chuyên nghiệp xây dựng trên Streamlit:
1. Dự đoán nguy cơ 33 loại bệnh ung thư từ dữ liệu biểu hiện gen RNA-Seq.
2. Hỗ trợ tải tệp (.csv, .tsv, .json, .npz, .gz) HOẶC lấy mẫu ngẫu nhiên tức thời.
3. Cấu hình kiểm nghiệm độ bền khi khuyết gen (missing gene policy & dropout simulation).
4. Biểu đồ xác suất trực quan, đối chiếu chẩn đoán thực tế, tải báo cáo .csv/.json.
5. Xem trọn vẹn 17 bảng Benchmark chuẩn và 13 biểu đồ thực nghiệm độ phân giải 300 DPI.
6. Thông số biên dịch và tối ưu hóa hiệu năng Apache SystemDS.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import datetime
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Thiết lập đường dẫn độc lập không hardcode
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from release_staging.tcga_inference.cancer_types import (
        TCGA_33_CANCER_INFO,
        get_cancer_code_by_id,
        get_cancer_name_by_id,
        get_id_by_cancer_code,
    )
    from release_staging.tcga_inference.engine import PretrainedTCGAInferenceEngine
    from release_staging.tcga_inference.missing_genes import (
        CoverageSafetyLevel,
        MissingGenePolicy,
    )
    from release_staging.tcga_inference.sampler import PatientSampler
except ImportError:
    from tcga_inference.cancer_types import (
        TCGA_33_CANCER_INFO,
        get_cancer_code_by_id,
        get_cancer_name_by_id,
        get_id_by_cancer_code,
    )
    from tcga_inference.engine import PretrainedTCGAInferenceEngine
    from tcga_inference.missing_genes import (
        CoverageSafetyLevel,
        MissingGenePolicy,
    )
    from tcga_inference.sampler import PatientSampler

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="TCGA Pan-Cancer Diagnostic & Research Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho giao diện y sinh hiện đại
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .badge-green {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 12px;
        border: 1px solid #86EFAC;
    }
    .badge-yellow {
        background-color: #FEF9C3;
        color: #854D0E;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 12px;
        border: 1px solid #FDE047;
    }
    .badge-red {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 12px;
        border: 1px solid #FCA5A5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        padding: 10px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_engine() -> PretrainedTCGAInferenceEngine:
    """Tải và lưu trữ bộ nhớ đệm cho Inference Engine."""
    models_dir = _CURRENT_DIR / "models_pretrained"
    return PretrainedTCGAInferenceEngine(models_dir=models_dir)


@st.cache_resource(show_spinner=False)
def load_sampler() -> PatientSampler:
    """Tải và lưu trữ bộ nhớ đệm cho Patient Sampler."""
    demo_dir = _CURRENT_DIR / "demo_data"
    prep_dir = _CURRENT_DIR / "models_pretrained" / "preprocessors"
    return PatientSampler(demo_dir=demo_dir, preprocessors_dir=prep_dir)


def prepare_csv_download(predictions: List[Dict[str, Any]]) -> str:
    """Tạo chuỗi CSV hoàn chỉnh từ danh sách kết quả dự đoán."""
    rows: List[Dict[str, Any]] = []
    for p in predictions:
        row: Dict[str, Any] = {
            "sample_index": p.get("sample_index", 0),
            "patient_id": p.get("patient_id", ""),
            "predicted_label_id": p.get("predicted_label_id", 0),
            "predicted_acronym": p.get("predicted_acronym", ""),
            "predicted_disease": p.get("predicted_disease", ""),
            "confidence_pct": p.get("confidence_pct", 0.0),
            "true_label_id": p.get("true_label_id", ""),
            "true_acronym": p.get("true_acronym", ""),
            "true_disease": p.get("true_disease", ""),
            "is_match": p.get("is_match", ""),
            "inference_time_ms": p.get("inference_time_ms", 0.0),
            "gene_coverage_rate": p.get("coverage_audit", {}).get("coverage_rate", 1.0),
            "safety_level": p.get("coverage_audit", {}).get("safety_level", "GREEN"),
            "is_clinically_reliable": p.get("coverage_audit", {}).get("is_clinically_reliable", True),
            "missing_warning": p.get("coverage_audit", {}).get("warning_message", ""),
        }
        for k_idx, item in enumerate(p.get("top_k", [])[:5], start=1):
            row[f"top{k_idx}_acronym"] = item.get("acronym", "")
            row[f"top{k_idx}_pct"] = item.get("percentage", 0.0)

        probs = p.get("probabilities", {})
        for cid in sorted(TCGA_33_CANCER_INFO.keys()):
            code = TCGA_33_CANCER_INFO[cid]["code"]
            row[f"prob_{code}_pct"] = probs.get(code, 0.0)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def main() -> None:
    engine = load_engine()
    sampler = load_sampler()

    # Tiêu đề giao diện chính
    st.markdown(
        """
        <div class="main-header">🧬 TCGA Pan-Cancer Diagnostic & Research Platform</div>
        <div class="sub-header">
            Hệ thống suy luận và mô phỏng phân loại 33 bệnh ung thư TCGA từ biểu hiện gen transcriptomic — Tối ưu hóa bởi Apache SystemDS
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # SIDEBAR: Cấu hình và điều khiển
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Thiết lập Suy luận")

        # 1. Nguồn dữ liệu
        st.subheader("1. Nguồn Dữ liệu Bệnh nhân")
        data_mode = st.radio(
            "Chọn phương thức nạp dữ liệu:",
            [
                "Lấy mẫu ngẫu nhiên (Random Sampler)",
                "Tải tệp dữ liệu lên (Upload File)",
            ],
            index=0,
        )

        input_data_for_engine: Any = None
        patient_ids_for_engine: Optional[List[str]] = None
        ground_truth_dict: Dict[str, Dict[str, Any]] = {}
        sample_description = ""

        if data_mode == "Lấy mẫu ngẫu nhiên (Random Sampler)":
            if st.session_state.get("use_demo_sample_state", False):
                st.session_state["use_demo_sample_state"] = False
            sample_source = st.selectbox(
                "Nguồn mẫu ngẫu nhiên:",
                [
                    "Kho mẫu thực tế TCGA Test (100 bệnh nhân chuẩn có nhãn)",
                    "Mô phỏng sinh học phân phối TCGA (Synthetic Patients)",
                ],
            )
            n_patients = st.slider("Số lượng bệnh nhân:", min_value=1, max_value=50, value=5, step=1)
            cancer_filter = st.selectbox(
                "Lọc loại ung thư cụ thể (Tùy chọn):",
                ["Tất cả 33 loại"] + [f"{info['code']} - {info['name']}" for cid, info in sorted(TCGA_33_CANCER_INFO.items())],
            )
            selected_cancer_code = None if cancer_filter == "Tất cả 33 loại" else cancer_filter.split(" - ")[0]
            random_seed = st.number_input("Random Seed:", min_value=1, max_value=999999, value=42, step=1)

            # Lấy mẫu
            if "thực tế" in sample_source:
                sample_data = sampler.sample_real_patients(
                    n=n_patients,
                    seed=int(random_seed),
                    cancer_type=selected_cancer_code,
                )
                input_data_for_engine = sample_data["X_raw"]
                patient_ids_for_engine = sample_data["patient_ids"]
                sample_description = f"Kho mẫu Test thực tế ({len(patient_ids_for_engine)} bệnh nhân)"
                for info in sample_data["patients_metadata"]:
                    ground_truth_dict[info["patient_id"]] = {
                        "true_label_id": info["true_label_id"],
                        "true_acronym": info["true_acronym"],
                        "true_disease": info["true_disease"],
                        "sample_type": info.get("sample_type", "Primary Tumor"),
                    }
            else:
                sample_data = sampler.generate_synthetic_patients(
                    n=n_patients,
                    seed=int(random_seed),
                    cancer_type=selected_cancer_code,
                    biological_variation=1.0,
                )
                input_data_for_engine = sample_data["X_raw"]
                patient_ids_for_engine = sample_data["patient_ids"]
                sample_description = f"Mô phỏng sinh học TCGA ({len(patient_ids_for_engine)} bệnh nhân)"
                for info in sample_data["patients_metadata"]:
                    ground_truth_dict[info["patient_id"]] = {
                        "true_label_id": info["simulated_target_label_id"],
                        "true_acronym": info["simulated_target_acronym"],
                        "true_disease": info["simulated_target_disease"],
                        "sample_type": "Synthetic Simulated Profile",
                    }

        else:
            uploaded_file = st.file_uploader(
                "Tải lên tệp biểu hiện gen (.csv, .tsv, .json, .npz, .gz):",
                type=["csv", "tsv", "json", "npz", "gz"],
            )
            col_demo_btn, col_demo_txt = st.columns([1, 1])
            with col_demo_btn:
                use_demo_sample = st.button("📁 Dùng mẫu demo có sẵn", use_container_width=True)

            if uploaded_file is not None:
                import tempfile
                file_bytes = uploaded_file.read()
                fname = uploaded_file.name
                temp_dir = Path(tempfile.gettempdir()) / "tcga_web_uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / f"upload_{fname}"
                temp_path.write_bytes(file_bytes)
                try:
                    loaded = sampler.load_patient_file(temp_path)
                    input_data_for_engine = loaded["X_raw"]
                    patient_ids_for_engine = loaded["patient_ids"]
                    ground_truth_dict = loaded["ground_truth_map"]
                    sample_description = loaded["description"]
                except Exception as ex:
                    st.error(f"Lỗi đọc tệp tải lên: {ex}")
                    input_data_for_engine = None
            elif use_demo_sample or st.session_state.get("use_demo_sample_state", False):
                st.session_state["use_demo_sample_state"] = True
                demo_single = _CURRENT_DIR / "demo_data" / "demo_patient_single.json"
                loaded = sampler.load_patient_file(demo_single)
                input_data_for_engine = loaded["X_raw"]
                patient_ids_for_engine = loaded["patient_ids"]
                ground_truth_dict = loaded["ground_truth_map"]
                sample_description = "Mẫu demo đơn bệnh nhân (demo_patient_single.json)"
                st.info("Đang sử dụng dữ liệu mẫu `demo_patient_single.json`.")

        st.divider()

        # 2. Cấu hình khuyết gen & độ bền
        st.subheader("2. Cấu hình Thiếu Gen & Bù trừ")
        coverage_pct = st.slider(
            "Tỷ lệ % gen đưa vào suy luận (Coverage %):",
            min_value=10,
            max_value=100,
            value=100,
            step=5,
            help="Giả lập tình huống xét nghiệm giải trình tự chỉ bao phủ một phần transcriptome.",
        )
        dropout_pct = st.slider(
            "Tỷ lệ % gen ngẫu nhiên bỏ bớt (Random Dropout %):",
            min_value=0,
            max_value=90,
            value=0,
            step=5,
            help="Mô phỏng gen bị đột biến mất biểu hiện hoặc suy giảm tín hiệu giải trình tự.",
        )
        dropped_genes_str = st.text_input(
            "Bỏ các gen chỉ định (Cách nhau bằng dấu phẩy):",
            value="",
            placeholder="Ví dụ: TP53, KRAS, EGFR, BRAF",
            help="Loại trừ đích danh các gen chỉ định để kiểm tra tính độc lập của mô hình.",
        )
        policy_choice = st.selectbox(
            "Chiến lược bù trừ khuyết gen (Policy):",
            ["linear_scale", "zero_fill", "sqrt_scale", "l2_renorm"],
            index=0,
            help="linear_scale: Tái chuẩn hóa năng lượng tín hiệu 1/rho (Khuyên dùng - Đạt Top-1 Accuracy 95.8% ở Prompt 3).",
        )

        st.divider()

        # 3. Cấu hình mô hình
        st.subheader("3. Mô hình & Tham số")
        model_choice = st.selectbox(
            "Mô hình chẩn đoán:",
            [
                "M6_auto (Ensemble tự thích ứng - Khuyên dùng)",
                "M1_multiLogReg (Softmax Logistic Regression)",
                "M2_msvm_reg100 (Multi-class Support Vector Machine)",
                "M3_mlp_lowrank (Deep Low-Rank Neural Network)",
                "M6_lightweight_no_m5 (Tri-Ensemble siêu nhẹ)",
            ],
            index=0,
        )
        resolved_model_name = model_choice.split(" ")[0]

        execution_mode = st.radio(
            "Động cơ thực thi:",
            ["native (NumPy Vectorized < 1ms)", "systemds (Apache SystemDS DML)"],
            index=0,
        )
        exec_mode_val = "native" if "native" in execution_mode else "systemds"

        top_k_val = st.slider("Số lượng Top-K bệnh hiển thị:", min_value=1, max_value=10, value=5)

        run_btn = st.button("🚀 Chạy Dự Đoán Lâm Sàng", type="primary", use_container_width=True)

    # =========================================================================
    # MAIN AREA: TABS GIAO DIỆN
    # =========================================================================
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔬 Dự Đoán & Phân Tích Bệnh Nhân",
            "📊 Bảng Điểm Benchmark 33 Ung Thư",
            "📈 Biểu Đồ Huấn Luyện & Đánh Giá 300 DPI",
            "⚙️ Thông Số & Tối Ưu Apache SystemDS",
        ]
    )

    # TAB 1: DỰ ĐOÁN & PHÂN TÍCH BỆNH NHÂN
    with tab1:
        # Tự động chạy lần đầu nếu chưa chạy
        if "last_predictions" not in st.session_state or run_btn:
            if input_data_for_engine is None:
                # Tự động nạp 5 bệnh nhân ngẫu nhiên
                sample_data = sampler.sample_real_patients(n=5, seed=42)
                input_data_for_engine = sample_data["X_raw"]
                patient_ids_for_engine = sample_data["patient_ids"]
                sample_description = "Kho mẫu Test thực tế (5 bệnh nhân mặc định)"
                for info in sample_data["patients_metadata"]:
                    ground_truth_dict[info["patient_id"]] = {
                        "true_label_id": info["true_label_id"],
                        "true_acronym": info["true_acronym"],
                        "true_disease": info["true_disease"],
                        "sample_type": info.get("sample_type", "Primary Tumor"),
                    }

            try:
                with st.spinner("Đang thực hiện suy luận transcriptomic..."):
                    preds = engine.predict(
                        model_name=resolved_model_name,
                        X_raw=input_data_for_engine,
                        top_k=top_k_val,
                        patient_ids=patient_ids_for_engine,
                        execution_mode=exec_mode_val,
                        gene_coverage_rate=float(coverage_pct) / 100.0,
                        gene_dropout_rate=float(dropout_pct) / 100.0,
                        dropped_genes=dropped_genes_str if dropped_genes_str.strip() else None,
                        missing_policy=policy_choice,
                        allow_missing_genes=True,
                        seed=42,
                    )

                    # Bổ sung thông tin ground truth nếu có
                    for p in preds:
                        pid = p["patient_id"]
                        if pid in ground_truth_dict:
                            gt = ground_truth_dict[pid]
                            p["true_label_id"] = gt["true_label_id"]
                            p["true_acronym"] = gt["true_acronym"]
                            p["true_disease"] = gt["true_disease"]
                            p["sample_type"] = gt.get("sample_type", "Tumor")
                            p["is_match"] = bool(p["predicted_acronym"] == gt["true_acronym"])
                        else:
                            p["true_label_id"] = None
                            p["true_acronym"] = None
                            p["true_disease"] = None
                            p["sample_type"] = "Clinical Sample"
                            p["is_match"] = None

                    st.session_state["last_predictions"] = preds
                    st.session_state["last_desc"] = sample_description
                    st.session_state["last_model"] = resolved_model_name
                    st.session_state["last_mode"] = exec_mode_val
            except Exception as e:
                st.error(f"❌ **Lỗi khi thực hiện suy luận:** {e}")
                preds = []

        preds: List[Dict[str, Any]] = st.session_state.get("last_predictions", [])
        desc: str = st.session_state.get("last_desc", "")
        used_model: str = st.session_state.get("last_model", "M6_auto")
        used_mode: str = st.session_state.get("last_mode", "native")

        if preds:
            total_n = len(preds)
            avg_conf = float(np.mean([p["confidence_pct"] for p in preds]))
            avg_ms = float(np.mean([p["inference_time_ms"] for p in preds]))
            has_gt = any(p.get("true_acronym") is not None for p in preds)
            match_n = sum(1 for p in preds if p.get("is_match") is True)
            acc_val = (match_n / total_n * 100.0) if (has_gt and total_n > 0) else None

            cov_rate = preds[0]["coverage_audit"]["coverage_rate"] * 100.0
            overall_safety = preds[0]["coverage_audit"]["safety_level"]

            # Hàng chỉ số KPI
            st.markdown("### 📌 Tổng Quan Kết Quả Suy Luận")
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            with col_m1:
                st.metric("Tổng Số Bệnh Nhân", f"{total_n} mẫu")
            with col_m2:
                if acc_val is not None:
                    st.metric("Độ Chính Xác Đối Chiếu", f"{acc_val:.1f}%", f"{match_n}/{total_n} đúng")
                else:
                    st.metric("Nhãn Đối Chiếu", "Không có sẵn")
            with col_m3:
                st.metric("Độ Tin Cậy Trung Bình", f"{avg_conf:.2f}%")
            with col_m4:
                st.metric("Thời Gian Suy Luận TB", f"{avg_ms:.2f} ms", f"{used_mode}")
            with col_m5:
                safety_text = "AN TOÀN" if overall_safety == "GREEN" else ("CẢNH BÁO" if overall_safety == "YELLOW" else "NGUY HIỂM")
                st.metric("Kiểm Định Gen", f"{cov_rate:.1f}%", f"Mức: {safety_text}")

            # Cảnh báo an toàn sinh học nếu độ phủ thấp
            if overall_safety != "GREEN" or not preds[0]["coverage_audit"]["is_clinically_reliable"]:
                warn_msg = preds[0]["coverage_audit"].get("warning_message", "")
                st.warning(f"⚠️ **CẢNH BÁO AN TOÀN SINH HỌC**: {warn_msg}")

            # Bảng tổng hợp kết quả
            st.markdown("#### 📋 Bảng Kết Quả Chi Tiết Từng Bệnh Nhân")
            table_rows = []
            for p in preds:
                t_acr = p.get("true_acronym") or "N/A"
                is_m = p.get("is_match")
                match_icon = "✅ Đúng" if is_m is True else ("❌ Khác" if is_m is False else "—")
                top2_txt = f"{p['top_k'][1]['acronym']} ({p['top_k'][1]['percentage']:.1f}%)" if len(p.get("top_k", [])) > 1 else "—"
                top3_txt = f"{p['top_k'][2]['acronym']} ({p['top_k'][2]['percentage']:.1f}%)" if len(p.get("top_k", [])) > 2 else "—"

                table_rows.append(
                    {
                        "Mã Bệnh Nhân": p["patient_id"],
                        "Chẩn Đoán Chính": p["predicted_acronym"],
                        "Tên Bệnh": p["predicted_disease"],
                        "Độ Tin Cậy": f"{p['confidence_pct']:.2f}%",
                        "Bệnh Thực Tế": t_acr,
                        "Đối Chiếu": match_icon,
                        "Nguy Cơ #2": top2_txt,
                        "Nguy Cơ #3": top3_txt,
                        "Thời Gian (ms)": f"{p['inference_time_ms']:.2f}",
                        "Mức An Toàn": p["coverage_audit"]["safety_level"],
                    }
                )
            df_display = pd.DataFrame(table_rows)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Nút tải file kết quả
            col_d1, col_d2, col_d3 = st.columns([1, 1, 2])
            with col_d1:
                csv_data = prepare_csv_download(preds)
                st.download_button(
                    label="📥 Tải Kết Quả (.CSV)",
                    data=csv_data,
                    file_name=f"tcga_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_d2:
                json_data = json.dumps(preds, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Tải Kết Quả (.JSON)",
                    data=json_data,
                    file_name=f"tcga_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            st.divider()

            # Phân tích sâu từng bệnh nhân (Drill-down)
            st.markdown("### 🔍 Phân Tích Sâu Từng Bệnh Nhân (Patient Drilldown)")
            patient_choices = [p["patient_id"] for p in preds]
            selected_pid = st.selectbox("Chọn bệnh nhân để xem chi tiết phân phối xác suất:", patient_choices)
            selected_p = next(p for p in preds if p["patient_id"] == selected_pid)

            col_chart, col_info = st.columns([3, 2])

            with col_chart:
                st.markdown("##### 📊 Phổ Xác Suất Top 10 Bệnh Ung Thư Có Nguy Cơ Cao Nhất")
                
                # Trích xuất dữ liệu xác suất đầy đủ (ưu tiên all_33_distribution -> probabilities -> top_k)
                chart_data = []
                if "all_33_distribution" in selected_p and selected_p["all_33_distribution"]:
                    for item in selected_p["all_33_distribution"]:
                        code = item["acronym"]
                        info = next((inf for inf in TCGA_33_CANCER_INFO.values() if inf["code"] == code), {})
                        pct_val = float(item.get("percentage", item.get("probability", 0.0) * 100.0))
                        chart_data.append(
                            {
                                "Mã Ung Thư": code,
                                "Tên Bệnh": item.get("disease", info.get("name", code)),
                                "Xác Suất (%)": round(pct_val, 3),
                                "Mức Độ Phổ Biến": info.get("tier", "Moderate"),
                            }
                        )
                elif "probabilities" in selected_p and selected_p["probabilities"]:
                    prob_dict = selected_p["probabilities"]
                    for cid, info in TCGA_33_CANCER_INFO.items():
                        code = info["code"]
                        prob = float(prob_dict.get(code, 0.0))
                        chart_data.append(
                            {
                                "Mã Ung Thư": code,
                                "Tên Bệnh": info.get("name", code),
                                "Xác Suất (%)": round(prob, 3),
                                "Mức Độ Phổ Biến": info.get("tier", "Moderate"),
                            }
                        )
                elif "top_k" in selected_p and selected_p["top_k"]:
                    for item in selected_p["top_k"]:
                        code = item["acronym"]
                        info = next((inf for inf in TCGA_33_CANCER_INFO.values() if inf["code"] == code), {})
                        pct_val = float(item.get("percentage", 0.0))
                        chart_data.append(
                            {
                                "Mã Ung Thư": code,
                                "Tên Bệnh": item.get("disease", info.get("name", code)),
                                "Xác Suất (%)": round(pct_val, 3),
                                "Mức Độ Phổ Biến": info.get("tier", "Moderate"),
                            }
                        )

                df_chart = pd.DataFrame(chart_data)
                if not df_chart.empty:
                    df_top10 = df_chart.sort_values("Xác Suất (%)", ascending=False).head(10).copy()
                    
                    # Định dạng nhãn hiển thị trực tiếp trên từng thanh
                    df_top10["Nhãn Xác Suất"] = df_top10["Xác Suất (%)"].apply(
                        lambda x: f"{x:.2f}%" if x >= 0.01 else ("<0.01%" if x > 0 else "0.00%")
                    )
                    
                    # Giới hạn trục X linh hoạt: tối thiểu 25% nếu max thấp, tối đa 115% nếu max cao
                    max_pct = float(df_top10["Xác Suất (%)"].max())
                    x_limit = 115.0 if max_pct > 60 else max(25.0, max_pct * 1.25)

                    # Biểu đồ thanh ngang với nhãn phần trăm hiển thị trực quan
                    bars = (
                        alt.Chart(df_top10)
                        .mark_bar(cornerRadiusEnd=4)
                        .encode(
                            x=alt.X("Xác Suất (%):Q", scale=alt.Scale(domain=[0, x_limit]), title="Xác Suất (%)"),
                            y=alt.Y(
                                "Mã Ung Thư:N",
                                sort=alt.EncodingSortField(field="Xác Suất (%)", order="descending"),
                                title="Mã Dòng Ung Thư",
                            ),
                            color=alt.condition(
                                alt.datum["Mã Ung Thư"] == selected_p["predicted_acronym"],
                                alt.value("#2563EB"),  # Xanh dương đậm cho bệnh dự đoán chính
                                alt.value("#94A3B8"),  # Xám đá cho các bệnh còn lại
                            ),
                            tooltip=[
                                alt.Tooltip("Mã Ung Thư:N", title="Mã Bệnh"),
                                alt.Tooltip("Tên Bệnh:N", title="Tên Bệnh Khoa Học"),
                                alt.Tooltip("Xác Suất (%):Q", title="Xác Suất Rủi Ro (%)", format=".3f"),
                                alt.Tooltip("Mức Độ Phổ Biến:N", title="Tần Suất Lâm Sàng"),
                            ],
                        )
                    )

                    text = (
                        alt.Chart(df_top10)
                        .mark_text(align="left", baseline="middle", dx=6, fontSize=11, fontWeight=600)
                        .encode(
                            x=alt.X("Xác Suất (%):Q"),
                            y=alt.Y(
                                "Mã Ung Thư:N",
                                sort=alt.EncodingSortField(field="Xác Suất (%)", order="descending"),
                            ),
                            text="Nhãn Xác Suất:N",
                            color=alt.condition(
                                alt.datum["Mã Ung Thư"] == selected_p["predicted_acronym"],
                                alt.value("#1E40AF"),
                                alt.value("#334155"),
                            ),
                        )
                    )

                    st.altair_chart((bars + text).properties(height=340), use_container_width=True)

                    with st.expander("📋 Chi Tiết Bảng Số Liệu Top 10 Bệnh Nguy Cơ Cao", expanded=False):
                        st.dataframe(
                            df_top10[["Mã Ung Thư", "Tên Bệnh", "Xác Suất (%)", "Mức Độ Phổ Biến"]],
                            use_container_width=True,
                            hide_index=True,
                        )
                else:
                    st.warning("Chưa có dữ liệu phân phối xác suất cho bệnh nhân này.")

            with col_info:
                st.markdown("##### Hồ Sơ Bệnh Án & Khảo Sát")
                pred_c = selected_p["predicted_acronym"]
                pred_info = next((inf for inf in TCGA_33_CANCER_INFO.values() if inf["code"] == pred_c), {})
                rarity = pred_info.get("tier", "Moderate")

                st.markdown(
                    f"""
                    - **Mã định danh:** `{selected_p['patient_id']}`
                    - **Chẩn đoán dự báo:** **{pred_c}** ({selected_p['predicted_disease']})
                    - **Độ tin cậy xác suất:** `{selected_p['confidence_pct']:.2f}%`
                    - **Mức độ phổ biến:** {rarity}
                    - **Độ trễ suy luận:** `{selected_p['inference_time_ms']:.2f} ms`
                    """
                )

                if selected_p.get("true_acronym"):
                    t_code = selected_p["true_acronym"]
                    t_dis = selected_p["true_disease"]
                    if selected_p.get("is_match"):
                        st.success(f"✅ **Đối chiếu nhãn chuẩn:** Chính xác khớp với `{t_code}` ({t_dis})")
                    else:
                        st.error(f"❌ **Đối chiếu nhãn chuẩn:** Bệnh thực tế là `{t_code}` ({t_dis})")

                # Bảng Top-5 chẩn đoán phân biệt
                st.markdown("**Top-5 Chẩn Đoán Phân Biệt (Differential Diagnoses):**")
                topk_df = pd.DataFrame(
                    [
                        {
                            "Thứ hạng": f"#{item['rank']}",
                            "Mã": item["acronym"],
                            "Xác suất": f"{item['percentage']:.2f}%",
                            "Tên bệnh lý": item["disease"],
                        }
                        for item in selected_p.get("top_k", [])
                    ]
                )
                st.dataframe(topk_df, use_container_width=True, hide_index=True)

    # TAB 2: BẢNG ĐIỂM BENCHMARK 33 UNG THƯ
    with tab2:
        st.markdown("### 📊 Bộ Bảng Điểm Thực Nghiệm Chuẩn TCGA Pan-Cancer (Canonical CSV Benchmarks)")
        st.markdown(
            "Toàn bộ 17 báo cáo định lượng chi tiết được sinh tự động từ các thí nghiệm huấn luyện, "
            "hiệu chuẩn xác suất, đánh giá trên 3 phân chia Train/Val/Test và khảo sát độ bền khi khuyết gen."
        )

        tables_dir = _CURRENT_DIR / "reports_and_charts" / "tables_csv"
        csv_files = sorted(tables_dir.glob("*.csv")) if tables_dir.exists() else []

        if csv_files:
            file_options = {f.name: f for f in csv_files}
            # Ưu tiên mặc định bảng per-class 33 ung thư
            default_key = next((k for k in file_options if "09_per_class" in k), list(file_options.keys())[0])
            selected_csv_name = st.selectbox(
                "Chọn bảng benchmark muốn xem:",
                list(file_options.keys()),
                index=list(file_options.keys()).index(default_key),
            )
            selected_csv_path = file_options[selected_csv_name]

            df_bench = pd.read_csv(selected_csv_path)
            st.markdown(f"**Bảng dữ liệu:** `{selected_csv_name}` (Kích thước: {df_bench.shape[0]} hàng × {df_bench.shape[1]} cột)")
            st.dataframe(df_bench, use_container_width=True)

            csv_text = selected_csv_path.read_text(encoding="utf-8")
            st.download_button(
                label=f"📥 Tải Bảng {selected_csv_name}",
                data=csv_text,
                file_name=selected_csv_name,
                mime="text/csv",
            )
        else:
            st.warning("Không tìm thấy thư mục `reports_and_charts/tables_csv/`.")

    # TAB 3: BIỂU ĐỒ HUẤN LUYỆN & ĐÁNH GIÁ 300 DPI
    with tab3:
        st.markdown("### 📈 Thư Viện Biểu Đồ Thực Nghiệm Độc Lập Chuẩn Xuất Bản (300 DPI Figures)")
        st.markdown(
            "13 biểu đồ khoa học trực quan hóa quá trình hội tụ, kiểm soát overfitting, "
            "hiệu chuẩn ECE, phân tích ma trận nhầm lẫn 33×33 và kiểm nghiệm độ bền khi thiếu gen."
        )

        figures_dir = _CURRENT_DIR / "reports_and_charts" / "figures_300dpi"
        png_files = sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []

        if png_files:
            fig_dict = {f.name: f for f in png_files}
            fig_captions = {
                "fig01_training_convergence_curves.png": "Hình 1: Đường cong hội tụ Loss và Accuracy của 6 họ mô hình M1-M6 qua các epoch.",
                "fig02_regularization_and_calibration_sweeps.png": "Hình 2: Khảo sát siêu tham số điều hòa L2 và hiệu chỉnh xác suất tiên nghiệm tau.",
                "fig03_tri_split_performance_and_95ci.png": "Hình 3: Hiệu năng tổng thể trên 3 phân chia Train / Val / Test kèm khoảng tin cậy 95%.",
                "fig04_overfitting_gap_and_generalization.png": "Hình 4: Khoảng cách tổng quát hóa (Train - Test gap) chứng minh không bị overfitting.",
                "fig05_feature_spaces_ablation_and_speedup.png": "Hình 5: Đánh giá các không gian đặc trưng F0, F1, F2 và tốc độ gia tốc tính toán.",
                "fig06_probability_calibration_and_reliability.png": "Hình 6: Biểu đồ độ tin cậy hiệu chuẩn xác suất và giảm thiểu sai số ECE.",
                "fig07_per_class_33_cancers_heatmap_and_pr.png": "Hình 7: Heatmap chi tiết F1-score và Precision-Recall trên toàn bộ 33 loại ung thư.",
                "fig08_confusion_matrix_33x33_and_error_anatomy.png": "Hình 8: Ma trận nhầm lẫn 33×33 và giải phẫu các cặp ung thư dễ nhầm lẫn nhất.",
                "fig09_biological_slices_and_metastatic_audit.png": "Hình 9: Đánh giá trên các lát cắt sinh học, nhóm ung thư hiếm và mẫu di căn.",
                "fig10_systemds_compiler_ablation_c1_c6.png": "Hình 10: Tối ưu hóa trình biên dịch Apache SystemDS qua các cấp độ C1 đến C6.",
                "fig11_systemds_heavy_hitters_and_256mb_guard.png": "Hình 11: Phân bổ thời gian các toán tử nặng nhất và cơ chế bảo vệ bộ nhớ 256MB.",
                "fig12_stage1_to_stage3_scaling_and_ops_tradeoff.png": "Hình 12: Đánh giá mở rộng quy mô từ tập mini D0 đến tập dữ liệu chuẩn D1.",
                "fig13_gene_dropout_robustness_curves.png": "Hình 13: Độ bền khi khuyết thiếu gen từ 100% xuống 10% với các chiến lược bù trừ.",
            }

            selected_fig_name = st.selectbox(
                "Chọn biểu đồ nghiên cứu muốn hiển thị:",
                list(fig_dict.keys()),
                format_func=lambda x: f"{x.split('.')[0]} — {fig_captions.get(x, x)}",
            )
            selected_fig_path = fig_dict[selected_fig_name]

            st.image(str(selected_fig_path), caption=fig_captions.get(selected_fig_name, selected_fig_name), use_container_width=True)

            # Tải ảnh
            fig_bytes = selected_fig_path.read_bytes()
            st.download_button(
                label=f"📥 Tải Biểu Đồ 300 DPI ({selected_fig_name})",
                data=fig_bytes,
                file_name=selected_fig_name,
                mime="image/png",
            )
        else:
            st.warning("Không tìm thấy thư mục `reports_and_charts/figures_300dpi/`.")

    # TAB 4: THÔNG SỐ & TỐI ƯU HÓA APACHE SYSTEMDS
    with tab4:
        st.markdown("### ⚙️ Kiến Trúc Biên Dịch & Tối Ưu Hóa Apache SystemDS")
        st.markdown(
            "Hệ thống tận dụng tối đa năng lực biên dịch đồ thị tính toán và thực thi ma trận quy mô lớn của Apache SystemDS:"
        )

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            st.markdown(
                """
                #### 🚀 Các Cấp Độ Tối Ưu Hóa Trình Biên Dịch (C1 - C6)
                1. **C1 (Constant Folding & Dead Code Elimination):** Tinh giản biểu thức hằng và loại bỏ nhánh không sử dụng.
                2. **C2 (Common Subexpression Elimination):** Tái sử dụng kết quả trung gian giữa các phép nhân ma trận lặp lại.
                3. **C3 (Algebraic Rewrites):** Tự động chuyển đổi thứ tự tính toán đại số tuyến tính để giảm độ phức tạp từ $O(N^3)$ xuống $O(N^2)$.
                4. **C4 (Operator Fusion):** Gộp chuỗi phép tính ma trận thành hạt nhân tính toán đơn nhất (Fused Kernels).
                5. **C5 (Memory Budget Management):** Quản lý nghiêm ngặt ngân sách bộ nhớ, ngăn chặn tràn RAM và Garbage Collection.
                6. **C6 (OpenBLAS Hardware Vectorization):** Kích hoạt SIMD AVX2/AVX-512 đa luồng qua thư viện OpenBLAS nguyên bản.
                """
            )
        with col_opt2:
            st.markdown(
                """
                #### 🛡️ Cơ Chế Bảo Vệ Bộ Nhớ 256 MB (256MB Guard)
                - **Mô hình triển khai siêu nhẹ:** Toàn bộ trọng số được lượng tử hóa và lưu trữ định dạng nhị phân tối ưu.
                - **Độ trễ suy luận tức thời:** Đạt < 1 ms trên mỗi bệnh nhân ở chế độ `native` và < 30 ms ở chế độ `systemds`.
                - **Không rò rỉ dữ liệu (Zero Leakage):** Trọng số Centering, Scalefactor và ANOVA được cô lập tuyệt đối khỏi tập Test.
                """
            )

        st.markdown("#### 📜 Kịch Bản DML Chạy Dự Đoán Chuẩn (`release_staging/dml/predict_tcga.dml`)")
        dml_path = _CURRENT_DIR / "dml" / "predict_tcga.dml"
        if dml_path.exists():
            dml_code = dml_path.read_text(encoding="utf-8")
            st.code(dml_code[:1800] + "\n\n# ... [Trích đoạn DML hoàn chỉnh trong release_staging/dml/predict_tcga.dml]", language="dml")
        else:
            st.info("Kịch bản DML nằm tại thư mục `release_staging/dml/`.")


if __name__ == "__main__":
    main()
