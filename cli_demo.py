#!/usr/bin/env python3
"""
TCGA Pan-Cancer Standalone Genomic Inference CLI Demonstration Tool
====================================================================
Công cụ demo dòng lệnh (CLI) hoàn chỉnh cho hệ thống phân loại 33 bệnh ung thư TCGA:
- Nhập tệp biểu hiện gen đơn hoặc đa bệnh nhân (.csv, .tsv, .csv.gz, .tsv.gz, .json, .npz).
- Chế độ Random N bệnh nhân từ tập Test thực tế (100 bệnh nhân có nhãn chuẩn) hoặc sinh ngẫu nhiên
  theo phân phối sinh học (synthetic biological generator).
- Cấu hình linh hoạt tỷ lệ % gen hiện diện / random dropout / bỏ các gen chỉ định.
- Tự động hiển thị kết quả trực quan trên Terminal kèm đối chiếu bệnh thực tế (nếu có).
- Tự động lưu toàn bộ kết quả chi tiết vào cả tệp .csv và .json.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Đảm bảo đường dẫn import tương đối và độc lập, không hardcode đường dẫn tuyệt đối
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

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


def _parse_rate_value(val: Optional[Union[str, float]], default: float = 1.0) -> float:
    """Chuyển đổi chuỗi tỷ lệ phần trăm (ví dụ: '100%', '75%', '0.75', '75', '1%') sang float 0.0 - 1.0."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        fval = float(val)
        if fval < 0.0 or fval > 100.0:
            raise ValueError(f"Giá trị tỷ lệ ngoài khoảng [0, 100]: {val}")
        return fval / 100.0 if fval > 1.0 else fval
    s = str(val).strip()
    is_pct = "%" in s
    s_clean = s.rstrip("%").strip()
    try:
        fval = float(s_clean)
    except ValueError as e:
        raise ValueError(
            f"Giá trị tỷ lệ không hợp lệ: '{val}'. Kỳ vọng dạng '75%', '0.75' hoặc '75'."
        ) from e

    if fval < 0.0 or fval > 100.0:
        raise ValueError(f"Giá trị tỷ lệ ngoài khoảng [0, 100]: '{val}'")

    if is_pct:
        return fval / 100.0
    return fval / 100.0 if fval > 1.0 else fval


def _format_box_line(content: str, width: int = 74) -> str:
    """Đảm bảo lề phải của khung hiển thị luôn căn thẳng hàng chính xác tuyệt đối."""
    if len(content) > width:
        content = content[:width - 3] + "..."
    padding = width - len(content)
    return f"  │ {content}{' ' * padding} │"


def _render_progress_bar(percentage: float, width: int = 24) -> str:
    """Tạo thanh tiến trình trực quan bằng ký tự Unicode."""
    clamped = max(0.0, min(100.0, float(percentage)))
    filled = int(round((clamped / 100.0) * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {clamped:6.2f}%"


def _format_safety_badge(safety_level: str) -> str:
    """Định dạng huy hiệu an toàn sinh học rõ ràng."""
    level = str(safety_level).upper()
    if level == "GREEN":
        return "[XANH LÁ - AN TOÀN SINH HỌC TUYỆT ĐỐI]"
    elif level == "YELLOW":
        return "[VÀNG - ĐỘ BỀN CAO / SUY GIẢM NHẸ]"
    elif level == "ORANGE":
        return "[CAM - CẢNH BÁO BÙ TRỪ KHUYẾT THIẾU]"
    else:
        return "[ĐỎ - NGUY HIỂM / KHÔNG TIN CẬY LÂM SÀNG]"


def _save_prediction_outputs(
    predictions: List[Dict[str, Any]],
    output_dir: Path,
    custom_json_path: Optional[Path] = None,
    custom_csv_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Tự động xuất toàn bộ kết quả chi tiết ra định dạng .json và .csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = custom_json_path if custom_json_path is not None else output_dir / f"predictions_{ts}.json"
    csv_path = custom_csv_path if custom_csv_path is not None else output_dir / f"predictions_{ts}.csv"

    # 1. Ghi tệp JSON đầy đủ cấu trúc
    json_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Xây dựng DataFrame phẳng hóa cho CSV
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
        # Thêm thông tin Top-K
        top_k_list = p.get("top_k", [])
        for k_idx, item in enumerate(top_k_list[:5], start=1):
            row[f"top{k_idx}_acronym"] = item.get("acronym", "")
            row[f"top{k_idx}_pct"] = item.get("percentage", 0.0)

        # Thêm toàn bộ xác suất của 33 loại ung thư
        all_probs = p.get("probabilities", {})
        for cid in sorted(TCGA_33_CANCER_INFO.keys()):
            code = TCGA_33_CANCER_INFO[cid]["code"]
            row[f"prob_{code}_pct"] = all_probs.get(code, 0.0)

        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(csv_path, index=False, encoding="utf-8")

    # Ghi nhận / sao chép ra bản latest để dễ truy cập
    if custom_json_path is None:
        latest_json = output_dir / "predictions_latest.json"
        try:
            latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
    if custom_csv_path is None:
        latest_csv = output_dir / "predictions_latest.csv"
        try:
            df_out.to_csv(latest_csv, index=False, encoding="utf-8")
        except Exception:
            pass

    return json_path, csv_path


def run_cli_demo(args: argparse.Namespace) -> None:
    """Hàm thực thi chính cho Demo CLI."""
    engine = PretrainedTCGAInferenceEngine(
        models_dir=args.models_dir,
        heavy_bundle_dir=args.heavy_bundle,
    )
    sampler = PatientSampler(demo_dir=args.demo_dir)

    # Giải quyết tỷ lệ coverage / dropout
    coverage_rate = _parse_rate_value(args.coverage, default=1.0)
    dropout_rate = _parse_rate_value(args.dropout, default=0.0)

    # Xác định nguồn dữ liệu đầu vào
    input_source_desc = ""
    X_input: Any = None
    patient_ids: Optional[List[str]] = None
    ground_truth_map: Dict[str, Dict[str, Any]] = {}

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[LỖI] Không tìm thấy tệp dữ liệu: {input_path}", file=sys.stderr)
            sys.exit(1)
        loaded = sampler.load_patient_file(input_path)
        input_source_desc = loaded["description"]
        X_input = loaded["X_raw"]
        patient_ids = loaded["patient_ids"]
        ground_truth_map = loaded["ground_truth_map"]
    elif args.random is not None:
        n_patients = max(1, int(args.random))
        if args.random_type == "real":
            input_source_desc = f"Lấy ngẫu nhiên {n_patients} bệnh nhân từ kho Test thực tế (100 bệnh nhân chuẩn)"
            sampled = sampler.sample_real_patients(
                n=n_patients,
                seed=args.seed,
                cancer_type=args.cancer,
            )
            X_input = sampled["X_raw"]
            patient_ids = sampled["patient_ids"]
            for info in sampled["patients_metadata"]:
                ground_truth_map[info["patient_id"]] = {
                    "true_label_id": info["true_label_id"],
                    "true_acronym": info["true_acronym"],
                    "true_disease": info["true_disease"],
                    "sample_type": info.get("sample_type", "Primary Tumor"),
                }
        else:
            input_source_desc = f"Sinh giả lập {n_patients} bệnh nhân theo phân phối sinh học Pan-Cancer"
            sampled = sampler.generate_synthetic_patients(
                n=n_patients,
                seed=args.seed,
                cancer_type=args.cancer,
                biological_variation=args.variation,
            )
            X_input = sampled["X_raw"]
            patient_ids = sampled["patient_ids"]
            for info in sampled["patients_metadata"]:
                ground_truth_map[info["patient_id"]] = {
                    "true_label_id": info["simulated_target_label_id"],
                    "true_acronym": info["simulated_target_acronym"],
                    "true_disease": info["simulated_target_disease"],
                    "sample_type": "Synthetic Simulated Tumor",
                }
    else:
        # Chế độ ngầm định: lấy ngẫu nhiên 1 bệnh nhân thực tế để người dùng thấy ngay kết quả
        print("[THÔNG BÁO] Không truyền tệp hay số lượng random, tự động lấy ngẫu nhiên 1 bệnh nhân từ tập Test...")
        input_source_desc = "Mặc định: 1 bệnh nhân ngẫu nhiên từ kho Test thực tế"
        sampled = sampler.sample_real_patients(n=1, seed=args.seed or 42)
        X_input = sampled["X_raw"]
        patient_ids = sampled["patient_ids"]
        for info in sampled["patients_metadata"]:
            ground_truth_map[info["patient_id"]] = {
                "true_label_id": info["true_label_id"],
                "true_acronym": info["true_acronym"],
                "true_disease": info["true_disease"],
                "sample_type": info.get("sample_type", "Primary Tumor"),
            }

    # Thực hiện suy luận qua PretrainedTCGAInferenceEngine
    resolved_model = engine.resolve_model_name(args.model)
    preds = engine.predict(
        model_name=resolved_model,
        X_raw=X_input,
        top_k=args.top_k,
        patient_ids=patient_ids,
        tau_override=args.tau,
        execution_mode=args.mode,
        allow_missing_genes=args.allow_missing,
        gene_coverage_rate=coverage_rate,
        gene_dropout_rate=dropout_rate,
        dropped_genes=args.drop_genes,
        missing_policy=args.policy,
        seed=args.seed,
    )

    # Bổ sung thông tin đối chiếu thực tế nếu có
    has_ground_truth = False
    matches_count = 0
    for p in preds:
        pid = p["patient_id"]
        if pid in ground_truth_map:
            has_ground_truth = True
            gt = ground_truth_map[pid]
            p["true_label_id"] = gt["true_label_id"]
            p["true_acronym"] = gt["true_acronym"]
            p["true_disease"] = gt["true_disease"]
            p["sample_type"] = gt.get("sample_type", "Tumor")
            is_match = (p["predicted_acronym"] == gt["true_acronym"])
            p["is_match"] = bool(is_match)
            if is_match:
                matches_count += 1
        else:
            p["true_label_id"] = None
            p["true_acronym"] = None
            p["true_disease"] = None
            p["sample_type"] = "Clinical Sample"
            p["is_match"] = None

    # IN KẾT QUẢ TRỰC QUAN RA TERMINAL
    term_width = 82
    print("\n" + "═" * term_width)
    print("  TCGA PAN-CANCER GENOMIC INFERENCE ENGINE - TERMINAL DEMO CLI")
    print("  Hệ thống suy luận phân loại 33 bệnh ung thư TCGA từ biểu hiện gen")
    print("═" * term_width)
    print(f"  • Nguồn dữ liệu       : {input_source_desc}")
    print(f"  • Mô hình sử dụng     : {resolved_model} (Engine: {args.mode})")
    print(f"  • Tỷ lệ gen sử dụng   : {coverage_rate*100:.1f}% ({int(coverage_rate*engine.num_genes):,}/{engine.num_genes:,} gen)")
    if dropout_rate > 0.0:
        print(f"  • Tỷ lệ gen dropout   : {dropout_rate*100:.1f}% ngẫu nhiên")
    if args.drop_genes:
        print(f"  • Gen loại trừ        : {args.drop_genes}")
    print(f"  • Chiến lược bù trừ   : {args.policy}")
    print("─" * term_width)

    max_display = args.max_display if args.max_display is not None else len(preds)
    for idx, p in enumerate(preds[:max_display]):
        pid = p["patient_id"]
        pred_code = p["predicted_acronym"]
        pred_name = p["predicted_disease"]
        conf_pct = p["confidence_pct"]
        latency = p["inference_time_ms"]
        stype = p.get("sample_type", "Sample")

        print(f"\n[{idx+1}/{len(preds)}] BỆNH NHÂN: {pid} ({stype})")
        print("  ┌────────────────────────────────────────────────────────────────────────────┐")
        print(_format_box_line(f"Chẩn đoán chính : {pred_code:5s} - {pred_name}"))
        print(_format_box_line(f"Độ tin cậy      : {conf_pct:6.2f}% (Thời gian suy luận: {latency:6.2f} ms)"))

        if p.get("true_acronym"):
            true_code = p["true_acronym"]
            true_name = p["true_disease"]
            is_match = p.get("is_match")
            match_badge = "[CHÍNH XÁC] ✓" if is_match else f"[KHÁC BIỆT (Thực tế: {true_code})] ✗"
            print(_format_box_line(f"Bệnh thực tế    : {true_code:5s} - {true_name} {match_badge}"))

        audit = p["coverage_audit"]
        s_level = audit["safety_level"]
        s_badge = _format_safety_badge(s_level)
        cov_pct = audit["coverage_rate"] * 100.0
        print(_format_box_line(f"Kiểm định gen   : {s_badge} Phủ: {cov_pct:5.1f}%"))
        if audit.get("warning_message"):
            warn_msg = audit["warning_message"]
            print(_format_box_line(f"CẢNH BÁO AN TOÀN: {warn_msg}"))
        print("  └────────────────────────────────────────────────────────────────────────────┘")

        print(f"  Top-{len(p['top_k'])} Nguy cơ Ung thư Cao nhất:")
        for tk in p["top_k"]:
            bar_str = _render_progress_bar(tk["percentage"], width=20)
            print(f"    #{tk['rank']} [{tk['acronym']:5s}] {bar_str}  {tk['disease']}")

    if len(preds) > max_display:
        print(f"\n  ... và {len(preds) - max_display} bệnh nhân khác (giới hạn hiển thị {max_display}).")

    # TỔNG KẾT
    print("\n" + "═" * term_width)
    print("  TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ")
    print("─" * term_width)
    total_evaluated = len(preds)
    avg_latency = float(np.mean([p["inference_time_ms"] for p in preds]))
    print(f"  • Tổng số bệnh nhân đánh giá : {total_evaluated}")
    print(f"  • Thời gian suy luận trung bình: {avg_latency:.2f} ms / bệnh nhân")

    if has_ground_truth:
        acc = (matches_count / total_evaluated) * 100.0
        print(f"  • Độ chính xác đối chiếu     : {matches_count}/{total_evaluated} ({acc:.1f}%)")

    green_count = sum(1 for p in preds if p["coverage_audit"]["safety_level"] == "GREEN")
    non_green = total_evaluated - green_count
    print(f"  • Phân loại an toàn sinh học : {green_count} An toàn (Green), {non_green} Cảnh báo/Giảm phủ")

    # TỰ ĐỘNG LƯU TOÀN BỘ KẾT QUẢ RA FILE .CSV VÀ .JSON
    _DEFAULT_OUTPUT_DIR = (
        _PROJECT_ROOT / "demo_output"
        if (_PROJECT_ROOT / "release_staging").exists()
        else _CURRENT_DIR / "demo_output"
    )
    out_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT_DIR
    custom_json = Path(args.output_json) if args.output_json else None
    custom_csv = Path(args.output_csv) if args.output_csv else None

    json_saved, csv_saved = _save_prediction_outputs(
        predictions=preds,
        output_dir=out_dir,
        custom_json_path=custom_json,
        custom_csv_path=custom_csv,
    )

    print("─" * term_width)
    print("  TỰ ĐỘNG LƯU TRỮ KẾT QUẢ ĐẦY ĐỦ:")
    print(f"  [JSON] -> {json_saved}")
    print(f"  [CSV ] -> {csv_saved}")
    print("═" * term_width + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Xây dựng bộ phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        prog="cli_demo.py",
        description="TCGA Pan-Cancer Genomic Inference Engine - Full Interactive Terminal Demo Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng thực tế:
  # 1. Trình diễn mặc định (lấy ngay 1 bệnh nhân thực tế):
  python cli_demo.py

  # 2. Random 5 bệnh nhân thực tế từ kho Test (đối chiếu nhãn chuẩn):
  python cli_demo.py --random 5 --model M6_auto --top-k 5

  # 3. Random 10 bệnh nhân ung thư phổi (LUAD) giả lập sinh học:
  python cli_demo.py --random 10 --random-type synthetic --cancer LUAD

  # 4. Truyền tệp dữ liệu gen người dùng (hỗ trợ .csv, .tsv, .gz, .json, .npz):
  python cli_demo.py --input demo_data/demo_patient_single.json

  # 5. Kiểm nghiệm giả lập khuyết gen (chỉ giữ 75% gen, bỏ gen TP53 và KRAS):
  python cli_demo.py --random 5 --coverage 75% --drop-genes TP53,KRAS --policy linear_scale
        """,
    )

    group_input = parser.add_argument_group("Tùy chọn Đầu vào Dữ liệu (Input Data)")
    group_input.add_argument(
        "--input",
        "-i",
        help="Đường dẫn đến tệp dữ liệu gen của 1 hoặc nhiều người (.csv, .tsv, .csv.gz, .json, .npz)",
    )
    group_input.add_argument(
        "--random",
        "-r",
        "--count",
        "-n",
        type=int,
        dest="random",
        help="Số lượng bệnh nhân muốn test ngẫu nhiên (ví dụ: 1, 5, 20...)",
    )
    group_input.add_argument(
        "--random-type",
        choices=["real", "synthetic"],
        default="real",
        help="Nguồn mẫu ngẫu nhiên: 'real' (tập Test thực tế có nhãn đối chiếu) hoặc 'synthetic' (sinh giả lập)",
    )
    group_input.add_argument(
        "--cancer",
        help="Lọc loại ung thư cụ thể khi random (ví dụ: LUAD, BRCA, GBM, UVM...)",
    )
    group_input.add_argument(
        "--variation",
        type=float,
        default=1.0,
        help="Độ biến thiên sinh học khi sinh giả lập (mặc định: 1.0)",
    )

    group_gene = parser.add_argument_group("Cấu hình Gen & Bù trừ Khuyết Gen (Missing Genes & Robustness)")
    group_gene.add_argument(
        "--coverage",
        "-c",
        default="100%",
        help="Tỷ lệ phần trăm gen đưa vào sử dụng (ví dụ: 100%%, 75%%, 50%% hoặc 0.75). Mặc định: 100%%",
    )
    group_gene.add_argument(
        "--dropout",
        "-d",
        default="0%",
        help="Tỷ lệ phần trăm gen bị ngẫu nhiên bỏ bớt / đột biến mất (ví dụ: 25%% hoặc 0.25). Mặc định: 0%%",
    )
    group_gene.add_argument(
        "--drop-genes",
        help="Danh sách gen chỉ định muốn loại trừ (dạng phân tách bởi dấu phẩy 'TP53,KRAS,EGFR' hoặc đường dẫn tệp)",
    )
    group_gene.add_argument(
        "--policy",
        choices=["linear_scale", "zero_fill", "sqrt_scale", "l2_renorm"],
        default="linear_scale",
        help="Chiến lược toán học bù trừ khuyết gen (mặc định: linear_scale [khuyên dùng])",
    )
    group_gene.add_argument(
        "--allow-missing",
        action="store_true",
        default=True,
        help="Cho phép suy luận ngay cả khi ma trận thiếu gen (tự động bật)",
    )

    group_model = parser.add_argument_group("Cấu hình Mô hình & Suy luận (Model & Inference)")
    group_model.add_argument(
        "--model",
        "-m",
        default="M6_auto",
        help="Mô hình dự đoán: M6_auto (khuyên dùng), M1_multiLogReg, M2_msvm_reg100, M3_mlp_lowrank, M6_lightweight_no_m5",
    )
    group_model.add_argument(
        "--mode",
        choices=["native", "systemds"],
        default="native",
        help="Động cơ thực thi: 'native' (NumPy vector hóa <1ms) hoặc 'systemds' (Apache SystemDS DML)",
    )
    group_model.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Số lượng bệnh ung thư nguy cơ cao nhất (Top-K) hiển thị (mặc định: 5)",
    )
    group_model.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Hệ số điều chỉnh xác suất tiên nghiệm tau (prior adjustment override)",
    )
    group_model.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed ngẫu nhiên để tái lập kết quả (mặc định: 42)",
    )

    group_out = parser.add_argument_group("Tùy chọn Xuất Kết quả (Output & Storage)")
    group_out.add_argument(
        "--output-dir",
        help="Thư mục lưu kết quả chi tiết .csv và .json (mặc định: demo_output)",
    )
    group_out.add_argument(
        "--output-json",
        "-oj",
        help="Đường dẫn tệp cụ thể để xuất kết quả JSON",
    )
    group_out.add_argument(
        "--output-csv",
        "-oc",
        help="Đường dẫn tệp cụ thể để xuất kết quả CSV",
    )
    group_out.add_argument(
        "--max-display",
        type=int,
        default=15,
        help="Số lượng bệnh nhân tối đa in chi tiết ra terminal (mặc định: 15)",
    )
    group_out.add_argument(
        "--models-dir",
        help="Đường dẫn thư mục chứa trọng số models_pretrained (tự động phát hiện)",
    )
    group_out.add_argument(
        "--heavy-bundle",
        help="Đường dẫn thư mục google_drive_heavy_bundle (tự động phát hiện)",
    )
    group_out.add_argument(
        "--demo-dir",
        help="Đường dẫn thư mục demo_data (tự động phát hiện)",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_cli_demo(args)


if __name__ == "__main__":
    main()
