#!/usr/bin/env python3
"""
Giao Diện Dòng Lệnh Trực Tiếp (TCGA Pan-Cancer CLI).

Hỗ trợ:
- `predict`: Dự đoán cho 1 bệnh nhân hoặc tập bệnh nhân từ tệp JSON, CSV, TSV, NPZ.
- `sample`: Lấy mẫu bệnh nhân thực tế hoặc sinh bệnh nhân giả lập sinh học.
- `benchmark-missing`: Chạy kiểm nghiệm bù trừ khuyết gen với các tỷ lệ phủ và chiến lược khác nhau.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
from typing import List, Optional

# Bootstrap sys.path for direct script execution
_cur = Path(__file__).resolve()
_project_root = _cur.parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_release_root = _cur.parents[1]
if str(_release_root) not in sys.path:
    sys.path.insert(0, str(_release_root))

import numpy as np
import pandas as pd

try:
    from .cancer_types import TCGA_33_CANCER_INFO
    from .engine import PretrainedTCGAInferenceEngine
    from .sampler import PatientSampler
except ImportError:
    from tcga_inference.cancer_types import TCGA_33_CANCER_INFO
    from tcga_inference.engine import PretrainedTCGAInferenceEngine
    from tcga_inference.sampler import PatientSampler


def cmd_predict(args: argparse.Namespace) -> None:
    engine = PretrainedTCGAInferenceEngine(
        models_dir=args.models_dir,
        heavy_bundle_dir=args.heavy_bundle,
    )
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Không tìm thấy tệp: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[TCGA Inference] Đang nạp dữ liệu từ {input_path}...")
    preds = engine.predict(
        model_name=args.model,
        X_raw=input_path,
        top_k=args.top_k,
        tau_override=args.tau,
        execution_mode=args.mode,
        allow_missing_genes=args.allow_missing,
        gene_coverage_rate=args.coverage,
        gene_dropout_rate=args.dropout,
        dropped_genes=args.drop_genes,
        missing_policy=args.policy,
        seed=args.seed,
    )

    print(f"\n{'='*75}")
    print(f"KẾT QUẢ DỰ BÁO UNG THƯ TCGA (Model: {args.model} | Mode: {args.mode})")
    print(f"{'='*75}")

    for p in preds[: args.max_display]:
        idx = p["sample_index"]
        pid = p["patient_id"]
        pred_code = p["predicted_acronym"]
        pred_dis = p["predicted_disease"]
        conf_pct = p["confidence_pct"]
        ms = p["inference_time_ms"]

        print(f"\n[Bệnh nhân #{idx+1}]: {pid}")
        print(f"  -> Chẩn đoán chính : {pred_code} ({pred_dis}) - Độ tin cậy: {conf_pct:.2f}% (Thời gian: {ms:.2f} ms)")

        audit = p["coverage_audit"]
        if not audit["is_clinically_reliable"] or audit["warning_message"]:
            print(f"  -> CẢNH BÁO AN TOÀN: {audit['warning_message']}")

        print(f"  -> Top-{len(p['top_k'])} bệnh nguy cơ cao nhất:")
        for tk in p["top_k"]:
            bar = "#" * int(round(tk["percentage"] / 5))
            print(f"     #{tk['rank']} {tk['acronym']:5s} ({tk['percentage']:6.2f}%): {bar:<20s} {tk['disease']}")

    if len(preds) > args.max_display:
        print(f"\n... và {len(preds) - args.max_display} bệnh nhân khác (hiển thị tối đa {args.max_display}).")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Lưu kết quả ra JSON
        out_p.write_text(json.dumps(preds, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[XUẤT BÁO CÁO] Đã lưu kết quả chi tiết vào {out_p}")


def cmd_sample(args: argparse.Namespace) -> None:
    sampler = PatientSampler()
    if args.type == "real":
        print(f"[PatientSampler] Đang lấy ngẫu nhiên {args.n} bệnh nhân từ kho Test thực tế...")
        data = sampler.sample_real_patients(
            n=args.n,
            seed=args.seed,
            cancer_type=args.cancer,
        )
        print(f"\nĐã lấy thành công {data['num_patients']} bệnh nhân thực tế:")
        for i, info in enumerate(data["patients_metadata"]):
            print(f"  #{i+1}: {info['patient_id']} | Nhãn thật: {info['true_acronym']:5s} ({info['true_disease']}) [{info['sample_type']}]")
    else:
        print(f"[PatientSampler] Đang sinh ngẫu nhiên {args.n} bệnh nhân giả lập theo phân phối sinh học...")
        data = sampler.generate_synthetic_patients(
            n=args.n,
            seed=args.seed,
            cancer_type=args.cancer,
            biological_variation=args.variation,
        )
        print(f"\nĐã sinh thành công {data['num_patients']} bệnh nhân giả lập:")
        for i, info in enumerate(data["patients_metadata"]):
            print(f"  #{i+1}: {info['patient_id']} | Lớp mô phỏng: {info['simulated_target_acronym']:5s} ({info['simulated_target_disease']})")

    if args.predict:
        engine = PretrainedTCGAInferenceEngine()
        preds = engine.predict(
            model_name=args.model,
            X_raw=data["X_raw"],
            top_k=args.top_k,
            patient_ids=data["patient_ids"],
        )
        print(f"\n{'='*75}")
        print(f"ĐỐI CHIẾU DỰ ĐOÁN TỨC THỜI (Model: {args.model})")
        print(f"{'='*75}")
        correct = 0
        for i, (p, info) in enumerate(zip(preds, data["patients_metadata"])):
            pred_code = p["predicted_acronym"]
            true_code = info.get("true_acronym") or info.get("simulated_target_acronym")
            is_match = (pred_code == true_code)
            if is_match:
                correct += 1
            status = "DỰ ĐOÁN ĐÚNG [OK]" if is_match else f"KHÁC BIỆT (Thật: {true_code})"
            print(f"  #{i+1} {p['patient_id']}: Dự đoán {pred_code:5s} ({p['confidence_pct']:5.1f}%) | {status}")
        print(f"\nĐộ chính xác trên lô mẫu: {correct}/{len(preds)} ({correct/len(preds)*100:.1f}%)")


def cmd_benchmark_missing(args: argparse.Namespace) -> None:
    engine = PretrainedTCGAInferenceEngine()
    demo_p = Path(__file__).resolve().parents[1] / "demo_data" / "demo_patients_100.npz"
    demo_data = np.load(demo_p)
    X = demo_data["X_raw"]
    y = demo_data["y_true"]

    coverages = [1.0, 0.90, 0.75, 0.50, 0.25, 0.10]
    policies = ["zero_fill", "linear_scale", "sqrt_scale"]
    models = [args.model] if args.model != "all" else ["M1_multiLogReg", "M2_msvm_reg100", "M3_mlp_lowrank", "M6_lightweight_no_m5"]

    print(f"\n{'='*80}")
    print(f"KHẢO SÁT THỰC NGHIỆM ĐỘ BỀN VỮNG KHI KHUYẾT GEN (N=100 BỆNH NHÂN TEST)")
    print(f"{'='*80}")

    for cov in coverages:
        print(f"\n>>> Tỷ lệ gen hiện diện: {cov*100:.0f}% ({int(cov*16335)} / 16335 gen)")
        for pol in policies:
            for m in models:
                P = engine.predict_proba(
                    model_name=m,
                    X_raw=X,
                    gene_coverage_rate=cov,
                    missing_policy=pol,
                    seed=args.seed,
                )
                yhat = P.argmax(axis=1) + 1
                acc = (yhat == y).mean() * 100.0
                print(f"  Policy={pol:12s} | {m:22s}: Accuracy = {acc:6.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tcga-predict",
        description="TCGA Pan-Cancer Standalone Inference & Biological Simulation CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: predict
    p_pred = subparsers.add_parser("predict", help="Suy luận ung thư cho 1 hoặc nhiều bệnh nhân")
    p_pred.add_argument("--input", "-i", required=True, help="Đường dẫn file JSON, CSV, TSV hoặc NPZ")
    p_pred.add_argument("--model", "-m", default="M6_auto", help="Tên mô hình (M1_multiLogReg, M2_msvm_reg100, M3_mlp_lowrank, M6_lightweight_no_m5, M6_auto)")
    p_pred.add_argument("--mode", default="native", choices=["native", "systemds"], help="Chế độ thực thi")
    p_pred.add_argument("--top-k", "-k", type=int, default=5, help="Số lớp top-K hiển thị")
    p_pred.add_argument("--coverage", type=float, default=1.0, help="Tỷ lệ gen sử dụng (0.0 < rate <= 1.0)")
    p_pred.add_argument("--dropout", type=float, default=0.0, help="Tỷ lệ random bỏ bớt gen (0.0 <= rate < 1.0)")
    p_pred.add_argument("--drop-genes", help="Danh sách gen chỉ định muốn bỏ đi (dạng comma 'TP53,KRAS' hoặc đường dẫn tệp)")
    p_pred.add_argument("--policy", default="linear_scale", choices=["zero_fill", "linear_scale", "sqrt_scale", "l2_renorm"], help="Chiến lược bù trừ khuyết gen")
    p_pred.add_argument("--tau", type=float, default=None, help="Hệ số điều chỉnh tiên nghiệm tau (prior adjustment override)")
    p_pred.add_argument("--seed", type=int, default=None, help="Seed ngẫu nhiên")
    p_pred.add_argument("--allow-missing", action="store_true", default=True, help="Cho phép suy luận ngay cả khi DataFrame thiếu gen")
    p_pred.add_argument("--no-allow-missing", action="store_false", dest="allow_missing", help="Báo lỗi nếu DataFrame thiếu bất kỳ gen nào")
    p_pred.add_argument("--max-display", type=int, default=10, help="Số bệnh nhân tối đa in ra terminal")
    p_pred.add_argument("--output", "-o", help="Đường dẫn lưu kết quả JSON")
    p_pred.add_argument("--models-dir", help="Thư mục chứa models_pretrained")
    p_pred.add_argument("--heavy-bundle", help="Thư mục chứa google_drive_heavy_bundle")

    # Subcommand: sample
    p_samp = subparsers.add_parser("sample", help="Lấy mẫu bệnh nhân thực tế hoặc sinh bệnh nhân giả lập")
    p_samp.add_argument("--type", choices=["real", "synthetic"], default="real", help="Loại bệnh nhân")
    p_samp.add_argument("--n", "--count", "-n", type=int, default=5, dest="n", help="Số lượng bệnh nhân cần lấy")
    p_samp.add_argument("--cancer", help="Lọc mã ung thư (ví dụ: LUAD, BRCA, GBM)")
    p_samp.add_argument("--variation", type=float, default=1.0, help="Độ biến thiên sinh học (cho synthetic)")
    p_samp.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    p_samp.add_argument("--predict", action="store_true", help="Chạy dự đoán đối chiếu ngay lập tức")
    p_samp.add_argument("--model", default="M6_lightweight_no_m5", help="Mô hình dùng khi có cờ --predict")
    p_samp.add_argument("--top-k", type=int, default=3, help="Top-K hiển thị")

    # Subcommand: benchmark-missing
    p_bench = subparsers.add_parser("benchmark-missing", help="Kiểm nghiệm thực tế tính năng khuyết gen")
    p_bench.add_argument("--model", default="all", help="Mô hình kiểm nghiệm (all hoặc tên mô hình)")
    p_bench.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")

    args = parser.parse_args()
    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "sample":
        cmd_sample(args)
    elif args.command == "benchmark-missing":
        cmd_benchmark_missing(args)


if __name__ == "__main__":
    main()
