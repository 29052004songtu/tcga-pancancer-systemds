#!/usr/bin/env python3
"""
Điều Phối Thực Thi Trực Tiếp Qua Apache SystemDS (Live SystemDS DML Inference Runner).

Hỗ trợ suy luận trực tiếp bằng cách gọi Engine SystemDS (thông qua Java DMLScript CLI hoặc Python API),
đảm bảo:
- Tự động nhận diện Java 17 (.jdk17 hoặc system Java), không phụ thuộc hardcode user.
- Tự động tìm nạp SystemDS JAR và Native BLAS nếu có.
- Chạy độc lập hoàn toàn trên mọi máy tính có Java + SystemDS.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("tcga_dml_engine")


def find_java_binary() -> Optional[Path]:
    """Tìm binary Java khả dụng (ưu tiên OpenJDK 17)."""
    # 1. Kiểm tra JAVA_HOME
    java_home = os.getenv("JAVA_HOME")
    if java_home:
        cand = Path(java_home) / "bin" / "java"
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand

    # 2. Kiểm tra thư mục .jdk17 cục bộ hoặc cha
    curr = Path(__file__).resolve()
    for parent in [curr] + list(curr.parents):
        cand = parent / ".jdk17" / "bin" / "java"
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand

    # 3. Quét các đường dẫn cài đặt chuẩn của OpenJDK 17/11
    std_jdk_candidates = [
        Path.home() / ".jdk17" / "bin" / "java",
        Path.home() / ".jdks" / "jdk-17" / "bin" / "java",
        Path("/usr/lib/jvm/java-17-openjdk/bin/java"),
        Path("/usr/lib/jvm/java-17-openjdk-amd64/bin/java"),
        Path("/usr/lib/jvm/jdk-17/bin/java"),
        Path("/usr/lib/jvm/java-11-openjdk/bin/java"),
        Path("/usr/lib/jvm/java-11-openjdk-amd64/bin/java"),
    ]
    for cand in std_jdk_candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand

    # 4. Kiểm tra PATH hệ thống
    which_java = shutil.which("java")
    if which_java:
        return Path(which_java)

    return None


def find_systemds_classpath() -> Optional[str]:
    """Tìm classpath cho SystemDS (JARs trong site-packages hoặc thư mục build)."""
    # 1. Kiểm tra SYSTEMDS_ROOT
    sds_env = os.getenv("SYSTEMDS_ROOT")
    if sds_env:
        p = Path(sds_env)
        jar = p / "SystemDS.jar" if (p / "SystemDS.jar").exists() else p / "target" / "SystemDS.jar"
        if jar.exists():
            lib_dir = p / "lib"
            return f"{jar}:{lib_dir / '*'}" if lib_dir.exists() else str(jar)

    # 2. Thử qua gói python systemds
    try:
        import systemds
        sds_pkg_dir = Path(systemds.__file__).resolve().parent
        jar = sds_pkg_dir / "SystemDS.jar"
        if jar.exists():
            lib_dir = sds_pkg_dir / "lib"
            return f"{jar}:{lib_dir / '*'}" if lib_dir.exists() else str(jar)
    except ImportError:
        pass

    # 3. Quét quanh project root
    curr = Path(__file__).resolve()
    for parent in list(curr.parents):
        for sub in [".venv/lib/python3.11/site-packages/systemds", ".venv/lib/python3.12/site-packages/systemds", "systemds"]:
            cand_dir = parent / sub
            jar = cand_dir / "SystemDS.jar" if (cand_dir / "SystemDS.jar").exists() else cand_dir / "target" / "SystemDS.jar"
            if jar.exists():
                lib_dir = cand_dir / "lib"
                return f"{jar}:{lib_dir / '*'}" if lib_dir.exists() else str(jar)

    return None


class SystemDSLiveInferenceRunner:
    """Thực thi suy luận trực tiếp bằng Apache SystemDS DML."""

    def __init__(
        self,
        dml_script_path: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        java_bin: Optional[Union[str, Path]] = None,
        heap_gb: int = 4,
    ) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        self.models_dir = Path(models_dir) if models_dir is not None else base_dir / "models_pretrained"
        self.dml_script = (
            Path(dml_script_path)
            if dml_script_path is not None
            else base_dir / "dml" / "predict_tcga.dml"
        )
        self.java_bin = Path(java_bin) if java_bin is not None else find_java_binary()
        self.heap_gb = max(1, int(heap_gb))
        self.classpath = find_systemds_classpath()

    def is_available(self) -> bool:
        """Kiểm tra xem môi trường đã có đủ Java và SystemDS JAR để chạy DML hay chưa."""
        return self.java_bin is not None and self.classpath is not None and self.dml_script.exists()

    def run_inference_dml(
        self,
        X_raw: np.ndarray,
        model_name: str = "M1_multiLogReg",
        tau: float = 0.0,
        scale_factor: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Thực thi suy luận SystemDS DML trực tiếp trên ma trận X_raw.
        
        Returns:
            (probabilities_N_x_33, predicted_labels_1based, elapsed_time_ms)
        """
        if not self.is_available():
            raise RuntimeError(
                f"Không thể khởi chạy SystemDS DML: java_bin={self.java_bin}, "
                f"classpath={self.classpath is not None}, script={self.dml_script.exists()}."
            )

        with tempfile.TemporaryDirectory(prefix="sysds_infer_") as tmp_dir:
            tmp = Path(tmp_dir)
            in_x_csv = tmp / "X_input.csv"
            out_p_csv = tmp / "P_out.csv"
            out_y_csv = tmp / "y_out.csv"

            # Lưu input sang CSV và tạo .mtd
            np.savetxt(in_x_csv, X_raw, delimiter=",", fmt="%.8f")
            mtd_content = json.dumps({
                "data_type": "matrix",
                "value_type": "double",
                "rows": int(X_raw.shape[0]),
                "cols": int(X_raw.shape[1]),
                "format": "csv",
                "header": False,
            }, indent=2)
            (tmp / "X_input.csv.mtd").write_text(mtd_content, encoding="utf-8")

            # Chuẩn bị nvargs cho predict_tcga.dml
            nvargs = {
                "in_X": str(in_x_csv),
                "in_C": str(self.models_dir / "preprocessors" / "F0_full" / "centering.bin"),
                "in_S": str(self.models_dir / "preprocessors" / "F0_full" / "scalefactor.bin"),
                "in_Proj": str(self.models_dir / "preprocessors" / "F2_anova_5000" / "proj.bin"),
                "models_dir": str(self.models_dir),
                "model_name": model_name,
                "tau": str(float(tau)),
                "scale_factor": str(float(scale_factor)),
                "out_prob": str(out_p_csv),
                "out_pred": str(out_y_csv),
            }

            env = dict(getattr(os, "environ"))
            if self.java_bin is not None:
                java_home = self.java_bin.parent.parent
                env["JAVA_HOME"] = str(java_home)
                env["PATH"] = f"{self.java_bin.parent}:{env.get('PATH', '')}"

            cmd = [
                str(self.java_bin),
                "-Xms512m",
                f"-Xmx{self.heap_gb}g",
                "--add-modules=jdk.incubator.vector",
                "-cp",
                self.classpath,
                "org.apache.sysds.api.DMLScript",
                "-exec",
                "singlenode",
                "-f",
                str(self.dml_script),
                "-nvargs",
            ]
            for k, v in nvargs.items():
                cmd.append(f"{k}={v}")

            t0 = time.time()
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            elapsed_ms = (time.time() - t0) * 1000.0

            if proc.returncode != 0 or "An Error Occurred" in proc.stdout:
                hint = ""
                if "UnsupportedOperationException" in proc.stdout or "getSubject is not supported" in proc.stdout:
                    hint = (
                        "\n[GỢI Ý KHẮC PHỤC]: Apache SystemDS / Hadoop yêu cầu OpenJDK 11 hoặc OpenJDK 17. "
                        "Phiên bản Java mặc định của hệ thống không hỗ trợ getSubject (JEP 411/486 trên JDK 24+). "
                        "Vui lòng thiết lập biến môi trường trước khi chạy: export JAVA_HOME=/path/to/openjdk-17 "
                        "hoặc sử dụng chế độ mặc định '--mode native' để suy luận Python siêu tốc mà không cần Java."
                    )
                raise RuntimeError(
                    f"SystemDS DML execution failed (code={proc.returncode}):\n{proc.stdout[-2000:]}{hint}"
                )

            # Đọc kết quả
            P = np.loadtxt(out_p_csv, delimiter=",", dtype=np.float64)
            if P.ndim == 1:
                P = P.reshape(1, -1)
            yhat = np.loadtxt(out_y_csv, delimiter=",", dtype=np.float64).astype(np.int64).ravel()

            return P, yhat, elapsed_ms
