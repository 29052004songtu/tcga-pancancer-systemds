"""
TCGA Pan-Cancer Standalone Inference & Research Toolkit
======================================================
Apache SystemDS Pretrained Transcriptomic Inference, Missing Gene Robustness,
and Biological Patient Simulation Package.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from .cancer_types import (
    TCGA_33_CANCER_INFO,
    get_cancer_code_by_id,
    get_cancer_name_by_id,
)
from .dml_engine import SystemDSLiveInferenceRunner
from .engine import (
    PretrainedTCGAInferenceEngine,
    apply_prior_adjustment,
)
from .missing_genes import (
    CoverageSafetyLevel,
    MissingGenePolicy,
    apply_missing_gene_policy,
    check_coverage_safety,
    simulate_gene_dropout,
)
from .sampler import PatientSampler

__version__ = "1.0.0"
__all__ = [
    "PretrainedTCGAInferenceEngine",
    "SystemDSLiveInferenceRunner",
    "apply_prior_adjustment",
    "PatientSampler",
    "MissingGenePolicy",
    "CoverageSafetyLevel",
    "check_coverage_safety",
    "simulate_gene_dropout",
    "apply_missing_gene_policy",
    "TCGA_33_CANCER_INFO",
    "get_cancer_code_by_id",
    "get_cancer_name_by_id",
]
