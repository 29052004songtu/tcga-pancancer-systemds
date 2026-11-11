#!/usr/bin/env python3
"""
TCGA 33 Cancer Types Metadata and Clinical Profiles.
Maps 1-based class IDs to official TCGA acronyms, disease names, and clinical groups.
Directly aligned with preprocessor label_mapping.json and metrics_and_overfitting_audit.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from typing import Dict, Any, Optional, Set

TCGA_33_CANCER_INFO: Dict[int, Dict[str, Any]] = {
    1: {"code": "LAML", "name": "acute myeloid leukemia", "tier": "Moderate", "samples_full": 173},
    2: {"code": "ACC", "name": "adrenocortical cancer", "tier": "Rare", "samples_full": 79},
    3: {"code": "BLCA", "name": "bladder urothelial carcinoma", "tier": "Abundant", "samples_full": 408},
    4: {"code": "LGG", "name": "brain lower grade glioma", "tier": "Abundant", "samples_full": 530},
    5: {"code": "BRCA", "name": "breast invasive carcinoma", "tier": "Abundant", "samples_full": 1104},
    6: {"code": "CESC", "name": "cervical & endocervical cancer", "tier": "Moderate", "samples_full": 307},
    7: {"code": "CHOL", "name": "cholangiocarcinoma", "tier": "Rare", "samples_full": 36},
    8: {"code": "COAD", "name": "colon adenocarcinoma", "tier": "Abundant", "samples_full": 454},
    9: {"code": "DLBC", "name": "diffuse large B-cell lymphoma", "tier": "Rare", "samples_full": 48},
    10: {"code": "ESCA", "name": "esophageal carcinoma", "tier": "Moderate", "samples_full": 185},
    11: {"code": "GBM", "name": "glioblastoma multiforme", "tier": "Moderate", "samples_full": 167},
    12: {"code": "HNSC", "name": "head & neck squamous cell carcinoma", "tier": "Abundant", "samples_full": 522},
    13: {"code": "KICH", "name": "kidney chromophobe", "tier": "Rare", "samples_full": 66},
    14: {"code": "KIRC", "name": "kidney clear cell carcinoma", "tier": "Abundant", "samples_full": 534},
    15: {"code": "KIRP", "name": "kidney papillary cell carcinoma", "tier": "Moderate", "samples_full": 291},
    16: {"code": "LIHC", "name": "liver hepatocellular carcinoma", "tier": "Moderate", "samples_full": 373},
    17: {"code": "LUAD", "name": "lung adenocarcinoma", "tier": "Abundant", "samples_full": 517},
    18: {"code": "LUSC", "name": "lung squamous cell carcinoma", "tier": "Abundant", "samples_full": 502},
    19: {"code": "MESO", "name": "mesothelioma", "tier": "Rare", "samples_full": 87},
    20: {"code": "OV", "name": "ovarian serous cystadenocarcinoma", "tier": "Moderate", "samples_full": 308},
    21: {"code": "PAAD", "name": "pancreatic adenocarcinoma", "tier": "Moderate", "samples_full": 179},
    22: {"code": "PCPG", "name": "pheochromocytoma & paraganglioma", "tier": "Moderate", "samples_full": 184},
    23: {"code": "PRAD", "name": "prostate adenocarcinoma", "tier": "Abundant", "samples_full": 498},
    24: {"code": "READ", "name": "rectum adenocarcinoma", "tier": "Moderate", "samples_full": 161},
    25: {"code": "SARC", "name": "sarcoma", "tier": "Moderate", "samples_full": 263},
    26: {"code": "SKCM", "name": "skin cutaneous melanoma", "tier": "Abundant", "samples_full": 473},
    27: {"code": "STAD", "name": "stomach adenocarcinoma", "tier": "Abundant", "samples_full": 415},
    28: {"code": "TGCT", "name": "testicular germ cell tumor", "tier": "Moderate", "samples_full": 156},
    29: {"code": "THYM", "name": "thymoma", "tier": "Rare", "samples_full": 120},
    30: {"code": "THCA", "name": "thyroid carcinoma", "tier": "Abundant", "samples_full": 513},
    31: {"code": "UCS", "name": "uterine carcinosarcoma", "tier": "Rare", "samples_full": 57},
    32: {"code": "UCEC", "name": "uterine corpus endometrioid carcinoma", "tier": "Abundant", "samples_full": 533},
    33: {"code": "UVM", "name": "uveal melanoma", "tier": "Rare", "samples_full": 80},
}

RARE_CANCER_CODES: Set[str] = {"DLBC", "UCS", "CHOL", "KICH", "UVM", "ACC", "MESO", "THYM"}
RARE_CANCER_IDS: Set[int] = {cid for cid, info in TCGA_33_CANCER_INFO.items() if info["code"] in RARE_CANCER_CODES}
ABUNDANT_CANCER_IDS: Set[int] = {cid for cid, info in TCGA_33_CANCER_INFO.items() if info["tier"] == "Abundant"}


def get_cancer_code_by_id(label_id_1based: int) -> str:
    """Trả về mã viết tắt TCGA (ví dụ: LUAD, BRCA) cho nhãn 1..33."""
    return TCGA_33_CANCER_INFO.get(int(label_id_1based), {}).get("code", "UNK")


def get_cancer_name_by_id(label_id_1based: int) -> str:
    """Trả về tên bệnh đầy đủ cho nhãn 1..33."""
    return TCGA_33_CANCER_INFO.get(int(label_id_1based), {}).get("name", f"Class_{label_id_1based}")


def get_id_by_cancer_code(code: str) -> Optional[int]:
    """Trả về ID (1..33) từ mã viết tắt (ví dụ: 'LUAD' -> 17)."""
    code_up = str(code).strip().upper()
    for cid, info in TCGA_33_CANCER_INFO.items():
        if info["code"] == code_up:
            return cid
    return None
