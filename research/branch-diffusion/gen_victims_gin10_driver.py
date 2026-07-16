#!/usr/bin/env python3
"""gen_victims_gin10_driver.py — THIN DRIVER (no model/attack/defense code modified).

Honors the task request for 10-fold GIN victims on ENZYMES-continuous. The stock trainer
prepare_cont_gin.py hardcodes N_FOLDS=5 (imported from prepare, no CLI override). This driver
imports that trainer verbatim, overrides only the N_FOLDS constant -> 10 and the output dir ->
models_cont_10fold/ (NON-DESTRUCTIVE: leaves the validated 5-fold victims/metas in models_cont/
untouched), then calls its main(). Device = cuda (the trainer's consistent path; graphs live on
the global cuda DEVICE, matching the GCN-cont/5-fold victims).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = ["gen", "--seeds", "42,7,1027", "--epochs", "200",
            "--dataset", "ENZYMES", "--device", "cuda"]
import prepare_cont_gin as P  # noqa: E402

P.N_FOLDS = 10                                   # override 5 -> 10 (no file edited)
P.MODELS_CONT_DIR = HERE / "models_cont_10fold"  # separate dir => non-destructive
P.MODELS_CONT_DIR.mkdir(parents=True, exist_ok=True)
P.main()
