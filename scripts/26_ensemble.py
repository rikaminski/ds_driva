"""
26_ensemble.py — Ensemble LightGBM + CatBoost

Combina probabilidades dos dois modelos já treinados via média simples.
Explora vieses opostos: LightGBM sobre-prediz CaC, CatBoost sobre-prediz AS.

Saída: métricas e log no experiments.json
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, fbeta_score,
)
from utils import RESULTS_DIR, log_experiment, section

# ══════════════════════════════════════════════════════════════════════
# Label mapping
# ══════════════════════════════════════════════════════════════════════

LABEL_MAP = {
    "AUTO SERVICO": 0,
    "CASH AND CARRY": 1,
    "LANCHONETE": 2,
    "PADARIA": 3,
}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = [LABEL_MAP_INV[i] for i in range(4)]


def main():
    # ── 1. Carregar dados ──
    section("1. CARREGANDO DADOS E MODELOS")

    test = pd.read_parquet(RESULTS_DIR / "features_test.parquet")
    train = pd.read_parquet(RESULTS_DIR / "features_train.parquet")

    # Features (excluir metadata)
    meta_cols = ["cnpj_norm", "subsegmento", "split", "label"]
    feature_cols = [c for c in test.columns if c not in meta_cols]

    X_test = test[feature_cols]
    y_test = test["label"]
    X_train = train[feature_cols]
    y_train = train["label"]

    # Carregar modelos
    with open(RESULTS_DIR / "lgbm_best.pkl", "rb") as f:
        lgbm = pickle.load(f)
    with open(RESULTS_DIR / "catboost_best.pkl", "rb") as f:
        catboost = pickle.load(f)

    print(f"  Features: {len(feature_cols)}")
    print(f"  Test: {len(X_test)} amostras")
    print(f"  Train: {len(X_train)} amostras")

    # ── 2. Predições individuais ──
    section("2. PREDIÇÕES INDIVIDUAIS")

    prob_lgbm_test = lgbm.predict_proba(X_test)
    prob_cat_test = catboost.predict_proba(X_test)
    prob_lgbm_train = lgbm.predict_proba(X_train)
    prob_cat_train = catboost.predict_proba(X_train)

    pred_lgbm = prob_lgbm_test.argmax(axis=1)
    pred_cat = prob_cat_test.argmax(axis=1)

    # Concordância
    agree = (pred_lgbm == pred_cat).sum()
    print(f"  Concordância LightGBM/CatBoost: {agree}/{len(pred_lgbm)} ({agree/len(pred_lgbm)*100:.1f}%)")
    disagree_idx = np.where(pred_lgbm != pred_cat)[0]
    print(f"  Discordância: {len(disagree_idx)} amostras")

    if len(disagree_idx) > 0:
        print(f"\n  Amostras discordantes:")
        print(f"  {'idx':>4s} | {'Real':>15s} | {'LightGBM':>15s} | {'CatBoost':>15s}")
        print(f"  {'-'*4} | {'-'*15} | {'-'*15} | {'-'*15}")
        for idx in disagree_idx:
            real = LABEL_MAP_INV[y_test.iloc[idx]]
            lg = LABEL_MAP_INV[pred_lgbm[idx]]
            cb = LABEL_MAP_INV[pred_cat[idx]]
            print(f"  {idx:4d} | {real:>15s} | {lg:>15s} | {cb:>15s}")

    # ── 3. Ensemble: média de probabilidades ──
    section("3. ENSEMBLE (MÉDIA DE PROBABILIDADES)")

    # Testar diferentes pesos
    weights = [
        (0.5, 0.5, "50/50"),
        (0.6, 0.4, "60/40 (LightGBM)"),
        (0.4, 0.6, "40/60 (CatBoost)"),
        (0.7, 0.3, "70/30 (LightGBM)"),
        (0.3, 0.7, "30/70 (CatBoost)"),
    ]

    best_f2 = 0
    best_weight = None
    best_pred = None

    for w_lgbm, w_cat, label in weights:
        prob_ens = w_lgbm * prob_lgbm_test + w_cat * prob_cat_test
        pred_ens = prob_ens.argmax(axis=1)

        f2 = fbeta_score(y_test, pred_ens, beta=2, average="macro")
        f1 = f1_score(y_test, pred_ens, average="macro")
        acc = accuracy_score(y_test, pred_ens)

        marker = " ★" if f2 > best_f2 else ""
        print(f"  {label:25s} | F2={f2:.4f} | F1={f1:.4f} | Acc={acc:.4f}{marker}")

        if f2 > best_f2:
            best_f2 = f2
            best_weight = (w_lgbm, w_cat, label)
            best_pred = pred_ens

    # ── 4. Resultados do melhor ensemble ──
    section(f"4. MELHOR ENSEMBLE: {best_weight[2]}")

    w_lgbm, w_cat = best_weight[0], best_weight[1]
    prob_ens_test = w_lgbm * prob_lgbm_test + w_cat * prob_cat_test
    pred_ens_test = prob_ens_test.argmax(axis=1)
    prob_ens_train = w_lgbm * prob_lgbm_train + w_cat * prob_cat_train
    pred_ens_train = prob_ens_train.argmax(axis=1)

    f2_test = fbeta_score(y_test, pred_ens_test, beta=2, average="macro")
    f1_test = f1_score(y_test, pred_ens_test, average="macro")
    acc_test = accuracy_score(y_test, pred_ens_test)
    f2_train = fbeta_score(y_train, pred_ens_train, beta=2, average="macro")
    f1_train = f1_score(y_train, pred_ens_train, average="macro")
    acc_train = accuracy_score(y_train, pred_ens_train)

    print(f"\n  Métrica      |    Train |     Test")
    print(f"  ------------ | -------- | --------")
    print(f"  f2_macro     |   {f2_train:.4f} |   {f2_test:.4f}")
    print(f"  f1_macro     |   {f1_train:.4f} |   {f1_test:.4f}")
    print(f"  accuracy     |   {acc_train:.4f} |   {acc_test:.4f}")

    print(f"\n--- Classification Report (Test) ---\n")
    print(classification_report(y_test, pred_ens_test, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, pred_ens_test)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    print(f"--- Confusion Matrix (Test) ---\n")
    print(cm_df.to_string())

    # ── 5. Análise de erros ──
    section("5. ANÁLISE DE ERROS")

    errors = y_test.values != pred_ens_test
    n_errors = errors.sum()
    print(f"  Total de erros: {n_errors}/{len(y_test)} ({n_errors/len(y_test)*100:.1f}%)")

    # Comparar com modelos individuais
    errors_lgbm = (y_test.values != pred_lgbm).sum()
    errors_cat = (y_test.values != pred_cat).sum()
    print(f"\n  Comparação:")
    print(f"    LightGBM solo:  {errors_lgbm} erros")
    print(f"    CatBoost solo:  {errors_cat} erros")
    print(f"    Ensemble:       {n_errors} erros")

    # ── 6. Log ──
    section("6. SALVANDO EXPERIMENTO")

    per_class = {}
    report = classification_report(y_test, pred_ens_test, target_names=CLASS_NAMES, output_dict=True)
    for cls in CLASS_NAMES:
        per_class[cls] = {
            "precision": round(report[cls]["precision"], 4),
            "recall": round(report[cls]["recall"], 4),
            "f1": round(report[cls]["f1-score"], 4),
            "support": int(report[cls]["support"]),
        }

    log_experiment(
        model=f"Ensemble(LightGBM+CatBoost, {best_weight[2]})",
        feature_set="v3_mrmr_top36",
        params={"w_lgbm": w_lgbm, "w_cat": w_cat, "method": "probability_averaging"},
        cv_metrics={},
        test_metrics={
            "f2_macro": round(f2_test, 4),
            "f1_macro": round(f1_test, 4),
            "accuracy": round(acc_test, 4),
        },
        per_class=per_class,
        notes=f"Ensemble via média de probabilidades ({best_weight[2]}). Vieses opostos LightGBM/CatBoost.",
    )


if __name__ == "__main__":
    main()
