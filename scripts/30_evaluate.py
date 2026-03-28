"""
30_evaluate.py — Avaliação Final

- Carrega o melhor modelo (LightGBM e CatBoost)
- Matrizes de confusão (train, test) salvas em plots/
- Hiperparâmetros do melhor modelo
- SHAP: summary plot + bar plot
- Classification report detalhado
- Análise de erros
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import json
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    fbeta_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)

from utils import section, subsection, RESULTS_DIR, PLOTS_DIR, EXPERIMENTS_FILE

warnings.filterwarnings("ignore")

BETA = 2
LABEL_MAP = {
    "AUTO SERVICO": 0,
    "CASH AND CARRY": 1,
    "LANCHONETE": 2,
    "PADARIA": 3,
}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
TARGET_NAMES = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV.keys())]


def compute_metrics(y_true, y_pred):
    return {
        "f2_macro": fbeta_score(y_true, y_pred, beta=BETA, average="macro"),
        "f1_macro": fbeta_score(y_true, y_pred, beta=1, average="macro"),
        "accuracy": (y_true == y_pred).mean(),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: str):
    """Salva confusion matrix como imagem."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=TARGET_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {path}")


def evaluate_model(model, model_name: str, X_train, y_train, X_test, y_test):
    """Avaliação completa de um modelo."""
    section(f"AVALIAÇÃO: {model_name}")

    # ── Métricas ──
    subsection("Métricas")
    if hasattr(model, "predict"):
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # CatBoost retorna array 2D
        if hasattr(y_pred_train, "flatten"):
            y_pred_train = y_pred_train.flatten()
            y_pred_test = y_pred_test.flatten()

    train_m = compute_metrics(y_train, y_pred_train)
    test_m = compute_metrics(y_test, y_pred_test)

    print(f"  {'Métrica':12s} | {'Train':>8s} | {'Test':>8s}")
    print(f"  {'-'*12} | {'-'*8} | {'-'*8}")
    for metric in ["f2_macro", "f1_macro", "accuracy"]:
        print(f"  {metric:12s} | {train_m[metric]:8.4f} | {test_m[metric]:8.4f}")

    # ── Classification Report ──
    subsection("Classification Report (Test)")
    print(classification_report(y_test, y_pred_test, target_names=TARGET_NAMES))

    subsection("Classification Report (Train)")
    print(classification_report(y_train, y_pred_train, target_names=TARGET_NAMES))

    # ── Confusion Matrices ──
    subsection("Confusion Matrices")
    plot_confusion_matrix(
        y_train, y_pred_train,
        f"{model_name} — Train",
        str(PLOTS_DIR / f"cm_{model_name.lower()}_train.png"),
    )
    plot_confusion_matrix(
        y_test, y_pred_test,
        f"{model_name} — Test",
        str(PLOTS_DIR / f"cm_{model_name.lower()}_test.png"),
    )

    # Confusion matrix numérica no console
    cm = confusion_matrix(y_test, y_pred_test)
    cm_df = pd.DataFrame(cm, index=TARGET_NAMES, columns=TARGET_NAMES)
    print(f"\n  Confusion Matrix (Test):")
    print(cm_df.to_string())

    # ── Hiperparâmetros ──
    subsection("Hiperparâmetros")
    if hasattr(model, "get_params"):
        params = model.get_params()
    elif hasattr(model, "get_all_params"):
        params = model.get_all_params()
    else:
        params = {}

    for k, v in sorted(params.items()):
        if v is not None and str(v) != "":
            print(f"  {k:30s}: {v}")

    # ── SHAP ──
    subsection("SHAP — Feature Importance")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Summary plot (beeswarm)
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_test,
            feature_names=X_test.columns.tolist(),
            class_names=TARGET_NAMES,
            show=False, max_display=20,
        )
        plt.title(f"SHAP Summary — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        summary_path = str(PLOTS_DIR / f"shap_summary_{model_name.lower()}.png")
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  SHAP summary salvo: {summary_path}")

        # Bar plot (importância global)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # SHAP multi-class values can be a list (Lgbm) or 3D array (Catboost)
        if isinstance(shap_values, list):
            # LightGBM style
            importances = np.mean([np.abs(v).mean(axis=0) for v in shap_values], axis=0)
        elif len(shap_values.shape) == 3:
            # CatBoost style (samples, features, classes)
            importances = np.abs(shap_values).mean(axis=(0, 2))
        else:
            importances = np.abs(shap_values).mean(axis=0)

        importance_series = pd.Series(importances, index=X_test.columns).sort_values(ascending=True)
        importance_series.plot(kind="barh", ax=ax, color="#4C78A8")
        ax.set_title(f"SHAP Feature Importance — {model_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()
        bar_path = str(PLOTS_DIR / f"shap_importance_{model_name.lower()}.png")
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  SHAP importance salvo: {bar_path}")

        # Print text ranking
        print(f"\n  Top features por SHAP |value|:")
        for feat, val in importance_series.sort_values(ascending=False).head(15).items():
            print(f"    {feat:35s} {val:.4f}")

    except Exception as e:
        print(f"  ⚠️ SHAP falhou: {e}")

    # ── Análise de erros ──
    subsection("Análise de Erros (Test)")
    errors_mask = y_pred_test != y_test.values
    n_errors = errors_mask.sum()
    print(f"  Total de erros: {n_errors}/{len(y_test)} ({n_errors/len(y_test)*100:.1f}%)")

    if n_errors > 0:
        test_df_full = pd.read_parquet(RESULTS_DIR / "features_test.parquet")
        error_df = test_df_full[errors_mask].copy()
        error_df["predicted"] = pd.Series(y_pred_test[errors_mask]).map(LABEL_MAP_INV).values
        error_df["actual"] = error_df["subsegmento"]

        # Padrão de erros
        print(f"\n  Erros por classe real:")
        for cls in TARGET_NAMES:
            cls_errors = error_df[error_df["actual"] == cls]
            if len(cls_errors) > 0:
                preds = cls_errors["predicted"].value_counts()
                pred_str = ", ".join([f"{k}={v}" for k, v in preds.items()])
                print(f"    {cls:20s} → {len(cls_errors)} erros ({pred_str})")

    return test_m


def main():
    # ── Carregar dados ──
    section("CARREGANDO DADOS E MODELOS")
    train_df = pd.read_parquet(RESULTS_DIR / "features_train.parquet")
    test_df = pd.read_parquet(RESULTS_DIR / "features_test.parquet")

    feature_cols = [c for c in train_df.columns if c not in ["cnpj_norm", "subsegmento", "split", "label"]]
    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # ── Avaliar cada modelo salvo ──
    results = {}

    model_files = {
        "LightGBM": RESULTS_DIR / "lgbm_best.pkl",
        "CatBoost": RESULTS_DIR / "catboost_best.pkl",
    }

    for name, path in model_files.items():
        if path.exists():
            with open(path, "rb") as f:
                model = pickle.load(f)
            test_m = evaluate_model(model, name, X_train, y_train, X_test, y_test)
            results[name] = test_m
        else:
            print(f"\n  ⚠️ Modelo {name} não encontrado em {path}")

    # ── Comparação final ──
    if len(results) > 1:
        section("COMPARAÇÃO FINAL")
        print(f"  {'Modelo':12s} | {'F2':>8s} | {'F1':>8s} | {'Acc':>8s}")
        print(f"  {'-'*12} | {'-'*8} | {'-'*8} | {'-'*8}")
        best_model = None
        best_f2 = -1
        for name, metrics in results.items():
            f2 = metrics["f2_macro"]
            flag = " ★" if f2 > best_f2 else ""
            if f2 > best_f2:
                best_f2 = f2
                best_model = name
            print(f"  {name:12s} | {f2:8.4f} | {metrics['f1_macro']:8.4f} | {metrics['accuracy']:8.4f}{flag}")

        print(f"\n  🏆 Melhor modelo: {best_model} (F2={best_f2:.4f})")

    # ── Lista de plots gerados ──
    section("PLOTS GERADOS")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
