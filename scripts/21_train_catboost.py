"""
21_train_catboost.py — Treino CatBoost com Optuna TPE

- 5-fold StratifiedKFold
- Optuna TPE Sampler: 100 trials
- Métrica: F2-score macro (dataset desbalanceado 6.2x)
- Report: média ± std para treino, validação e teste
- Salva modelo e atualiza experiments.json
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import json
import pickle
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import fbeta_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

from utils import section, subsection, log_experiment, RESULTS_DIR

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────────

N_FOLDS = 5
N_TRIALS = 100
BETA = 2  # F2-score
RANDOM_STATE = 42

LABEL_MAP = {
    "AUTO SERVICO": 0,
    "CASH AND CARRY": 1,
    "LANCHONETE": 2,
    "PADARIA": 3,
}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


# ── Métricas ───────────────────────────────────────────────────────────────────


def compute_metrics(y_true, y_pred):
    """Calcula F2 macro, F1 macro, accuracy."""
    return {
        "f2_macro": fbeta_score(y_true, y_pred, beta=BETA, average="macro"),
        "f1_macro": fbeta_score(y_true, y_pred, beta=1, average="macro"),
        "accuracy": (y_true == y_pred).mean(),
    }


# ── Objetivo Optuna ────────────────────────────────────────────────────────────


def objective(trial, X, y, skf):
    """Objetivo Optuna: F2-score macro médio dos folds."""
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "auto_class_weights": "Balanced",
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False,
    }

    val_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        # CatBoost lida com NaN nativamente
        model = CatBoostClassifier(**params)
        model.fit(X_fold_train, y_fold_train, eval_set=(X_fold_val, y_fold_val), early_stopping_rounds=50)

        y_pred = model.predict(X_fold_val).flatten()
        f2 = fbeta_score(y_fold_val, y_pred, beta=BETA, average="macro")
        val_scores.append(f2)

    return np.mean(val_scores)


# ── Pipeline principal ─────────────────────────────────────────────────────────


def main():
    # ── Carregar features ──
    section("1. CARREGANDO FEATURES")
    train_df = pd.read_parquet(RESULTS_DIR / "features_train.parquet")
    test_df = pd.read_parquet(RESULTS_DIR / "features_test.parquet")

    feature_cols = [c for c in train_df.columns if c not in ["cnpj_norm", "subsegmento", "split", "label"]]
    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Features: {feature_cols}")

    # ── Optuna ──
    section("2. OTIMIZAÇÃO COM OPTUNA (TPE)")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, skf),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params.update({
        "auto_class_weights": "Balanced",
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False,
    })

    subsection("Melhores hiperparâmetros")
    for k, v in best_params.items():
        print(f"  {k:25s}: {v}")
    print(f"\n  Melhor F2 médio (CV): {study.best_value:.4f}")

    # ── Retreinar com CV para métricas detalhadas ──
    section("3. MÉTRICAS POR FOLD (melhores hiperparâmetros)")

    train_metrics_per_fold = []
    val_metrics_per_fold = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = CatBoostClassifier(**best_params)
        model.fit(X_fold_train, y_fold_train, eval_set=(X_fold_val, y_fold_val), early_stopping_rounds=50)

        y_pred_train = model.predict(X_fold_train).flatten()
        train_m = compute_metrics(y_fold_train, y_pred_train)
        train_metrics_per_fold.append(train_m)

        y_pred_val = model.predict(X_fold_val).flatten()
        val_m = compute_metrics(y_fold_val, y_pred_val)
        val_metrics_per_fold.append(val_m)

        print(f"  Fold {fold_idx+1}: train F2={train_m['f2_macro']:.4f} | val F2={val_m['f2_macro']:.4f}")

    # Resumo CV
    subsection("Resumo CV (média ± std)")
    cv_metrics = {}
    for metric_name in ["f2_macro", "f1_macro", "accuracy"]:
        train_vals = [m[metric_name] for m in train_metrics_per_fold]
        val_vals = [m[metric_name] for m in val_metrics_per_fold]
        print(f"  {metric_name:12s} | train: {np.mean(train_vals):.4f} ± {np.std(train_vals):.4f} | "
              f"val: {np.mean(val_vals):.4f} ± {np.std(val_vals):.4f}")
        cv_metrics[f"train_{metric_name}_mean"] = round(float(np.mean(train_vals)), 4)
        cv_metrics[f"train_{metric_name}_std"] = round(float(np.std(train_vals)), 4)
        cv_metrics[f"val_{metric_name}_mean"] = round(float(np.mean(val_vals)), 4)
        cv_metrics[f"val_{metric_name}_std"] = round(float(np.std(val_vals)), 4)

    # ── Treinar modelo final no train completo ──
    section("4. MODELO FINAL (train completo)")
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X_train, y_train)

    y_pred_train_full = final_model.predict(X_train).flatten()
    train_full_m = compute_metrics(y_train, y_pred_train_full)
    print(f"  Train (full): F2={train_full_m['f2_macro']:.4f} | F1={train_full_m['f1_macro']:.4f} | Acc={train_full_m['accuracy']:.4f}")

    # ── Teste final ──
    section("5. TESTE FINAL")
    y_pred_test = final_model.predict(X_test).flatten()
    test_m = compute_metrics(y_test, y_pred_test)
    print(f"  Test: F2={test_m['f2_macro']:.4f} | F1={test_m['f1_macro']:.4f} | Acc={test_m['accuracy']:.4f}")

    subsection("Classification Report (Test)")
    target_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV.keys())]
    print(classification_report(y_test, y_pred_test, target_names=target_names))

    subsection("Confusion Matrix (Test)")
    cm = confusion_matrix(y_test, y_pred_test)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df.to_string())

    # ── Salvar modelo ──
    section("6. SALVANDO MODELO")
    model_path = RESULTS_DIR / "catboost_best.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"  Modelo salvo em: {model_path}")

    # ── Log experiment ──
    test_metrics = {
        "f2_macro": round(test_m["f2_macro"], 4),
        "f1_macro": round(test_m["f1_macro"], 4),
        "accuracy": round(test_m["accuracy"], 4),
    }

    log_experiment(
        model="CatBoost",
        feature_set=f"mrmr_top{len(feature_cols)}",
        params={k: v for k, v in best_params.items() if k != "allow_writing_files"},
        cv_metrics=cv_metrics,
        test_metrics=test_metrics,
        notes=f"Optuna TPE {N_TRIALS} trials, {N_FOLDS}-fold CV, F2 macro, auto_class_weights=Balanced",
    )


if __name__ == "__main__":
    main()
