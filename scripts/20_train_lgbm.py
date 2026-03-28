"""
20_train_lgbm.py — Treino LightGBM com Optuna TPE

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
import lightgbm as lgb

from utils import section, subsection, log_experiment, RESULTS_DIR

warnings.filterwarnings("ignore", category=UserWarning)
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
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1,
    }

    val_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_fold_train, y_fold_train)

        y_pred = model.predict(X_fold_val)
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
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1,
    })

    subsection("Melhores hiperparâmetros")
    for k, v in best_params.items():
        print(f"  {k:25s}: {v}")
    print(f"\n  Melhor F2 médio (CV): {study.best_value:.4f}")

    # ── Retreinar com CV para métricas detalhadas ──
    section("3. MÉTRICAS POR FOLD (melhores hiperparâmetros)")

    train_metrics_per_fold = []
    val_metrics_per_fold = []
    val_preds_all = np.full(len(y_train), -1, dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_fold_train, y_fold_train)

        # Métricas no fold de treino
        y_pred_train = model.predict(X_fold_train)
        train_m = compute_metrics(y_fold_train, y_pred_train)
        train_metrics_per_fold.append(train_m)

        # Métricas no fold de validação
        y_pred_val = model.predict(X_fold_val)
        val_m = compute_metrics(y_fold_val, y_pred_val)
        val_metrics_per_fold.append(val_m)
        val_preds_all[val_idx] = y_pred_val

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
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # Métricas no treino completo
    y_pred_train_full = final_model.predict(X_train)
    train_full_m = compute_metrics(y_train, y_pred_train_full)
    print(f"  Train (full): F2={train_full_m['f2_macro']:.4f} | F1={train_full_m['f1_macro']:.4f} | Acc={train_full_m['accuracy']:.4f}")

    # ── Teste final ──
    section("5. TESTE FINAL")
    y_pred_test = final_model.predict(X_test)
    test_m = compute_metrics(y_test, y_pred_test)
    print(f"  Test: F2={test_m['f2_macro']:.4f} | F1={test_m['f1_macro']:.4f} | Acc={test_m['accuracy']:.4f}")

    # Classification report
    subsection("Classification Report (Test)")
    target_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV.keys())]
    print(classification_report(y_test, y_pred_test, target_names=target_names))

    # Confusion matrix
    subsection("Confusion Matrix (Test)")
    cm = confusion_matrix(y_test, y_pred_test)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df.to_string())

    # ── Salvar modelo ──
    section("6. SALVANDO MODELO")
    model_path = RESULTS_DIR / "lgbm_best.pkl"
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
        model="LightGBM",
        feature_set=f"mrmr_top{len(feature_cols)}",
        params=best_params,
        cv_metrics=cv_metrics,
        test_metrics=test_metrics,
        notes=f"Optuna TPE {N_TRIALS} trials, {N_FOLDS}-fold CV, F2 macro, class_weight=balanced",
    )


if __name__ == "__main__":
    main()
