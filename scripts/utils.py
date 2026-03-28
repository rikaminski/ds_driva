"""
Funções utilitárias compartilhadas entre scripts do pipeline.
- Carregamento de dados
- Harmonização de CNPJ
- Experiment tracking (log_experiment, gerar comparison.md)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
EXPERIMENTS_FILE = RESULTS_DIR / "experiments.json"

# Garante que as pastas existem
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ── Carregamento de dados ──────────────────────────────────────────────────────

def load_food() -> pd.DataFrame:
    """Dataset rotulado com subsegmentos e split train/test."""
    return pd.read_parquet(DATA_DIR / "food.parquet")


def load_places() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "places.parquet")


def load_delivery() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "delivery.parquet")


def load_eb() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "eb.parquet")


def load_cnaes() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "cnaes.parquet")


def load_cnaes_desc() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "cnaes_desc.csv", sep=";")


def load_divida() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "divida.parquet")


# ── Harmonização de CNPJ ──────────────────────────────────────────────────────

import re as _re

_CNPJ_INVALID_RE = _re.compile(r"[^0-9eE.+\-/]")


def normalize_cnpj(series: pd.Series) -> pd.Series:
    """
    Converte CNPJ para string padronizada com 14 dígitos zero-padded.
    Retorna None para CNPJs corrompidos (caracteres inválidos, vazios).
    """
    def _norm(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        if not s:
            return None
        # Rejeita se contém caracteres inválidos (#, letras fora de E, etc.)
        if _CNPJ_INVALID_RE.search(s):
            return None
        # Trata notação científica e floats
        try:
            numeric = int(float(s))
            result = str(numeric).zfill(14)
            # Deve ter exatamente 14 dígitos numéricos
            if not result.isdigit():
                return None
            return result
        except (ValueError, OverflowError):
            pass
        # Remove pontuação restante
        s = s.replace(".", "").replace("-", "").replace("/", "")
        if not s.isdigit():
            return None
        return s.zfill(14)

    return series.map(_norm)


def validate_cnpj_column(df: pd.DataFrame, cnpj_col: str = "cnpj", name: str = "") -> pd.DataFrame:
    """
    Valida CNPJs de um DataFrame. Imprime registros corrompidos.
    Retorna o DataFrame filtrado (sem registros corrompidos).
    """
    normed = normalize_cnpj(df[cnpj_col])
    corrupted_mask = normed.isna() & df[cnpj_col].notna() & (df[cnpj_col].astype(str).str.strip() != "")
    n_corrupted = corrupted_mask.sum()
    n_empty = (df[cnpj_col].isna() | (df[cnpj_col].astype(str).str.strip() == "")).sum()

    if n_corrupted > 0 or n_empty > 0:
        label = f" ({name})" if name else ""
        print(f"  ⚠️ CNPJs com problemas{label}: {n_corrupted} corrompidos, {n_empty} vazios")
        for idx in df[corrupted_mask].index[:10]:
            print(f"    idx={idx} | cnpj={repr(df.loc[idx, cnpj_col])}")

    return df[~corrupted_mask].copy()


# ── Helpers de print ──────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Imprime um separador de seção formatado."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def subsection(title: str) -> None:
    print(f"\n--- {title} ---\n")


# ── Experiment tracking ───────────────────────────────────────────────────────

def log_experiment(
    model: str,
    feature_set: str,
    params: dict,
    cv_metrics: dict,
    test_metrics: dict,
    per_class: dict | None = None,
    notes: str = "",
) -> int:
    """
    Appenda um experimento no experiments.json e regenera comparison.md.
    Retorna o ID do experimento.
    """
    # Carrega existentes
    if EXPERIMENTS_FILE.exists():
        with open(EXPERIMENTS_FILE) as f:
            experiments = json.load(f)
    else:
        experiments = []

    exp_id = len(experiments) + 1
    experiment = {
        "id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "feature_set": feature_set,
        "params": params,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "per_class": per_class,
        "notes": notes,
    }
    experiments.append(experiment)

    # Salva JSON
    with open(EXPERIMENTS_FILE, "w") as f:
        json.dump(experiments, f, indent=2, ensure_ascii=False)

    # Regenera comparison.md
    _generate_comparison_md(experiments)

    print(f"\n✅ Experimento #{exp_id} salvo em {EXPERIMENTS_FILE}")
    return exp_id


def _generate_comparison_md(experiments: list[dict]) -> None:
    """Gera comparison.md a partir dos experimentos."""
    md_path = RESULTS_DIR / "comparison.md"
    lines = [
        "# Comparação de Experimentos\n",
        "| # | Timestamp | Modelo | Features | Notas |",
        "|---|-----------|--------|----------|-------|",
    ]

    for exp in experiments:
        ts = exp["timestamp"][:16]  # corta segundos
        line = f"| {exp['id']} | {ts} | {exp['model']} | {exp['feature_set']} | {exp.get('notes', '')} |"
        lines.append(line)

    # Adiciona métricas se existirem
    if experiments and experiments[0].get("test_metrics"):
        lines.append("\n## Métricas Detalhadas\n")
        for exp in experiments:
            tm = exp.get("test_metrics", {})
            lines.append(f"### Experimento #{exp['id']} — {exp['model']}")
            for k, v in tm.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
