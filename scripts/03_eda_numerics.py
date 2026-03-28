"""
03_eda_numerics.py — Features numéricas

Investiga: estatísticas descritivas, outliers, distribuição por subsegmento.
Decisões: transformações, clipping, imputação.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import pandas as pd
from utils import (
    load_food, load_places, load_delivery, load_eb, load_divida,
    normalize_cnpj, section, subsection,
)


NUMERIC_COLS = {
    "places": [
        "score_vinculo", "total_hours", "hours_weekdays", "hours_weekend",
        "hours_daytime", "hours_nighttime", "rating", "user_ratings_total",
    ],
    "delivery": [
        "pedido_minimo", "tempo_de_preparo", "tempo_para_retirada",
        "AVALIACAO", "numero_avaliacoes",
    ],
    "eb": ["capital_social"],
    "divida": ["total_fgts", "total_previdenciaria", "total_nao_previdenciaria", "total"],
}


def main():
    food = load_food()
    food["cnpj_norm"] = normalize_cnpj(food["cnpj"])

    loaders = {
        "places": load_places,
        "delivery": load_delivery,
        "eb": load_eb,
        "divida": load_divida,
    }

    # ── Estatísticas descritivas ───────────────────────────────────────────
    section("1. ESTATÍSTICAS DESCRITIVAS POR TABELA")
    merged_dfs = {}
    for name, loader in loaders.items():
        df = loader()
        df["cnpj_norm"] = normalize_cnpj(df["cnpj"])

        # Para tabelas 1:N, pega o primeiro registro por CNPJ
        if df["cnpj_norm"].duplicated().any():
            df = df.drop_duplicates(subset="cnpj_norm", keep="first")

        cols = [c for c in NUMERIC_COLS.get(name, []) if c in df.columns]
        if not cols:
            continue

        subsection(f"{name} — colunas numéricas")
        print(df[cols].describe().round(2).to_string())

        # Merge com food para análise por subsegmento
        merged = food[["cnpj_norm", "subsegmento"]].merge(
            df[["cnpj_norm"] + cols], on="cnpj_norm", how="left"
        )
        merged_dfs[name] = (merged, cols)

    # ── Campos mortos (variância zero / 100% zeros) ────────────────────────
    section("1b. CAMPOS MORTOS (variância zero)")
    found_dead = False
    for name, (merged, cols) in merged_dfs.items():
        for col in cols:
            s = merged[col].dropna()
            if len(s) == 0:
                continue
            if s.std() == 0 or (s == 0).all():
                found_dead = True
                print(f"  ⚠️ {name}.{col:30s} | valor único: {s.unique()[0]} | NÃO usar como feature")
    if not found_dead:
        print("  ✅ Nenhum campo morto detectado")

    # ── Outliers (IQR) ─────────────────────────────────────────────────────
    section("2. OUTLIERS (método IQR)")
    for name, (merged, cols) in merged_dfs.items():
        subsection(name)
        for col in cols:
            s = merged[col].dropna()
            if len(s) == 0:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = s[(s < lower) | (s > upper)]
            pct = len(outliers) / len(s) * 100
            flag = "⚠️" if pct > 5 else ""
            print(f"  {col:30s} | outliers: {len(outliers):4d} ({pct:5.1f}%) | range: [{lower:.1f}, {upper:.1f}] {flag}")

    # ── Distribuição por subsegmento ───────────────────────────────────────
    section("3. MEDIANAS POR SUBSEGMENTO")
    for name, (merged, cols) in merged_dfs.items():
        subsection(name)
        grouped = merged.groupby("subsegmento")[cols].median()
        print(grouped.round(2).to_string())
        print()

    # ── Variância entre classes (sinal discriminativo?) ────────────────────
    section("4. VARIÂNCIA DAS MEDIANAS ENTRE CLASSES (features discriminativas)")
    print("  Features com alta variância entre classes são potencialmente discriminativas:\n")
    for name, (merged, cols) in merged_dfs.items():
        subsection(name)
        grouped = merged.groupby("subsegmento")[cols].median()
        # Coeficiente de variação das medianas entre classes
        for col in cols:
            medians = grouped[col].dropna()
            if len(medians) < 2 or medians.mean() == 0:
                continue
            cv = medians.std() / abs(medians.mean()) * 100
            signal = "🟢 forte" if cv > 50 else ("🟡 médio" if cv > 20 else "🔴 fraco")
            print(f"  {col:30s} | CV entre classes: {cv:6.1f}% | sinal: {signal}")


if __name__ == "__main__":
    main()
