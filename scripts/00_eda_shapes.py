"""
00_eda_shapes.py — Estrutura dos dados

Investiga: shapes, dtypes reais, % missings, duplicatas por CNPJ.
Decisões: harmonização de tipos do CNPJ entre tabelas.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from utils import (
    load_food, load_places, load_delivery, load_eb,
    load_cnaes, load_cnaes_desc, load_divida,
    normalize_cnpj, validate_cnpj_column,
    section, subsection,
)


def main():
    datasets = {
        "food": load_food(),
        "places": load_places(),
        "delivery": load_delivery(),
        "eb": load_eb(),
        "cnaes": load_cnaes(),
        "cnaes_desc": load_cnaes_desc(),
        "divida": load_divida(),
    }

    # ── Shapes e dtypes ────────────────────────────────────────────────────
    section("1. SHAPES E DTYPES")
    for name, df in datasets.items():
        subsection(f"{name} — {df.shape[0]} linhas x {df.shape[1]} colunas")
        print(df.dtypes.to_string())
        print()

    # ── Tipo do CNPJ em cada tabela ────────────────────────────────────────
    section("2. TIPO DO CNPJ POR TABELA")
    for name, df in datasets.items():
        if "cnpj" in df.columns:
            dtype = df["cnpj"].dtype
            sample = df["cnpj"].dropna().iloc[:3].tolist() if len(df) > 0 else []
            print(f"  {name:12s} | dtype: {str(dtype):10s} | exemplos: {sample}")
    print()

    # ── Missings ───────────────────────────────────────────────────────────
    section("3. MISSINGS POR COLUNA (apenas colunas com >0%)")
    for name, df in datasets.items():
        missing = df.isnull().mean() * 100
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            subsection(f"{name}")
            for col, pct in missing.items():
                print(f"  {col:40s} {pct:6.2f}%")
        else:
            subsection(f"{name} — ✅ sem missings")

    # ── Duplicatas de CNPJ ─────────────────────────────────────────────────
    section("4. DUPLICATAS DE CNPJ")
    for name, df in datasets.items():
        if "cnpj" in df.columns:
            total = len(df)
            unique = df["cnpj"].nunique()
            dupes = total - unique
            pct = (dupes / total * 100) if total > 0 else 0
            status = "⚠️" if dupes > 0 else "✅"
            print(f"  {name:12s} | total: {total:6d} | únicos: {unique:6d} | dupes: {dupes:5d} ({pct:.1f}%) {status}")

    # ── Resumo rápido do food (dataset principal) ──────────────────────────
    section("5. RESUMO DO DATASET PRINCIPAL (food)")
    food = datasets["food"]
    print(f"  Colunas: {list(food.columns)}")
    print(f"  Shape:   {food.shape}")
    print(f"\n  Valores únicos por coluna:")
    for col in food.columns:
        print(f"    {col}: {food[col].nunique()} únicos")

    # ── Integridade de CNPJs ───────────────────────────────────────────────
    section("6. INTEGRIDADE DE CNPJs")
    for name, df in datasets.items():
        if "cnpj" in df.columns:
            normed = normalize_cnpj(df["cnpj"])
            total = len(df)
            valid = normed.notna().sum()
            invalid = total - valid
            # Separar: vazios vs corrompidos
            empty = (df["cnpj"].isna() | (df["cnpj"].astype(str).str.strip() == "")).sum()
            corrupted = invalid - empty
            status = "✅" if invalid == 0 else "⚠️"
            print(f"  {name:12s} | válidos: {valid:6d}/{total} | corrompidos: {corrupted} | vazios: {empty} {status}")
            if corrupted > 0:
                bad_mask = normed.isna() & df["cnpj"].notna() & (df["cnpj"].astype(str).str.strip() != "")
                for idx in df[bad_mask].index[:5]:
                    print(f"    → {repr(df.loc[idx, 'cnpj'])}")
    print()


if __name__ == "__main__":
    main()
