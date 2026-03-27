"""
02_eda_joins.py — Cobertura dos joins

Investiga: quantos CNPJs de food existem em cada tabela auxiliar.
Decisões: quais tabelas realmente agregam informação.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from utils import (
    load_food, load_places, load_delivery, load_eb,
    load_cnaes, load_divida, normalize_cnpj,
    section, subsection,
)


def main():
    food = load_food()
    food["cnpj_norm"] = normalize_cnpj(food["cnpj"])
    food_cnpjs = set(food["cnpj_norm"])

    auxiliares = {
        "places": load_places(),
        "delivery": load_delivery(),
        "eb": load_eb(),
        "cnaes": load_cnaes(),
        "divida": load_divida(),
    }

    # ── Cobertura geral ────────────────────────────────────────────────────
    section("1. COBERTURA DE CNPJS (food → tabelas auxiliares)")
    print(f"  CNPJs no food: {len(food_cnpjs)}\n")
    print(f"  {'Tabela':12s} | {'Match':>6s} | {'%':>6s} | {'Total tab':>10s} | {'CNPJs únicos tab':>16s}")
    print(f"  {'-'*12} | {'-'*6} | {'-'*6} | {'-'*10} | {'-'*16}")

    coverage = {}
    for name, df in auxiliares.items():
        df["cnpj_norm"] = normalize_cnpj(df["cnpj"])
        tab_cnpjs = set(df["cnpj_norm"])
        matched = food_cnpjs & tab_cnpjs
        pct = len(matched) / len(food_cnpjs) * 100
        coverage[name] = {"matched": len(matched), "pct": pct}
        print(f"  {name:12s} | {len(matched):6d} | {pct:5.1f}% | {len(df):10d} | {len(tab_cnpjs):16d}")

    # ── CNPJs sem nenhum match ─────────────────────────────────────────────
    section("2. CNPJS SEM MATCH EM NENHUMA TABELA AUXILIAR")
    all_aux_cnpjs = set()
    for df in auxiliares.values():
        all_aux_cnpjs |= set(df["cnpj_norm"])

    orphan = food_cnpjs - all_aux_cnpjs
    print(f"  CNPJs sem match em nenhuma tabela: {len(orphan)} ({len(orphan)/len(food_cnpjs)*100:.1f}%)")
    if orphan:
        print(f"  Exemplos: {list(orphan)[:5]}")

    # ── Registros múltiplos por CNPJ (1:N) ─────────────────────────────────
    section("3. RELAÇÕES 1:N (múltiplos registros por CNPJ)")
    for name, df in auxiliares.items():
        dups = df.groupby("cnpj_norm").size()
        multi = dups[dups > 1]
        if len(multi) > 0:
            print(f"  {name:12s} | {len(multi):5d} CNPJs com múltiplos registros | max: {multi.max()}")
        else:
            print(f"  {name:12s} | ✅ 1:1")

    # ── Cobertura por split ────────────────────────────────────────────────
    section("4. COBERTURA POR SPLIT (train vs test)")
    for split_name in ["train", "test"]:
        subsection(f"Split: {split_name}")
        subset_cnpjs = set(food[food["split"] == split_name]["cnpj_norm"])
        for name, df in auxiliares.items():
            tab_cnpjs = set(df["cnpj_norm"])
            matched = subset_cnpjs & tab_cnpjs
            pct = len(matched) / len(subset_cnpjs) * 100 if len(subset_cnpjs) > 0 else 0
            print(f"  {name:12s} | {len(matched):5d}/{len(subset_cnpjs):5d} ({pct:5.1f}%)")

    # ── Cobertura por subsegmento ──────────────────────────────────────────
    section("5. COBERTURA POR SUBSEGMENTO")
    for cls in sorted(food["subsegmento"].unique()):
        subsection(cls)
        cls_cnpjs = set(food[food["subsegmento"] == cls]["cnpj_norm"])
        for name, df in auxiliares.items():
            tab_cnpjs = set(df["cnpj_norm"])
            matched = cls_cnpjs & tab_cnpjs
            pct = len(matched) / len(cls_cnpjs) * 100 if len(cls_cnpjs) > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  {name:12s} | {pct:5.1f}% {bar}")


if __name__ == "__main__":
    main()
