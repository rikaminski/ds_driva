"""
05_eda_categoricals.py — Features categóricas

Investiga: crosstabs de categorias vs subsegmento.
Decisões: encoding strategy, features com sinal forte.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import pandas as pd
from utils import (
    load_food, load_places, load_delivery, load_eb, load_cnaes, load_cnaes_desc,
    normalize_cnpj, section, subsection,
)


def crosstab_top(merged: pd.DataFrame, cat_col: str, target: str = "subsegmento", top_n: int = 15) -> None:
    """Mostra crosstab de um campo categórico vs target (top N valores)."""
    top_vals = merged[cat_col].value_counts().head(top_n).index
    subset = merged[merged[cat_col].isin(top_vals)]
    ct = pd.crosstab(subset[cat_col], subset[target], margins=True)
    print(ct.to_string())
    print()


def main():
    food = load_food()
    food["cnpj_norm"] = normalize_cnpj(food["cnpj"])

    # ── main_category (Places) vs subsegmento ──────────────────────────────
    section("1. PLACES: main_category vs subsegmento")
    places = load_places()
    places["cnpj_norm"] = normalize_cnpj(places["cnpj"])
    places_dedup = places.drop_duplicates(subset="cnpj_norm", keep="first")
    merged_places = food[["cnpj_norm", "subsegmento"]].merge(
        places_dedup[["cnpj_norm", "main_category", "price_level"]], on="cnpj_norm", how="inner"
    )
    if "main_category" in merged_places.columns:
        crosstab_top(merged_places, "main_category")

    # ── price_level (Places) vs subsegmento ────────────────────────────────
    section("2. PLACES: price_level vs subsegmento")
    if "price_level" in merged_places.columns:
        ct = pd.crosstab(merged_places["price_level"], merged_places["subsegmento"], margins=True)
        print(ct.to_string())
        print()

    # ── categoria_principal_nome (iFood) vs subsegmento ────────────────────
    section("3. IFOOD: categoria_principal_nome vs subsegmento")
    delivery = load_delivery()
    delivery["cnpj_norm"] = normalize_cnpj(delivery["cnpj"])
    delivery_dedup = delivery.drop_duplicates(subset="cnpj_norm", keep="first")
    merged_delivery = food[["cnpj_norm", "subsegmento"]].merge(
        delivery_dedup[["cnpj_norm", "categoria_principal_nome"]].dropna(),
        on="cnpj_norm", how="inner"
    )
    if len(merged_delivery) > 0:
        crosstab_top(merged_delivery, "categoria_principal_nome")
    else:
        print("  ⚠️ Nenhum match entre food e delivery")

    # ── porte (EB) vs subsegmento ──────────────────────────────────────────
    section("4. EB: porte vs subsegmento")
    eb = load_eb()
    eb["cnpj_norm"] = normalize_cnpj(eb["cnpj"])
    eb_dedup = eb.drop_duplicates(subset="cnpj_norm", keep="first")
    merged_eb = food[["cnpj_norm", "subsegmento"]].merge(
        eb_dedup[["cnpj_norm", "porte", "situacao_cadastral", "matriz"]], on="cnpj_norm", how="inner"
    )
    if "porte" in merged_eb.columns:
        ct = pd.crosstab(merged_eb["porte"], merged_eb["subsegmento"], margins=True)
        print(ct.to_string())
        print()

    # ── situacao_cadastral vs subsegmento ──────────────────────────────────
    section("5. EB: situacao_cadastral vs subsegmento")
    if "situacao_cadastral" in merged_eb.columns:
        ct = pd.crosstab(merged_eb["situacao_cadastral"], merged_eb["subsegmento"], margins=True)
        print(ct.to_string())
        print()

    # ── matriz vs subsegmento ──────────────────────────────────────────────
    section("6. EB: matriz vs subsegmento")
    if "matriz" in merged_eb.columns:
        ct = pd.crosstab(merged_eb["matriz"], merged_eb["subsegmento"], margins=True)
        print(ct.to_string())
        print()

    # ── CNAE primário vs subsegmento ───────────────────────────────────────
    section("7. CNAE PRIMÁRIO vs subsegmento")
    cnaes = load_cnaes()
    cnaes["cnpj_norm"] = normalize_cnpj(cnaes["cnpj"])
    cnaes_desc = load_cnaes_desc()

    # Filtra primários
    primarios = cnaes[cnaes["tipo_cnae"] == "principal"].drop_duplicates(subset="cnpj_norm", keep="first")
    primarios = primarios.merge(
        cnaes_desc[["subclasse", "desc_subclasse"]].drop_duplicates(),
        left_on="cnae", right_on="subclasse", how="left"
    )
    merged_cnae = food[["cnpj_norm", "subsegmento"]].merge(
        primarios[["cnpj_norm", "cnae", "desc_subclasse"]], on="cnpj_norm", how="inner"
    )

    subsection("Top 20 CNAEs primários")
    top_cnaes = merged_cnae["cnae"].value_counts().head(20)
    for cnae_code, count in top_cnaes.items():
        desc = merged_cnae[merged_cnae["cnae"] == cnae_code]["desc_subclasse"].iloc[0] if len(merged_cnae[merged_cnae["cnae"] == cnae_code]) > 0 else "?"
        desc = str(desc)[:50]
        print(f"  {cnae_code:8d} | {count:4d} | {desc}")

    subsection("Crosstab CNAE primário (top 10) vs subsegmento")
    crosstab_top(merged_cnae, "cnae", top_n=10)

    # ── CNAEs secundários: contagem por CNPJ ──────────────────────────────
    section("8. CONTAGEM DE CNAEs SECUNDÁRIOS POR CNPJ")
    secundarios = cnaes[cnaes["tipo_cnae"] == "secundario"]
    sec_count = secundarios.groupby("cnpj_norm").size().reset_index(name="n_cnaes_sec")
    merged_sec = food[["cnpj_norm", "subsegmento"]].merge(sec_count, on="cnpj_norm", how="left")
    merged_sec["n_cnaes_sec"] = merged_sec["n_cnaes_sec"].fillna(0).astype(int)
    print(merged_sec.groupby("subsegmento")["n_cnaes_sec"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
