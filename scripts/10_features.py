"""
10_features.py — Feature Engineering

Constrói a feature matrix para treino e teste a partir dos achados do EDA.
Decisões aplicadas:
- 57 registros sem split → incluídos no treino
- 5 CNPJs corrompidos → excluídos
- Campos mortos (tempo_preparo, tempo_retirada) → excluídos
- Merge 1:N → agregação (mean/max/count) em vez de drop_duplicates(first)
- Seleção de features com mRMR

Saída: results/features_train.parquet, results/features_test.parquet
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from utils import (
    load_food, load_places, load_delivery, load_eb,
    load_cnaes, load_cnaes_desc, load_divida,
    normalize_cnpj, section, subsection,
    RESULTS_DIR,
)

# ══════════════════════════════════════════════════════════════════════
# 1. PREPARAÇÃO BASE
# ══════════════════════════════════════════════════════════════════════


def prepare_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepara datasets de treino e teste limpos."""
    food = load_food()
    food["cnpj_norm"] = normalize_cnpj(food["cnpj"])

    # Excluir CNPJs corrompidos (normalize_cnpj retorna None)
    n_before = len(food)
    food = food[food["cnpj_norm"].notna()].copy()
    n_excluded = n_before - len(food)
    print(f"  CNPJs corrompidos excluídos: {n_excluded}")

    # Incluir 57 sem split no treino
    no_split_mask = food["split"].isna() | (food["split"] == "")
    n_no_split = no_split_mask.sum()
    food.loc[no_split_mask, "split"] = "train"
    print(f"  Registros sem split incluídos no treino: {n_no_split}")

    train = food[food["split"] == "train"].copy()
    test = food[food["split"] == "test"].copy()

    print(f"  Train: {len(train)} | Test: {len(test)}")
    return train, test


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURES EB
# ══════════════════════════════════════════════════════════════════════


def build_eb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features do cadastro empresarial (EB)."""
    eb = load_eb()
    eb["cnpj_norm"] = normalize_cnpj(eb["cnpj"])
    eb = eb[eb["cnpj_norm"].notna()].copy()

    # EB é 1:1, sem necessidade de agregar
    eb_feats = eb[["cnpj_norm"]].copy()

    # capital_social → log1p (outliers 19.7%, skewed)
    eb_feats["capital_social_log"] = np.log1p(eb["capital_social"].clip(lower=0))

    # porte → ordinal
    porte_map = {"MICRO EMPRESA": 0, "PEQUENO PORTE": 1, "DEMAIS": 2}
    eb_feats["porte_ordinal"] = eb["porte"].map(porte_map)

    # is_matriz → binary
    eb_feats["is_matriz"] = eb["matriz"].astype(int)

    # situacao_ativa → binary (ATIVA=1, BAIXADA/SUSPENSA=0)
    eb_feats["situacao_ativa"] = (eb["situacao_cadastral"] == "ATIVA").astype(int)

    # Text keywords do nome_fantasia e razao_social
    nome = (
        eb["nome_fantasia"].fillna("").str.lower() + " " +
        eb["razao_social"].fillna("").str.lower()
    )
    eb_feats["nome_contem_atacarejo"] = nome.str.contains(
        r"atacad|atacarejo|atacadão|cash|carry|wholesale|varejo", na=False, regex=True
    ).astype(int)
    eb_feats["nome_contem_padaria"] = nome.str.contains(
        r"padaria|panificadora|panificação|pão|paes", na=False, regex=True
    ).astype(int)
    eb_feats["nome_contem_lanchonete"] = nome.str.contains(
        r"lanchonete|lanches|lanche|hamburguer|burger|burguer|fast.?food", na=False, regex=True
    ).astype(int)
    eb_feats["nome_contem_supermercado"] = nome.str.contains(
        r"supermercado|mercado|minimercado|mercearia|mercadinho", na=False, regex=True
    ).astype(int)

    merged = df.merge(eb_feats, on="cnpj_norm", how="left")

    # Flag de cobertura
    merged["has_eb"] = merged["capital_social_log"].notna().astype(int)

    # ── n_filiais_rede: quantos CNPJs compartilham a mesma raiz_cnpj ──
    # Proxy de escala: CaC (redes como Atacadão) tem p90=5 filiais vs AS sempre 1
    raiz_count = eb.groupby("raiz_cnpj").size().reset_index(name="n_filiais_rede")
    eb_raiz = eb[["cnpj_norm", "raiz_cnpj"]].merge(raiz_count, on="raiz_cnpj", how="left")
    eb_raiz = eb_raiz[["cnpj_norm", "n_filiais_rede"]]
    merged = merged.merge(eb_raiz, on="cnpj_norm", how="left")

    return merged


# ══════════════════════════════════════════════════════════════════════
# 3. FEATURES CNAES
# ══════════════════════════════════════════════════════════════════════


def build_cnaes_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dos CNAEs (primário como one-hot, contagem de secundários)."""
    cnaes = load_cnaes()
    cnaes["cnpj_norm"] = normalize_cnpj(cnaes["cnpj"])
    cnaes = cnaes[cnaes["cnpj_norm"].notna()].copy()

    # ── CNAE primário: one-hot dos top 5 ──
    primarios = cnaes[cnaes["tipo_cnae"] == "principal"].drop_duplicates(
        subset="cnpj_norm", keep="first"
    )

    # Top 5 CNAEs primários (cobrem ~90% dos dados)
    top_cnaes = primarios["cnae"].value_counts().head(5).index.tolist()
    cnae_df = primarios[["cnpj_norm", "cnae"]].copy()

    for cnae_code in top_cnaes:
        cnae_df[f"cnae_is_{cnae_code}"] = (cnae_df["cnae"] == cnae_code).astype(int)

    # cnae_other para CNAEs fora do top 5
    cnae_df["cnae_is_other"] = (~cnae_df["cnae"].isin(top_cnaes)).astype(int)
    cnae_df = cnae_df.drop(columns=["cnae"])

    # ── Contagem de CNAEs secundários ──
    secundarios = cnaes[cnaes["tipo_cnae"] == "secundario"]
    sec_count = secundarios.groupby("cnpj_norm").size().reset_index(name="n_cnaes_sec")

    # ── CNAEs secundários discriminativos para AS vs CaC ──
    # Baseado no EDA: esses CNAEs têm ratio >10x entre CaC e AS
    DISCRIM_CNAES = {
        "has_cnae_sec_4711302": 4711302,  # CaC 35.1% vs AS 0.9%
        "has_cnae_sec_4635499": 4635499,  # CaC 16.3% vs AS 1.1%
        "has_cnae_sec_4930202": 4930202,  # CaC 14.7% vs AS 1.7%
        "has_cnae_sec_4649408": 4649408,  # CaC 12.2% vs AS 0.0%
    }
    for feat_name, cnae_code in DISCRIM_CNAES.items():
        has_cnae = secundarios[secundarios["cnae"] == cnae_code][["cnpj_norm"]].drop_duplicates()
        has_cnae[feat_name] = 1
        df = df.merge(has_cnae, on="cnpj_norm", how="left")
        df[feat_name] = df[feat_name].fillna(0).astype(int)

    # Merge
    df = df.merge(cnae_df, on="cnpj_norm", how="left")
    df = df.merge(sec_count, on="cnpj_norm", how="left")
    df["n_cnaes_sec"] = df["n_cnaes_sec"].fillna(0).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════
# 4. FEATURES PLACES
# ══════════════════════════════════════════════════════════════════════


def build_places_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features do Google Places (agregação correta para 1:N)."""
    places = load_places()
    places["cnpj_norm"] = normalize_cnpj(places["cnpj"])
    places = places[places["cnpj_norm"].notna()].copy()

    # ── Numéricas: agregar por CNPJ (mean para duplicados) ──
    num_cols = ["score_vinculo", "rating", "user_ratings_total",
                "total_hours", "hours_daytime", "hours_nighttime"]
    places_num = places.groupby("cnpj_norm")[num_cols].mean().reset_index()
    rename_map = {c: f"places_{c}" for c in num_cols}
    places_num = places_num.rename(columns=rename_map)

    # ── main_category: food-related com lógica OR (se QUALQUER registro tem a cat, flag=1) ──
    # Categorias selecionadas com base no EDA (crosstab da seção 05)
    FOOD_CATS = {
        "places_cat_supermercado":  ["Supermercado"],
        "places_cat_mercado":      ["Mercado", "Mercado tradicional", "Mercearia"],
        "places_cat_atacadista":   ["Atacadista", "Hipermercado", "Supermercado de descontos"],
        "places_cat_padaria":      ["Padaria", "Panificadora de pães no vapor"],
        "places_cat_lanchonete":   ["Lanchonete", "Hamburgueria", "Restaurante"],
        "places_cat_sorveteria":   ["Sorveteria", "Loja de açaí", "Loja de sucos"],
    }

    cat_lower = places["main_category"].fillna("").str.strip()
    cat_feats_list = []
    for feat_name, values in FOOD_CATS.items():
        mask = cat_lower.isin(values)
        per_cnpj = places.loc[mask].groupby("cnpj_norm").size().reset_index(name="_cnt")
        per_cnpj[feat_name] = 1
        cat_feats_list.append(per_cnpj[["cnpj_norm", feat_name]])

    # Merge todas as categorias
    cat_df = places[["cnpj_norm"]].drop_duplicates()
    for cf in cat_feats_list:
        cat_df = cat_df.merge(cf, on="cnpj_norm", how="left")
    # NaN = 0 (não tem a categoria)
    for col in FOOD_CATS.keys():
        cat_df[col] = cat_df[col].fillna(0).astype(int)

    # ── Text: keywords do nome (OR across all records per CNPJ) ──
    name_lower = places["name"].fillna("").str.lower()

    text_patterns = {
        "places_nome_atacarejo": r"atacad|atacarejo|atacadão|cash|carry",
        "places_nome_padaria": r"padaria|panificadora|pão|paes",
        "places_nome_lanchonete": r"lanchonete|lanches|lanche|burger|hamburguer",
        "places_nome_supermercado": r"supermercado|mercado|minimercado|mercearia|mercadinho",
    }

    name_feats = places[["cnpj_norm"]].copy()
    for feat_name, pattern in text_patterns.items():
        name_feats[feat_name] = name_lower.str.contains(pattern, na=False, regex=True).astype(int)

    # Agregar por CNPJ usando max (= OR lógico)
    name_agg = name_feats.groupby("cnpj_norm").max().reset_index()

    # Merge tudo
    df = df.merge(places_num, on="cnpj_norm", how="left")
    df = df.merge(cat_df, on="cnpj_norm", how="left")
    df = df.merge(name_agg, on="cnpj_norm", how="left")

    # Flag de cobertura
    df["has_places"] = df["places_rating"].notna().astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════
# 5. FEATURES DELIVERY
# ══════════════════════════════════════════════════════════════════════


def build_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features do iFood (agregação para 1:N, exclui campos mortos)."""
    delivery = load_delivery()
    delivery["cnpj_norm"] = normalize_cnpj(delivery["cnpj"])
    delivery = delivery[delivery["cnpj_norm"].notna()].copy()

    # Numéricas: agregar (excluir campos mortos: tempo_de_preparo, tempo_para_retirada)
    num_cols = ["pedido_minimo", "AVALIACAO", "numero_avaliacoes"]
    del_num = delivery.groupby("cnpj_norm")[num_cols].agg({
        "pedido_minimo": "mean",
        "AVALIACAO": "mean",
        "numero_avaliacoes": "max",
    }).reset_index()

    rename_map = {
        "pedido_minimo": "delivery_pedido_minimo",
        "AVALIACAO": "delivery_avaliacao",
        "numero_avaliacoes": "delivery_num_avaliacoes",
    }
    del_num = del_num.rename(columns=rename_map)

    df = df.merge(del_num, on="cnpj_norm", how="left")

    # Flag de cobertura (fortemente discriminativo para LANCHONETE)
    df["has_ifood"] = df["delivery_pedido_minimo"].notna().astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════
# 6. FEATURES DÍVIDA
# ══════════════════════════════════════════════════════════════════════


def build_divida_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features da dívida ativa."""
    divida = load_divida()
    divida["cnpj_norm"] = normalize_cnpj(divida["cnpj"])
    divida = divida[divida["cnpj_norm"].notna()].copy()

    # Dívida é 1:1
    div_feats = divida[["cnpj_norm"]].copy()
    div_feats["divida_total_log"] = np.log1p(divida["total"].clip(lower=0))
    div_feats["divida_nao_prev_log"] = np.log1p(
        divida["total_nao_previdenciaria"].clip(lower=0)
    )

    df = df.merge(div_feats, on="cnpj_norm", how="left")
    df["has_divida"] = df["divida_total_log"].notna().astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════
# 7. SELEÇÃO DE FEATURES COM mRMR
# ══════════════════════════════════════════════════════════════════════


def select_features_mrmr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_features: int = 25,
) -> list[str]:
    """Seleciona features usando mRMR (Minimum Redundancy Maximum Relevance)."""
    try:
        from mrmr import mrmr_classif

        # mRMR requer dados sem NaN — imputar com -999 para seleção
        X_filled = X_train.fillna(-999)
        selected = mrmr_classif(X=X_filled, y=y_train, K=max_features, show_progress=False, n_jobs=1)
        return selected
    except Exception as e:
        print(f"  ⚠️ mRMR falhou ({e}), usando mutual_info_classif como fallback")
        from sklearn.feature_selection import mutual_info_classif

        X_filled = X_train.fillna(-999)
        mi = mutual_info_classif(X_filled, y_train, random_state=42)
        mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
        return mi_series.head(max_features).index.tolist()


# ══════════════════════════════════════════════════════════════════════
# 8. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════


# Colunas que não são features (metadata)
META_COLS = ["cnpj", "cnpj_norm", "subsegmento", "split"]

# Label encoding da target
LABEL_MAP = {
    "AUTO SERVICO": 0,
    "CASH AND CARRY": 1,
    "LANCHONETE": 2,
    "PADARIA": 3,
}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


def main():
    section("1. PREPARAÇÃO BASE")
    train, test = prepare_base()

    # ── Construir features ──
    section("2. CONSTRUÇÃO DE FEATURES")

    for label, df in [("train", train), ("test", test)]:
        subsection(f"Processando {label} ({len(df)} registros)")

        df = build_eb_features(df)
        df = build_cnaes_features(df)
        df = build_places_features(df)
        df = build_delivery_features(df)
        df = build_divida_features(df)

        if label == "train":
            train = df
        else:
            test = df

    # ── Identificar colunas de features ──
    feature_cols = [c for c in train.columns if c not in META_COLS]
    print(f"\n  Total de features construídas: {len(feature_cols)}")

    # ── Encode da target ──
    train["label"] = train["subsegmento"].map(LABEL_MAP)
    test["label"] = test["subsegmento"].map(LABEL_MAP)

    # ── Report de features ──
    section("3. REPORT DE FEATURES")
    print(f"  Features construídas: {len(feature_cols)}")
    print(f"\n  Missings por feature (train):")
    for col in sorted(feature_cols):
        pct = train[col].isna().mean() * 100
        if pct > 0:
            print(f"    {col:40s} {pct:5.1f}%")

    # ── Seleção mRMR ──
    section("4. SELEÇÃO DE FEATURES (mRMR)")
    X_train = train[feature_cols]
    y_train = train["label"]

    # Forçar inclusão de features críticas de escala/natureza (achados do EDA)
    forced_features = [
        "porte_ordinal", "is_matriz", "capital_social_log",
        "nome_contem_atacarejo", "n_cnaes_sec", "has_ifood",
        "n_filiais_rede",  # proxy de escala (rede de filiais)
    ]
    # Também incluir as features de CNAE primário se existirem
    cnae_feats = [c for c in feature_cols if c.startswith("cnae_is_")]
    forced_features.extend(cnae_feats)
    
    # Filtrar apenas as que realmente existem no DF
    forced_features = [f for f in forced_features if f in feature_cols]

    # Selecionar via mRMR (K=30 para real seleção)
    selected_mrmr = select_features_mrmr(X_train, y_train, max_features=30)
    
    # Unir forçadas + mRMR (sem duplicatas)
    selected = list(dict.fromkeys(forced_features + selected_mrmr))
    
    print(f"\n  Features selecionadas ({len(selected)}):")
    for i, feat in enumerate(selected, 1):
        status = "[FORCED]" if feat in forced_features else "[mRMR]"
        print(f"    {i:2d}. {feat:35s} {status}")

    # ── Salvar ──
    section("5. SALVANDO FEATURES")

    # Guardar apenas features selecionadas + metadata
    save_cols = selected + ["cnpj_norm", "subsegmento", "split", "label"]

    train_out = train[save_cols].copy()
    test_out = test[[c for c in save_cols if c in test.columns]].copy()

    train_path = RESULTS_DIR / "features_train.parquet"
    test_path = RESULTS_DIR / "features_test.parquet"

    train_out.to_parquet(train_path, index=False)
    test_out.to_parquet(test_path, index=False)

    print(f"  Train: {train_path} ({train_out.shape})")
    print(f"  Test:  {test_path} ({test_out.shape})")

    # Salvar lista de features para referência
    features_path = RESULTS_DIR / "selected_features.txt"
    with open(features_path, "w") as f:
        f.write("\n".join(selected))
    print(f"  Features: {features_path}")

    # ── Resumo final ──
    section("RESUMO")
    print(f"  Features construídas: {len(feature_cols)}")
    print(f"  Features selecionadas: {len(selected)}")
    print(f"  Train shape: {train_out.shape}")
    print(f"  Test shape:  {test_out.shape}")
    print(f"  Classes: {LABEL_MAP}")


if __name__ == "__main__":
    main()
