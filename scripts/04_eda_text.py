"""
04_eda_text.py — Features textuais

Investiga: comprimento, top termos, termos discriminativos por classe.
Decisões: TF-IDF vale a pena? Quais campos usar?
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from collections import Counter

import pandas as pd
from utils import (
    load_food, load_places, load_delivery, load_eb,
    normalize_cnpj, section, subsection,
)

TEXT_COLS = {
    "places": ["name", "category", "main_category", "description", "review_keywords"],
    "delivery": ["ifood_nome", "descricao", "categoria_principal_nome"],
    "eb": ["razao_social", "nome_fantasia"],
}


def tokenize(text: str) -> list[str]:
    """Tokenização simples: lowercase, split em espaços, remove tokens curtos."""
    if not isinstance(text, str):
        return []
    return [t for t in text.lower().split() if len(t) > 2]


def main():
    food = load_food()
    food["cnpj_norm"] = normalize_cnpj(food["cnpj"])

    loaders = {
        "places": load_places,
        "delivery": load_delivery,
        "eb": load_eb,
    }

    # ── Comprimento dos campos textuais ────────────────────────────────────
    section("1. COMPRIMENTO DOS CAMPOS TEXTUAIS")
    merged_dfs = {}
    for name, loader in loaders.items():
        df = loader()
        df["cnpj_norm"] = normalize_cnpj(df["cnpj"])
        if df["cnpj_norm"].duplicated().any():
            df = df.drop_duplicates(subset="cnpj_norm", keep="first")

        cols = [c for c in TEXT_COLS.get(name, []) if c in df.columns]
        if not cols:
            continue

        subsection(name)
        for col in cols:
            lengths = df[col].dropna().astype(str).str.len()
            filled = df[col].notna().sum()
            total = len(df)
            print(f"  {col:35s} | preenchido: {filled:5d}/{total} ({filled/total*100:.1f}%) | "
                  f"len médio: {lengths.mean():.0f} | mediano: {lengths.median():.0f} | max: {lengths.max():.0f}")

        # Merge com food
        merged = food[["cnpj_norm", "subsegmento"]].merge(
            df[["cnpj_norm"] + cols], on="cnpj_norm", how="left"
        )
        merged_dfs[name] = (merged, cols)

    # ── Top termos por campo ───────────────────────────────────────────────
    section("2. TOP 15 TERMOS POR CAMPO")
    for name, (merged, cols) in merged_dfs.items():
        for col in cols:
            subsection(f"{name}.{col}")
            all_tokens = []
            for text in merged[col].dropna():
                all_tokens.extend(tokenize(str(text)))
            top = Counter(all_tokens).most_common(15)
            for token, count in top:
                print(f"  {token:25s} {count:5d}")

    # ── Termos discriminativos por subsegmento ─────────────────────────────
    section("3. TERMOS DISCRIMINATIVOS POR SUBSEGMENTO")
    print("  Termos que aparecem muito mais em uma classe que nas outras.\n")

    for name, (merged, cols) in merged_dfs.items():
        for col in cols:
            subsection(f"{name}.{col}")

            # Conta termos por classe
            class_counters = {}
            for cls in sorted(merged["subsegmento"].dropna().unique()):
                tokens = []
                for text in merged[merged["subsegmento"] == cls][col].dropna():
                    tokens.extend(tokenize(str(text)))
                class_counters[cls] = Counter(tokens)

            # Calcula "exclusividade": % do total de ocorrências que vem de uma classe
            all_terms = set()
            for c in class_counters.values():
                all_terms.update(c.keys())

            discriminative = []
            for term in all_terms:
                total = sum(c.get(term, 0) for c in class_counters.values())
                if total < 5:  # ignora termos raros
                    continue
                for cls, counter in class_counters.items():
                    ratio = counter.get(term, 0) / total
                    if ratio > 0.6:  # >60% das ocorrências vêm de uma classe
                        discriminative.append((term, cls, ratio, total))

            discriminative.sort(key=lambda x: -x[2])
            for term, cls, ratio, total in discriminative[:10]:
                print(f"  '{term:20s}' → {cls:20s} ({ratio*100:.0f}% de {total} ocorrências)")

            if not discriminative:
                print("  Nenhum termo fortemente discriminativo encontrado.")

    # ── Padrões nome → classe ──────────────────────────────────────────────
    section("4. PADRÕES DIRETOS: NOME CONTÉM PALAVRA-CHAVE DA CLASSE?")
    keywords = {
        "PADARIA": ["padaria", "panificadora", "pão", "paes"],
        "CONFEITARIA": ["confeitaria", "doces", "bolos", "cake"],
        "LANCHONETE": ["lanchonete", "lanches", "lanche", "hamburguer", "burger"],
        "CASH AND CARRY": ["atacado", "atacarejo", "atacadão", "cash", "carry", "wholesale"],
        "AUTO SERVICO": ["supermercado", "mercado", "minimercado", "mercearia", "supermarket"],
    }

    for name, (merged, cols) in merged_dfs.items():
        name_cols = [c for c in cols if "nome" in c.lower() or "name" in c.lower() or "razao" in c.lower() or "fantasia" in c.lower()]
        if not name_cols:
            continue

        subsection(f"Fonte: {name}")
        for col in name_cols:
            print(f"  Campo: {col}")
            for cls, kws in keywords.items():
                subset = merged[merged[col].notna()]
                matches = subset[subset[col].str.lower().str.contains("|".join(kws), na=False)]
                if len(matches) == 0:
                    continue
                correct = (matches["subsegmento"] == cls).sum()
                total_matches = len(matches)
                precision = correct / total_matches * 100 if total_matches > 0 else 0
                print(f"    Keywords {kws[:3]}... → {total_matches} matches | {correct} corretos ({precision:.0f}% precisão)")
            print()


if __name__ == "__main__":
    main()
