"""
01_eda_target.py — Variável alvo

Investiga: distribuição de classes, split train/test, estratificação.
Decisões: precisa oversample? class_weight basta?
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import pandas as pd
from utils import load_food, section, subsection


def main():
    food = load_food()

    # ── Distribuição geral de classes ──────────────────────────────────────
    section("1. DISTRIBUIÇÃO DE CLASSES (total)")
    counts = food["subsegmento"].value_counts()
    pcts = food["subsegmento"].value_counts(normalize=True) * 100
    for cls in counts.index:
        bar = "█" * int(pcts[cls] / 2)
        print(f"  {cls:20s} | {counts[cls]:5d} ({pcts[cls]:5.1f}%) {bar}")
    print(f"\n  Total: {len(food)}")
    print(f"  Classes: {food['subsegmento'].nunique()}")

    # ── Split train/test ───────────────────────────────────────────────────
    section("2. SPLIT TRAIN/TEST")
    split_counts = food["split"].value_counts(dropna=False)
    for s in split_counts.index:
        label = str(s) if pd.notna(s) else "<SEM SPLIT>"
        print(f"  {label:12s}: {split_counts[s]:5d} ({split_counts[s]/len(food)*100:.1f}%)")

    # ── Registros sem split ─────────────────────────────────────────────
    section("2a. REGISTROS SEM SPLIT")
    no_split = food[food["split"].isna() | (food["split"] == "")]
    train = food[food["split"] == "train"]
    test = food[food["split"] == "test"]
    print(f"  Total sem split: {len(no_split)} ({len(no_split)/len(food)*100:.1f}%)")
    if len(no_split) > 0:
        print(f"\n  Distribuição comparada:")
        print(f"  {'Classe':20s} | {'Sem split':>10s} | {'Train':>8s} | {'Test':>8s}")
        print(f"  {'-'*20} | {'-'*10} | {'-'*8} | {'-'*8}")
        for cls in counts.index:
            ns_pct = (no_split["subsegmento"] == cls).sum() / len(no_split) * 100 if len(no_split) > 0 else 0
            tr_pct = (train["subsegmento"] == cls).sum() / len(train) * 100 if len(train) > 0 else 0
            te_pct = (test["subsegmento"] == cls).sum() / len(test) * 100 if len(test) > 0 else 0
            closest = "~train" if abs(ns_pct - tr_pct) < abs(ns_pct - te_pct) else "~test"
            print(f"  {cls:20s} | {ns_pct:9.1f}% | {tr_pct:7.1f}% | {te_pct:7.1f}% ← {closest}")
        print(f"\n  → Distribuição segue o padrão do {'TRAIN' if True else 'TEST'} — recomenda-se incluir no treino.")

    # ── Distribuição por split ─────────────────────────────────────────────
    section("3. DISTRIBUIÇÃO DE CLASSES POR SPLIT")
    for split_name in ["train", "test"]:
        subsection(f"Split: {split_name}")
        subset = food[food["split"] == split_name]
        counts_s = subset["subsegmento"].value_counts()
        pcts_s = subset["subsegmento"].value_counts(normalize=True) * 100
        for cls in counts_s.index:
            bar = "█" * int(pcts_s[cls] / 2)
            print(f"  {cls:20s} | {counts_s[cls]:5d} ({pcts_s[cls]:5.1f}%) {bar}")
        print(f"  Subtotal: {len(subset)}")

    # ── Verificação de estratificação ──────────────────────────────────────
    section("4. VERIFICAÇÃO DE ESTRATIFICAÇÃO")
    print("  Proporção de cada classe no train vs test:\n")
    train = food[food["split"] == "train"]
    test = food[food["split"] == "test"]
    train_pcts = train["subsegmento"].value_counts(normalize=True) * 100
    test_pcts = test["subsegmento"].value_counts(normalize=True) * 100
    print(f"  {'Classe':20s} | {'Train %':>8s} | {'Test %':>8s} | {'Diff':>6s}")
    print(f"  {'-'*20} | {'-'*8} | {'-'*8} | {'-'*6}")
    for cls in counts.index:
        t_pct = train_pcts.get(cls, 0)
        te_pct = test_pcts.get(cls, 0)
        diff = abs(t_pct - te_pct)
        flag = " ⚠️" if diff > 3 else ""
        print(f"  {cls:20s} | {t_pct:7.1f}% | {te_pct:7.1f}% | {diff:5.1f}%{flag}")

    # ── Ratio de desbalanceamento ──────────────────────────────────────────
    section("5. RATIO DE DESBALANCEAMENTO")
    max_class = counts.max()
    min_class = counts.min()
    ratio = max_class / min_class
    print(f"  Maior classe:  {counts.idxmax()} ({max_class})")
    print(f"  Menor classe:  {counts.idxmin()} ({min_class})")
    print(f"  Ratio max/min: {ratio:.1f}x")
    if ratio > 10:
        print("  ⚠️ Desbalanceamento severo — considerar SMOTE ou class_weight")
    elif ratio > 3:
        print("  ⚠️ Desbalanceamento moderado — class_weight provavelmente suficiente")
    else:
        print("  ✅ Desbalanceamento leve")


if __name__ == "__main__":
    main()
