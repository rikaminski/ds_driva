---
description: Execução do pipeline de classificação food service (EDA → Features → Treino → Validação)
---

# Pipeline Food Service

## Pré-requisitos
- Ambiente `uv` ativo (`.venv`)
- Dados em `data/`

## Execução

### Etapa 1 — EDA
// turbo-all

1. Rodar cada script EDA sequencialmente e analisar outputs:
```bash
uv run python scripts/00_eda_shapes.py
uv run python scripts/01_eda_target.py
uv run python scripts/02_eda_joins.py
uv run python scripts/03_eda_numerics.py
uv run python scripts/04_eda_text.py
uv run python scripts/05_eda_categoricals.py
```

2. **AVALIAÇÃO**: Revisar resultados com o usuário. Validar se os achados do EDA sustentam a abordagem planejada. Ajustar rumo se necessário antes de avançar.

### Etapa 2 — Feature Engineering

3. Gerar features:
```bash
uv run python scripts/10_features.py
```

4. **AVALIAÇÃO**: Revisar features geradas com o usuário. Confirmar que as escolhas de encoding, imputação e transformações fazem sentido antes de treinar modelos.

### Etapa 3 — Treino + Tuning

5. Treinar modelos (cada script atualiza `scripts/results/experiments.json` e regenera `comparison.md` automaticamente):
```bash
uv run python scripts/20_train_lgbm.py
uv run python scripts/21_train_catboost.py
```

### Etapa 4 — Validação

6. Avaliação final:
```bash
uv run python scripts/30_evaluate.py
```

### Etapa 5 — Notebook Final

7. Consolidar resultados aprovados no `desafio.ipynb` (definido em conjunto com o usuário)
