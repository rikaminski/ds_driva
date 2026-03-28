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

2. **AVALIAÇÃO**: Revisar resultados com o usuário. Checklist de qualidade:
   - [ ] CNPJs corrompidos identificados e excluídos (script 00, seção 6)
   - [ ] Registros sem split contabilizados (script 01, seção 2a)
   - [ ] Campos mortos identificados (script 03, seção 1b)
   - [ ] tipo_cnae usando valores corretos: `'principal'` e `'secundario'`
   - [ ] Cobertura desigual por classe documentada
   - [ ] Split não-estratificado reconhecido (proposital do avaliador)
   
Ajustar rumo se necessário antes de avançar.

### Etapa 2 — Feature Engineering

3. Gerar features:
```bash
uv run python scripts/10_features.py
```

Decisões já tomadas:
- Incluir 57 registros sem split no treino
- Excluir 5 CNPJs corrompidos
- Excluir campos mortos (tempo_de_preparo, tempo_para_retirada)
- Usar seleção de features via mRMR ou sklearn feature_selection

4. **AVALIAÇÃO**: Revisar features geradas com o usuário. Confirmar encoding, imputação e transformações.

### Etapa 3 — Treino + Tuning

5. Treinar modelos com Optuna TPE + 5-fold StratifiedKFold:
```bash
uv run python scripts/20_train_lgbm.py
uv run python scripts/21_train_catboost.py
```

Cada script atualiza `scripts/results/experiments.json` e regenera `comparison.md`.
Métricas: F1 (balanceado) ou F2/G-mean (desbalanceado), média ± desvio padrão.

### Etapa 4 — Validação

6. Avaliação final (matrizes de confusão, SHAP, hiperparâmetros):
```bash
uv run python scripts/30_evaluate.py
```

### Etapa 5 — Notebook Final

7. Consolidar resultados aprovados no `desafio.ipynb` (definido em conjunto com o usuário)
