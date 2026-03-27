---
name: ds_driva_food_classification
description: Contexto e convenções do projeto de classificação de subsegmentos food service da Driva.
---

# Projeto ds_driva — Food Service Classification

## Contexto
Desafio de classificação multiclasse de estabelecimentos food service em 5 subsegmentos:
- CASH AND CARRY, AUTO SERVICO, LANCHONETE, CONFEITARIA, PADARIA

## Stack
- **Python 3.13** com **uv** como gerenciador de pacotes
- **Pandas** para manipulação de dados
- **LightGBM** e **CatBoost** como modelos candidatos
- **Optuna** para hyperparameter tuning
- **scikit-learn** para utilities de ML

## Convenções

### Estrutura
- Dados em `data/` (inclui `dicionario.md`)
- Scripts de desenvolvimento em `scripts/` (prefixo numérico: `00-05` EDA, `10` features, `20-21` treino, `30` avaliação)
- Resultados em `scripts/results/` (junto dos scripts)
- `experiments.json` é a fonte de verdade para tracking — atualizado automaticamente pelos scripts de treino via `log_experiment()`
- `comparison.md` é gerado automaticamente a partir do JSON
- `desafio.ipynb` é o entregável final, preenchido somente após validação nos scripts

### Código
- Rodar scripts via `uv run python scripts/XX_nome.py`
- Usar `print()` formatado para outputs de EDA
- `scripts/utils.py` contém funções compartilhadas (load, merge, `log_experiment()`)

### Processo
- **Etapas 1 (EDA) e 2 (Features) passam por rodadas de avaliação** — não avançamos para modelagem sem antes validar que a abordagem está correta
- Métricas e estratégias de avaliação serão definidas após EDA e feature engineering
