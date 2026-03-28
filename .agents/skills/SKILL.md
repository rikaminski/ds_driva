---
name: ds_driva_food_classification
description: Contexto, convenções e decisões do projeto de classificação de subsegmentos food service da Driva.
---

# Projeto ds_driva — Food Service Classification

## Contexto
Desafio de classificação multiclasse de estabelecimentos food service em 4 subsegmentos:
- CASH AND CARRY, AUTO SERVICO, LANCHONETE, PADARIA
- ⚠️ O enunciado menciona CONFEITARIA mas ela NÃO EXISTE nos dados

## Stack
- **Python 3.13** com **uv** como gerenciador de pacotes
- **Pandas** para manipulação de dados
- **LightGBM** e **CatBoost** como modelos candidatos
- **Optuna** (TPE Sampler) para hyperparameter tuning
- **mRMR** ou **sklearn feature_selection** para seleção de features
- **SHAP** para interpretabilidade
- **scikit-learn** para utilities de ML

## Decisões Tomadas (EDA)

### Dados
- **4 classes** (não 5): CONFEITARIA ausente nos dados
- **57 registros sem split → incluir no treino**: distribuição segue padrão do train, zero overlap de CNPJ com train/test, sem risco de leakage
- **5 CNPJs corrompidos → excluir**: `#`, `ABC`, `XX`, vazio — normalize_cnpj agora retorna None para esses
- **Split train/test NÃO estratificado**: desbalanceamento radical (CaC 45%→10% no teste). Respeitar split do avaliador como teste final, usar StratifiedKFold interno no treino
- **Desbalanceamento 6.2x** (CaC/PADARIA): usar `class_weight='balanced'` ou similar, não precisa de SMOTE
- **Campos mortos**: `tempo_de_preparo` e `tempo_para_retirada` (100% zero) — excluir
- **tipo_cnae**: valores são `'principal'` e `'secundario'` (NÃO `'primário'`)

### Cobertura das tabelas auxiliares
- **eb/cnaes** → 82.4% — base principal
- **places** → 58.0% — boa para AUTO SERVICO (84%) e PADARIA (81%)
- **delivery** → 18.3% — quase exclusivo de LANCHONETE (78%)
- **divida** → 19.6% — sinal forte mas cobertura baixa

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
- `scripts/utils.py` contém funções compartilhadas (load, merge, normalize_cnpj, validate_cnpj_column, `log_experiment()`)
- `normalize_cnpj()` retorna None para CNPJs corrompidos (caracteres inválidos, vazios)

### Processo
- **Etapas 1 (EDA) e 2 (Features) passam por rodadas de avaliação** — não avançamos para modelagem sem antes validar que a abordagem está correta
- Métricas: F1 ou F2-score dependendo do balanceamento, G-mean como alternativa
- Validação: 5-fold StratifiedKFold no treino, teste final no split do avaliador
