# Comparação de Experimentos

| # | Timestamp | Modelo | Features | Notas |
|---|-----------|--------|----------|-------|
| 1 | 2026-03-27T22:45 | LightGBM | mrmr_top25 | Optuna TPE 100 trials, 5-fold CV, F2 macro, class_weight=balanced |
| 2 | 2026-03-27T23:01 | CatBoost | mrmr_top25 | Optuna TPE 100 trials, 5-fold CV, F2 macro, auto_class_weights=Balanced |
| 3 | 2026-03-27T23:52 | LightGBM | mrmr_top41 | Optuna TPE 100 trials, 5-fold CV, F2 macro, class_weight=balanced |
| 4 | 2026-03-27T23:56 | CatBoost | mrmr_top41 | Optuna TPE 100 trials, 5-fold CV, F2 macro, auto_class_weights=Balanced |
| 5 | 2026-03-28T00:18 | LightGBM | mrmr_top36 | Optuna TPE 100 trials, 5-fold CV, F2 macro, class_weight=balanced |
| 6 | 2026-03-28T00:22 | CatBoost | mrmr_top36 | Optuna TPE 100 trials, 5-fold CV, F2 macro, auto_class_weights=Balanced |
| 7 | 2026-03-28T00:42 | LightGBM | mrmr_top39 | Optuna TPE 100 trials, 5-fold CV, F2 macro, class_weight=balanced |
| 8 | 2026-03-28T00:56 | CatBoost | mrmr_top39 | Optuna TPE 100 trials, 5-fold CV, F2 macro, auto_class_weights=Balanced |

## Métricas Detalhadas

### Experimento #1 — LightGBM
- **f2_macro**: 0.7261
- **f1_macro**: 0.7215
- **accuracy**: 0.75

### Experimento #2 — CatBoost
- **f2_macro**: 0.7276
- **f1_macro**: 0.7179
- **accuracy**: 0.7419

### Experimento #3 — LightGBM
- **f2_macro**: 0.8365
- **f1_macro**: 0.8327
- **accuracy**: 0.879

### Experimento #4 — CatBoost
- **f2_macro**: 0.8436
- **f1_macro**: 0.8436
- **accuracy**: 0.8871

### Experimento #5 — LightGBM
- **f2_macro**: 0.8411
- **f1_macro**: 0.8317
- **accuracy**: 0.871

### Experimento #6 — CatBoost
- **f2_macro**: 0.8155
- **f1_macro**: 0.8179
- **accuracy**: 0.871

### Experimento #7 — LightGBM
- **f2_macro**: 0.8285
- **f1_macro**: 0.8285
- **accuracy**: 0.871

### Experimento #8 — CatBoost
- **f2_macro**: 0.8281
- **f1_macro**: 0.8276
- **accuracy**: 0.871
