# Changelog de Experimentos

Registro de mudanças entre cada rodada de treino. O `experiments.json` guarda métricas; este arquivo explica **o que mudou e por quê**.

---

## Exp #1-2 — Baseline (v1)
**Feature set:** mRMR top 25 features (sem forçar nada)
- Features: 25 selecionadas de ~35 construídas
- Categorias Places: top-6 genéricas (incluía "escritório da empresa", "loja de roupa")
- Agregação Places: `drop_duplicates(keep='first')` — arbitrário para 1:N
- Sem keywords do EB (nome_fantasia/razao_social)
- **Resultado:** LightGBM F2=0.726, CatBoost F2=0.728

## Exp #3-4 — Features forçadas + keywords EB (v2)
**Mudanças:**
- Forçou features de escala: `porte_ordinal`, `is_matriz`, `capital_social_log`, `nome_contem_atacarejo`
- Forçou CNAEs primários: `cnae_is_4639701`, `cnae_is_4712100`, etc.
- Adicionou keywords do EB: `nome_contem_padaria`, `nome_contem_lanchonete`, `nome_contem_supermercado`, `nome_contem_confeitaria`
- mRMR K=40 (selecionou tudo, zero filtro real)
- **Por quê:** Confusão AS↔CaC era o maior erro. Features de escala/nome faltavam.
- **Resultado:** LightGBM F2=0.837, CatBoost F2=0.844 ★ (melhor até agora)

## Exp #5-6 — Limpeza de features + correção de agregação (v3)
**Mudanças:**
- Removeu `nome_contem_confeitaria` (3 matches = ruído)
- Removeu categorias lixo do Places: `escritório_da_empresa`, `loja_de_roupa`
- Adicionou categorias food-related: `atacadista`, `sorveteria`, `lanchonete` (com sub-valores agrupados)
- **Corrigiu agregação 1:N**: categorias e texto agora usam lógica OR (`groupby().max()`) em vez de `drop_duplicates(keep='first')`
- mRMR K=30 (seleção real, descartou 4 features)
- mRMR com `n_jobs=1` para evitar travamento
- **Por quê:** Bug silencioso: CNPJ `11224929000110` descartava dado real por keep='first'. Categorias "escritório" eram ruído.
- **Resultado:** LightGBM F2=0.841, CatBoost F2=0.816

## Exp #7-8 — n_filiais_rede + CNAEs sec discriminativos (v4) ❌ REVERTIDO
**Mudanças (revertidas):**
- Adicionou `n_filiais_rede`: contagem de CNPJs com mesma raiz_cnpj (proxy de rede)
- Adicionou 4 CNAEs secundários discriminativos (ratio >10x CaC/AS)
- **Por quê tentamos:** EDA mostrou que CaC tem p90=5 filiais vs AS max=2, e CNAEs sec como 4711302 tinham 35% CaC vs 0.9% AS.
- **Por quê revertemos:** F2 caiu de 0.844 → 0.828. Features que parecem discriminativas no EDA nem sempre melhoram tree models que já capturam o sinal indiretamente.
- **Resultado:** LightGBM F2=0.829, CatBoost F2=0.828

---

## Versão Final: v3 (Exp #5-6)
- 36 features selecionadas (12 forçadas + 24 mRMR)
- Melhor modelo: **CatBoost** do exp #4 (F2=0.844, Acc=88.7%)
- Features refeitas com correções da v3 (agregação correta, categorias food-related)
