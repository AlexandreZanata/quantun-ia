# LTP-07 — Fronteira de Pareto e promoção

- **Data**: 2026-09-16
- **Fase**: LTP-07 (única fase aberta; LTP-08 não iniciada)
- **Dependência**: LTP-00 a LTP-06 concluídas
- **Orçamento**: T1 em CPU (BM25 corrigido; sem treino, sem GPU, sem serviços pagos)
- **Gate**: fronteira Pareto documentada com critérios congelados antes da
  seleção e no máximo 30 hipóteses promovidas
- **Decisão**: **gate aprovado**; 8 hipóteses promovidas (teto de 30 não
  atingido porque só 8 passaram no T0/T1)

## Objetivo

Reexecutar o BM25 com o fechamento transitivo de imports corrigido (errata do
LTP-06) e promover, pela fronteira de Pareto, as hipóteses elegíveis para a
triagem de LTP-08, sem abrir conjuntos selados e sem treinar modelos.

## BM25 corrigido (val, benchmark 4)

Candidatos agora incluem as premissas dos arquivos transitivamente importados
(mediana de 762 arquivos por arquivo, LTP-06), como no protocolo oficial do
ReProver.

| Split | R@1 errata | R@1 corrigido | R@10 errata | R@10 corrigido | MRR errata | MRR corrigido | Consultas |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 9,003 | **8,963** | 24,100 | **23,322** | 0,224 | **0,215** | 125 |
| novel_premises | 2,132 | **3,431** | 15,544 | **12,353** | 0,120 | **0,108** | 154 |

Os números corrigidos são a referência a partir desta fase; os artefatos do
LTP-03 permanecem preservados e marcados como superseded. Valores em
percentual para R@k. A hipótese LH-021 (BM25 como cérebro externo) continua
inconclusiva: falta o gerador neural integrado para medir sucesso fim a fim.

## Critérios de promoção (congelados antes da seleção)

- **Elegibilidade**: apenas hipóteses com microteste `valid` (gate T0/T1
  passado em LTP-06).
- **Eixos**: sucesso Lean (margem sobre o gate), classe de custo
  (negligenciável/offline/limitado por verificação), parâmetros (zero-aprendido
  vs alvo nano) e novidade (classe de capacidade).
- **Dominância**: ≥ em todos os eixos e > em pelo menos um, avaliada somente
  dentro da mesma classe de capacidade; mecanismos complementares não competem.
- **Teto**: 30. Se menos hipóteses passarem nos gates, não há preenchimento
  artificial.

## Promovidas (8)

| ID | Título | Classe de capacidade | Custo | Parâmetros | Evidência |
| --- | --- | --- | --- | --- | --- |
| LH-026 | Cache por assinatura de objetivo | objective_cache | negligenciável | zero | hit-rate 25,3% ≥ 5% |
| LH-061 | Compilação por fronteira sintática | syntax_boundary | verificação | zero | 4260/4260 alinhados |
| LH-068 | Microteoremas de regressão | regression_suite | offline | zero | 107 casos, 71 rejeições |
| LH-069 | Reexecução em ambiente limpo | clean_replay | verificação | zero | replay idêntico 8/8 |
| LH-070 | Hash imutável da afirmação | statement_hash | verificação | zero | 32/32 determinístico |
| LH-072 | Superamostragem de táticas raras | data_rebalancing | offline | nano | 104 táticas raras (<1%) |
| LH-074 | Corrupções mínimas de prova | corruption_negatives | offline | nano | 32/32 rejeições |
| LH-075 | Mineração de subteoremas | subtheorem_mining | offline | nano | 64,2% das provas com ≥2 passos |

Nenhum par de dominância foi detectado (`dominance_pairs: []`), pois cada
promovida cobre uma classe de capacidade distinta.

## Não promovidas

- **Rejeitadas/refutadas (10)**: LH-001, LH-004, LH-005, LH-008, LH-024,
  LH-035, LH-037, LH-059, LH-076, LH-088 — resultados negativos preservados.
- **Inconclusivas (9)**: LH-007, LH-021, LH-023, LH-039, LH-084, LH-085,
  LH-093, LH-095, LH-098 — seguem no registro com a medição parcial.
- **Bloqueadas (73)**: mantidas como `proposed` com o pré-requisito registrado;
  podem ser reavaliadas em fases T2.

## Comandos

```bash
.venv/bin/python scripts/ltp03_bm25.py --proofs 50 \
  --tokenizer-out .local/runs/lean/LTP-07/bm25_tokenizer.json \
  --out research/lean/runs/LTP-07/bm25_corrected.json
.venv/bin/python scripts/ltp07_frontier.py \
  --microtests research/lean/runs/LTP-06/microtests.json \
  --out research/lean/runs/LTP-07/frontier.json
.venv/bin/python scripts/validate_lean_research.py
```

## Gate da fase

| Critério | Resultado |
| --- | --- |
| BM25 corrigido reexecutado e registrado | sim |
| Critérios de promoção congelados antes da seleção | sim |
| Fronteira Pareto documentada com dominância | sim (0 pares, classes distintas) |
| Promoções ≤ 30 | sim (8) |
| Status atualizados com evidência | sim (`promoted` para as 8) |
| Nenhum test selado aberto; sem treino/GPU/serviço pago | sim |

## Limitações

- Métricas de gates distintos não são comparáveis entre si; por isso a
  dominância só vale dentro da mesma classe de capacidade.
- Os eixos de custo e parâmetros são ordinais, não medições contínuas
  uniformes; o registro por hipótese preserva a evidência original.
- A seleção não usou o conjunto de teste selado e nenhuma hipótese promovida
  resolveu teoremas em test; a "novidade de teoremas resolvidos" será medida na
  confirmação (LTP-08/09).
- O teto de 30 não foi atingido: 73 bloqueios de T2 e 9 inconclusões limitam o
  funil nesta etapa.

## Decisão

Gate aprovado. LTP-07 encerrada. A próxima fase (LTP-08, confirmação de no
máximo 10 sem consultar o teste) **não** foi iniciada.

## Prompt para iniciar LTP-08 (não executar agora)

```text
Trabalhe no projeto /home/iiii/PESSOAL-PROJETOS-ALEXANDRE/quantun-ia como
executor do programa Lean Tiny Prover. Antes de agir, leia README.md,
docs/LEAN_RESEARCH_PROGRAM.md, research/lean/hypotheses.json,
research/lean/datasets.json, research/lean/baselines.json,
research/lean/acquisitions/*.json, research/lean/ltp/LTP-00.md a LTP-07.md e
.local/PROMPT-CONTINUAR.md. Rode git status --short, make lean-plan-check e
make check. Verifique Lean/Lake/elan, GPU/VRAM, RAM e disco. Execute SOMENTE
LTP-08: confirmar no máximo 10 hipóteses promovidas no conjunto de validação
congelado, com cinco seeds e intervalos pareados por bootstrap sobre teoremas,
teste de McNemar para sucesso binário pareado e correção de Holm em alfa 0,05,
sem consultar o conjunto de teste selado; use o ambiente nativo e o BM25
corrigido como referência de recuperação; não treine acima do orçamento T1 e
não use serviços pagos. Registre objetivo, entregas, orçamento, gate, comandos,
hashes, métricas, limitações e decisão em research/lean/ltp/LTP-08.md; promova
no máximo três configurações Pareto apenas se os efeitos sobreviverem à
correção e pare antes de LTP-09.
```
