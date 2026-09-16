# LTP-06 — Microteste T0/T1 de LH-001…LH-100

- **Data**: 2026-09-15
- **Fase**: LTP-06 (única fase aberta; LTP-07 não iniciada)
- **Dependência**: LTP-00 a LTP-05 concluídas
- **Orçamento**: T0/T1 em CPU; sem treino; sem GPU; sem serviços pagos
- **Gate**: todas as 100 hipóteses com resultado explícito (válida, refutada,
  inconclusiva ou bloqueada); nenhuma descartada em silêncio; resultados
  negativos preservados
- **Decisão**: **gate aprovado**; funil segue para LTP-07 com 8 sobreviventes
  T0/T1

## Objetivo

Dar a cada hipótese do registro um microteste executável ou um bloqueio
justificado, com uma alteração causal única quando medida, registrando status,
gate, métricas, duração, pico de RAM, número de verificações e seed.

## Resultado consolidado

| Status | Hipóteses |
| --- | --- |
| **válida** (sobrevive ao T0/T1) | 8 |
| **refutada/rejeitada** | 10 |
| **inconclusiva** | 9 |
| **bloqueada** (razão registrada) | 73 |
| Total | 100 |

Nenhuma hipótese foi apagada ou descartada em silêncio. Status em
`research/lean/hypotheses.json` atualizado apenas com evidência desta fase:
`screening` (8), `rejected` (10), `inconclusive` (9), `proposed` com microteste
bloqueado (73).

## Hipóteses válidas no T0/T1

| ID | Título | Métrica principal |
| --- | --- | --- |
| LH-026 | Cache por assinatura de objetivo | hit-rate 25,3% (≥5%) |
| LH-061 | Compilação por fronteira sintática | 4260/4260 passos alinhados |
| LH-068 | Microteoremas de regressão | 107 casos, 71 rejeitados, gate ok |
| LH-069 | Reexecução em ambiente limpo | replay idêntico 8/8, 5 aceitas |
| LH-070 | Hash imutável da afirmação | 32/32 determinísticos, 6/6 mutações rejeitadas |
| LH-072 | Superamostragem de táticas raras | 104 táticas raras (<1%), top-5 = 55,8% |
| LH-074 | Corrupções mínimas de prova | 32/32 corrupções rejeitadas |
| LH-075 | Mineração de subteoremas | 64,2% das provas com ≥2 passos |

## Hipóteses refutadas/rejeitadas

| ID | Título | Evidência |
| --- | --- | --- |
| LH-001 | Bytes canônicos | redução 1,3%/3,0% vs bruto (LTP-05) |
| LH-004 | Ponteiros para subexpressões | redução de bigramas 21,9% (<25%) |
| LH-005 | Estado mínimo por dependência | perda por hipótese descartada 17,3% (>2 pp) |
| LH-008 | Ordem objetivo↔hipóteses | rank médio observado 10,3 > aleatório 7,7 |
| LH-024 | Filtro por tipo | recall 69,1%/78,2% (<99%) (LTP-05) |
| LH-035 | Macros de prova mineradas | cobertura 0,0% de bigramas frequentes |
| LH-037 | Cache de backtracking | 0 repetições estado-tática em 4260 pares |
| LH-059 | Vocabulário enxuto | redução de vocabulário 39,6% (<50%) |
| LH-076 | Deduplicação alfa | duplicatas 0,5% (<1%) |
| LH-088 | Filtro Bloom | 0 repetições evitáveis em 4260 pares |

## Hipóteses inconclusivas (medida parcial)

LH-007 (delta de estado, 25,4% de diferença média), LH-021 (BM25 medido; falta
gerador integrado), LH-023 (ranking por arquivo grosseiro demais), LH-039
(estagnação em 47,4% das provas; efeito em busca não medido), LH-084 (0,94% de
objetivos repetidos entre arquivos), LH-085 (mediana de 762 arquivos importados),
LH-093 (roteador de LTP-05), LH-095 e LH-098 (atribuição de custo de import
instável sob cache de página).

## Bloqueios (73 hipóteses)

Razões por família, registradas individualmente no JSON:

- arquitetura (ARC), objetivos (OBJ), compressão (CMP): exigem treino (T2);
- busca (SEA), feedback (FBK), dados (DAT), memória (MEM) e sistemas (SYS):
  exigem loop integrado, parser incremental, histórico do mathlib, unificação
  real ou pipeline de workers;
- representação (REP, exceto LH-004/005/007/008) e recuperação (RET, exceto
  LH-021/023/024/026): exigem AST no Lean, renomeação verificada, reranker
  treinado ou ablação com verificação por premissa.

Nenhum bloqueio é definitivo: cada um aponta o pré-requisito para reavaliação
em fases T2.

## Errata registrada

O campo `imports` do corpus do LeanDojo 4 contém **caminhos de arquivo**, não
nomes de módulo. O fechamento transitivo usado nos números de BM25 do LTP-03
ficou vazio, de modo que as premissas acessíveis eram apenas as do próprio
arquivo. A função `transitive_closure` foi corrigida nesta fase
(`scripts/ltp03_bm25.py`); os números de BM25 do LTP-03 devem ser reexecutados
antes de qualquer uso confirmatório. Os artefatos originais foram preservados.

## Infraestrutura

- Harness: `scripts/ltp06_microtests.py` (dispatching por hipótese, medição de
  duração/RAM, seed determinístico nulo).
- Resultados: `research/lean/runs/LTP-06/microtests.json` (100 linhas com
  status, gate, métricas, notas, evidência, duração, pico de RSS, verificações e
  seed).
- `imports` corrigidos também beneficiam LH-023/LH-085.

## Gate da fase

| Critério | Resultado |
| --- | --- |
| 100/100 hipóteses com resultado registrado | sim |
| Nenhuma descartada em silêncio | sim |
| Medições com dados reais e artefatos | 27 hipóteses não bloqueadas |
| Status atualizados apenas com evidência | sim |
| Resultados negativos preservados | sim |
| Nenhum test selado aberto; sem treino/GPU/serviço pago | sim |

## Limitações

- 73 hipóteses bloqueadas por orçamento/escopo T0/T1; os gates delas não foram
  avaliados, apenas o pré-requisito registrado.
- As medições usam val (desenvolvimento) e artefatos internos; não substituem
  confirmação em test selado (LTP-09).
- LH-023, LH-095/098 e LH-084 precisam de métodos melhores antes de qualquer
  decisão; foram mantidas inconclusivas de propósito.
- A máquina operou com carga alta; durações não são comparáveis entre fases.

## Decisão

Gate aprovado. LTP-06 encerrada. A próxima fase (LTP-07, promoção de no máximo
30 pela fronteira Pareto) **não** foi iniciada.

## Prompt para iniciar LTP-07 (não executar agora)

```text
Trabalhe no projeto /home/iiii/PESSOAL-PROJETOS-ALEXANDRE/quantun-ia como
executor do programa Lean Tiny Prover. Antes de agir, leia README.md,
docs/LEAN_RESEARCH_PROGRAM.md, research/lean/hypotheses.json,
research/lean/datasets.json, research/lean/baselines.json,
research/lean/acquisitions/*.json, research/lean/ltp/LTP-00.md a LTP-06.md e
.local/PROMPT-CONTINUAR.md. Rode git status --short, make lean-plan-check e
make check. Verifique Lean/Lake/elan, GPU/VRAM, RAM e disco. Execute SOMENTE
LTP-07: promover no máximo 30 hipóteses pela fronteira Pareto (sucesso Lean,
custo, parâmetros e novidade de teoremas resolvidos), reexecutando antes o BM25
corrigido do LTP-03 com o fechamento de imports; não abra conjuntos de teste
selados, não treine modelos acima do orçamento T1 e não use serviços pagos.
Registre objetivo, entregas, orçamento, gate, comandos, hashes, métricas,
limitações e decisão em research/lean/ltp/LTP-07.md; promova artefatos apenas se
a fronteira estiver documentada e pare antes de LTP-08.
```
