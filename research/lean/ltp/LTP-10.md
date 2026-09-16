# LTP-10 — Generalização e encerramento do programa

- **Data**: 2026-09-16
- **Fase**: LTP-10 (fase final; funil encerrado)
- **Dependência**: LTP-00 a LTP-09 concluídas
- **Orçamento**: T1 em CPU; ~370 s de parede e pico de RSS 1,86 GB nesta fase;
  sem treino, sem GPU e sem serviços pagos
- **Gate**: avaliar generalização em ProofNet Lean 4 e miniCTX v2 após auditoria
  de licença e aquisição pinada; PutnamBench somente como stress test; abrir
  cada conjunto selado no máximo uma vez
- **Decisão**: **gate aprovado**; generalização medida onde havia ambiente
  compatível, bloqueios registrados, funil do programa encerrado

## Auditoria de licenças e aquisição

| Conjunto | Licença | Revisão pinada | Tamanho | Status |
| --- | --- | --- | --- | --- |
| ProofNet Lean 4 | MIT (LICENSE no repo) | commit `6deae98b3895`; Lean v4.20.0; mathlib v4.20.0 | 2,47 MB (tarball) | `acquired_verified` |
| miniCTX v2 | Apache-2.0 (card + LICENSE) | HF `91bd27f994c6fd6e3e1a85e7ad01c4ee0e6a01de` | 7,7 MB (16 arquivos) | `acquired_verified` |
| PutnamBench | **nenhuma licença detectada** | — | — | `blocked_on_license` |

- Manifestos: `research/lean/acquisitions/proofnet_lean4.json` e
  `minictx_v2.json`; schema estendido para registrar abertura selada única
  (`sealed_opening`), coberto por teste.
- PutnamBench não foi baixado: sem licença explícita, o stress test fica
  reservado. Todo uso futuro exige decisão de licença.

## Abertura única e portabilidade (ambiente nativo)

| Conjunto | Amostra | Portáveis | Taxa | Referência |
| --- | --- | --- | --- | --- |
| ProofNet (test) | 24 | 24 | **100%** | 12 metas |
| miniCTX mathlib (test) | 12 | 1 | **8,3%** | 1 meta |
| miniCTX outros projetos | — | — | — | bloqueado (ConNF, FLT, HepLean, Seymour, carleson, foundation exigem ambientes próprios) |

- Registro da abertura: `proofnet_lean4.jsonl` sha256
  `cffeb0fcc279a578a803ce7fd3e2583d688d0a3f1e0cdc38c96ff53a46270139`;
  `minictx-test/mathlib.jsonl` sha256
  `3aecd340e0a8bfaa07638f9ed4ed715543449713344400372311042ca32c2200`; cópias
  seladas em `.local/sealed/LTP-10/`, abertas uma única vez.
- ProofNet mira mathlib v4.20.0 e elaborou integralmente no ambiente nativo
  (v4.10.0-rc1 + mathlib 29dcec07) na amostra; o miniCTX usa contextos de
  projetos com histórico próprio e só 8,3% elaboraram.

## Linhas de referência (mesmo protocolo)

| Conjunto | Linha | pass@k | Chamadas | Tempo |
| --- | --- | --- | --- | --- |
| ProofNet | simbólica (k=5) | **0/12** | 60 | 85,7 s |
| ProofNet | ReProver tacgen (k=1) | **0/12** | 12 | 38,7 s |
| miniCTX mathlib | simbólica (k=1 portável) | **0/1** | 5 | 54,8 s |

Nenhuma meta foi resolvida exclusivamente por qualquer linha nesta fase.

## Errata registrada (LH-093 / LTP-05)

O pareamento estado↔teorema no smoke32 estava deslocado em um índice
(marcadores lidos no stream errado). Correção aplicada
(`scripts/ltp05_router.py`) e medição refeita nesta fase:

| Métrica | LTP-05 (original) | LTP-10 (corrigido) |
| --- | --- | --- |
| smoke32, modelo direto | 19/32 | **21/32** |
| smoke32, veredito do roteador | refutada | **não refutada** (27,0 vs 22,4 / 100 s) |
| val, roteador | refutada | refutada (inalterado) |
| união, roteador | não refutada | não refutada (4,47 vs 4,15) |

Artefatos originais preservados em `research/lean/runs/LTP-05/`; correção em
`research/lean/runs/LTP-10/`.

## Resultado final do programa (funil 100 → 30 → 10 → 3)

| Etapa | Resultado |
| --- | --- |
| LTP-00 laboratório formal | Lean v4.34.0 + mathlib pinados; gate adversarial 36/36 e 71/71 |
| LTP-01/02 aquisições | LeanDojo 4 + ReProver (LFS verificado, 299.637.760 parâmetros) |
| LTP-03 baselines | val: BM25 corrigido R@1 8,96%/3,43%; simbólicas e tacgen medidos |
| LTP-06 microteste | 100/100 com resultado: 8 válidas, 10 refutadas, 9 inconclusivas, 73 bloqueadas |
| LTP-07 Pareto | 8 promovidas (teto de 30 não atingido) |
| LTP-08 confirmação | 8/8 significativas após Holm |
| LTP-09 teste selado | LH-070 16/16, LH-069 16/16, LH-026 12,5% |
| LTP-10 generalização | ProofNet 100% portável (0/12 resolvidas); miniCTX mathlib 8,3% portável |

Limite honesto do programa:

- **Não existe provador neural treinado por este programa**; a pergunta primária
  (nano ≤100M atingir ≥85% do ReProver) permanece **não testada**, porque exige
  treino T2 que o orçamento das fases não autorizou.
- Os entregáveis científicos são o laboratório formal, os baselines
  reprodutíveis, 8 mecanismos confirmados e 3 finalistas validados no teste
  selado do benchmark 4, além do mapa completo de 73 hipóteses bloqueadas com
  pré-requisitos registrados.
- Nenhuma alegação de superioridade, de vantagem de tamanho ou de eficiência é
  feita; qualquer uma exigirá treino e comparação pareada pré-registrada.

## Comandos

```bash
.venv/bin/python scripts/ltp10_generalization.py --out research/lean/runs/LTP-10/generalization.json
.venv/bin/python scripts/ltp05_router.py smoke-states --out research/lean/runs/LTP-10/lh093_smoke_states.json
.venv/bin/python scripts/ltp05_router.py smoke-model --states research/lean/runs/LTP-10/lh093_smoke_states.json --out research/lean/runs/LTP-10/lh093_smoke_model.json
.venv/bin/python scripts/ltp05_router.py evaluate --smoke-model research/lean/runs/LTP-10/lh093_smoke_model.json --out research/lean/runs/LTP-10/lh093_router_corrected.json
```

## Gate da fase

| Critério | Resultado |
| --- | --- |
| Licenças auditadas (ProofNet MIT, miniCTX Apache-2.0) | sim |
| PutnamBench sem licença → bloqueado e não adquirido | sim |
| Revisões pinadas e checksums registrados | sim |
| Abertura selada única por conjunto | sim |
| Protocolo justo (hashes, timeout, pass@k, chamadas) | sim |
| Generalização medida onde há ambiente | sim (ProofNet e miniCTX-mathlib) |
| Bloqueios registrados sem descarte silencioso | sim |
| Sem treino/GPU/serviço pago | sim |

## Limitações

- Amostras de 24 e 12 metas; ProofNet sem split oficial no port.
- miniCTX fora de mathlib exige montar cada projeto (dependências, toolchain e
  licenças transitivas) antes de qualquer avaliação.
- As linhas de referência não têm um provador neural treinado; pass@k mede o
  piso simbólico e o tacgen de 2024, não um candidato nano.
- PutnamBench permanece reservado e bloqueado por licença.

## Decisão

Gate aprovado. LTP-10 encerrada e o funil do programa Lean Tiny Prover
concluído. Próxima etapa, fora deste programa: treino T2 autorizado,
pré-registro de comparação pareada contra ReProver e avaliação confirmatória.
