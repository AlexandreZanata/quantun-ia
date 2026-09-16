# LTP-08 — Confirmação das 8 promovidas

- **Data**: 2026-09-16
- **Fase**: LTP-08 (única fase aberta; LTP-09 não iniciada)
- **Dependência**: LTP-00 a LTP-07 concluídas
- **Orçamento**: T1 em CPU (14 passes de replay; 5 corrupções novas no Lean); sem
  treino, sem GPU, sem serviços pagos
- **Gate**: efeitos sobrevivem à correção de Holm em alfa 0,05 e no máximo três
  configurações Pareto são promovidas para o teste
- **Decisão**: **gate aprovado**; 8 confirmadas e 3 finalistas promovidas
  (LH-070, LH-069, LH-026)

## Objetivo

Confirmar as 8 hipóteses promovidas em LTP-07 no conjunto de validação congelado
(val do LeanDojo Benchmark 4, `random` + `novel_premises`), com cinco seeds,
intervalos de 95% por bootstrap sobre itens, teste de McNemar exato para
comparações pareadas e correção de Holm em alfa 0,05, sem consultar o test.

## Protocolo congelado

- Hash do protocolo: `26852e2054de046d…` (JSON completo em `confirmation.json`).
- Itens congelados: `val_steps_sha256 =
  d1aeb1e7e5b3292c8c70ae95b3a3501c48ea27c3eda2d8fe728c09ed6cb06cc3`
  (10.500 passos), 76 casos de corrupção/regressão, 2.240 provas para mineração,
  112 comparações de replay, 18 mutações de afirmação.
- Seeds: `[0, 1, 2, 3, 4]`; bootstrap: 10.000 reamostragens; alfa 0,05; Holm
  sobre os 8 p-valores.
- Orçamento de amostragem: minibatch 200 exemplos; corrupção 20; mineração 100;
  hash 16; fronteira 100; regressão 20.

## Resultados por hipótese

| ID | Métrica | Intervenção vs controle | IC / discordância | p (Holm) | Conclusão |
| --- | --- | --- | --- | --- | --- |
| LH-026 | taxa de acerto de cache | 27,46% vs limite 5% | IC inferior ≥ 5% | ~0 | confirmada |
| LH-061 | alinhamento de fronteira | 500/500 passos | IC ≥ 95% | 7,3e-12 | confirmada |
| LH-068 | detecção de regressões | 100/100 amostras | IC ≥ 95% | 5,9e-3 | confirmada |
| LH-069 | replay idêntico | 112/112 comparações (14 passes) | IC ≥ 95% | 3,2e-3 | confirmada |
| LH-070 | detecção de mutações | 80/80 amostras | IC ≥ 95% | 1,7e-2 | confirmada |
| LH-072 | inclusão de táticas raras | 26,02% vs 12,85% | IC do ganho > 0 | 3,4e-184 | confirmada |
| LH-074 | rejeição de corrupções | 100/100 amostras | IC ≥ 95% | 5,9e-3 | confirmada |
| LH-075 | rendimento de mineração | 66,6% vs limite 30% | IC ≥ 30% | 1,0e-63 | confirmada |

Observações metodológicas:

- **LH-026**: o par canônico vs string exata não teve discordância (0 pares) — os
  estados do val já são canônicos — mas o gate original (taxa absoluta ≥ 5%) é
  confirmado com folga (27,46%). O nulo pareado fica registrado.
- **LH-069**: poder insuficiente com 6 passes (48 comparações, p=0,085); foram
  executados 14 passes independentes, totalizando 112 comparações idênticas.
- **LH-068/LH-074**: compartilham os 76 itens de rejeição; 71 vêm de artefatos
  congelados do LTP-00 e 5 são corrupções novas verificadas no Lean nesta fase.
- **LH-070**: a detecção usa a camada textual determinística; a camada semântica
  foi confirmada em LTP-04 (32/32 determinísticos, 6/6 mutações rejeitadas).
- **LH-075**: rendimento medido como provas com ≥2 passos (proxy de submetas),
  sem verificação Lean de submetas individuais.

## Promoção (≤3 configurações Pareto)

Regra congelada: núcleo de validade formal primeiro (LH-070 hash imutável,
LH-069 replay limpo) e, em seguida, o mecanismo confirmado de menor custo
(LH-026 cache de objetivos).

| Finalista | Papel | Custo | Parâmetros |
| --- | --- | --- | --- |
| LH-070 | hash textual + semântico da afirmação | verificação | zero |
| LH-069 | replay limpo determinístico | verificação | zero |
| LH-026 | cache de objetivos | negligenciável | zero |

As outras cinco confirmadas (LH-061, LH-068, LH-072, LH-074, LH-075) permanecem
`confirmed` para reuso e combinação futura, sem entrare no teste selado.

## Comandos

```bash
.venv/bin/python scripts/ltp08_confirm.py --out research/lean/runs/LTP-08/confirmation.json
# passes extras de replay (LTP-04) para poder estatístico: 12 passes adicionais
.venv/bin/python scripts/ltp04_replay.py --statements research/lean/runs/LTP-04/statements_v2.jsonl \
  --hashes research/lean/runs/LTP-04/semantic_hashes.json \
  --runs-root .local/runs/lean/LTP-04/replay_root --out .local/runs/lean/LTP-08/replay_extra_<x>.json
```

## Gate da fase

| Critério | Resultado |
| --- | --- |
| Conjunto de validação congelado e hasheado | sim |
| Cinco seeds por hipótese | sim |
| Bootstrap de 95% sobre itens | sim |
| McNemar exato onde há pares | sim (LH-072; LH-026 pareado nulo registrado) |
| Holm em alfa 0,05 | sim (8/8 significativas) |
| Test selado não consultado | sim |
| Até 3 configurações promovidas | sim (3) |
| Sem treino/GPU/serviço pago | sim |

## Limitações

- A confirmação é em val (desenvolvimento); o test selado segue intocado e só
  será aberto em LTP-09 com os três finalistas.
- LH-026, LH-061, LH-068, LH-069 e LH-070 têm efeitos majoritariamente
  determinísticos; a variabilidade por seeds é pequena ou nula, e o poder
  estatístico de LH-069 veio de repetir passes, não de novas amostras.
- LH-072 e LH-075 usam proxies (inclusão em minibatch e passos ≥2) que não
  medem sucesso de prova fim a fim.
- As métricas continuam sem comparação pareada contra um provador neural
  treinado; isso pertence às fases de combinação/treino.

## Decisão

Gate aprovado. LTP-08 encerrada. A próxima fase (LTP-09, abertura única do teste
selado para no máximo três finalistas) **não** foi iniciada.

## Prompt para iniciar LTP-09 (não executar agora)

```text
Trabalhe no projeto /home/iiii/PESSOAL-PROJETOS-ALEXANDRE/quantun-ia como
executor do programa Lean Tiny Prover. Antes de agir, leia README.md,
docs/LEAN_RESEARCH_PROGRAM.md, research/lean/hypotheses.json,
research/lean/datasets.json, research/lean/baselines.json,
research/lean/acquisitions/*.json, research/lean/ltp/LTP-00.md a LTP-08.md e
.local/PROMPT-CONTINUAR.md. Rode git status --short, make lean-plan-check e
make check. Verifique Lean/Lake/elan, GPU/VRAM, RAM e disco. Execute SOMENTE
LTP-09: abrir uma única vez o conjunto de teste selado do LeanDojo Benchmark 4
para os três finalistas (LH-070, LH-069 e LH-026) em ambiente nativo, com as
mesmas afirmações hasheadas, mesmo timeout, pass@k e número de chamadas ao
verificador; reporte pass@k, chamadas, tempo, pico de RAM/VRAM e teoremas
resolvidos exclusivamente por finalista; não use serviços pagos, não faça
treino novo e não reabra o teste depois do resultado. Registre objetivo,
entregas, orçamento, gate, comandos, hashes, métricas, limitações e decisão em
research/lean/ltp/LTP-09.md; pare antes de LTP-10.
```
