# LTP-05 — Pilotos LH-093, LH-024 e LH-001

- **Data**: 2026-09-15
- **Fase**: LTP-05 (única fase aberta; nenhuma fase seguinte iniciada)
- **Dependência**: LTP-00 a LTP-04 concluídas
- **Orçamento**: T0/T1 em CPU; sem treino, sem GPU e sem serviços pagos
- **Gate**: avaliar as condições de falsificação das três hipóteses com custos
  medidos e registrar o veredito de cada uma
- **Decisão**: LH-024 e LH-001 **refutadas**; LH-093 **inconclusiva** (refutada
  em conjuntos homogêneos, não refutada na distribuição mista medida)

## Entregas

| Entrega | Artefato |
| --- | --- |
| LH-001 bytes canônicos | `scripts/ltp05_canonical_bytes.py` + `runs/LTP-05/canonical_bytes.json` |
| LH-024 filtro por tipo | `scripts/ltp05_type_filter.py` + `runs/LTP-05/type_filter.json` |
| LH-093 roteador | `scripts/ltp05_router.py` + `runs/LTP-05/{smoke_states,smoke_model,router}.json` |
| Testes | `tests/test_lean_verify.py` (26 testes no total) |

## LH-093 — Roteador simbólico-neural (T1)

Política: tentar táticas simbólicas primeiro e escalar ao ReProver tacgen apenas
nas metas não resolvidas. Custo por meta medido nos artefatos (simbólicas no
LTP-00/LTP-04; modelo no LTP-03/LTP-04 e em geração própria para o smoke32).
Eficiência = metas resolvidas por 100 s.

| Conjunto | Metas | Modelo direto | Roteador | Overhead | Veredito |
| --- | --- | --- | --- | --- | --- |
| smoke32 (triviais) | 32 | 19 resolvidas, 42,0 s, ef. 45,2 | 32 resolvidas, 118,4 s, ef. 27,0 | +76,4 s | refutada |
| val (difíceis) | 16 | 1 resolvida, 436,7 s, ef. 0,229 | 1 resolvida, 807,4 s, ef. 0,124 | +370,7 s | refutada |
| união | 48 | 20 resolvidas, 478,7 s, ef. 4,18 | 33 resolvidas, 738,7 s, ef. 4,47 | +260,0 s | não refutada |

Leitura: o roteador aumenta cobertura (33 vs 20) porque as táticas simbólicas
fecham 100% do smoke32 enquanto o modelo fecha 59%; porém é mais lento por meta
resolvida no conjunto trivial e só gasta no conjunto difícil. Em cargas
homogêneas a hipótese falha; na distribuição mista medida ela se paga pelo
aumento de cobertura, não por eficiência por meta. Veredito: inconclusiva.

## LH-024 — Filtro por tipo antes da busca (T1)

Filtro aproximado por interseção de identificadores entre o objetivo (linha
`⊢`) e o código/nome da premissa; recall medido contra as premissas realmente
usadas nas anotações de val.

| Split | Passos | Recall | Passos com recall total | Redução do corpus | Veredito |
| --- | --- | --- | --- | --- | --- |
| random | 48 | 69,07% | 30/48 | 68,4% | refutada |
| novel_premises | 53 | 78,21% | 38/53 | 70,6% | refutada |

O gate exige recall ≥ 99%; o filtro descarta 21–31% das premissas usadas. A
redução de candidatos é alta, mas às custas de quebrar provas. Refutada.

## LH-001 — Estado em bytes canônicos (T0)

Normalização determinística (NFC, espaços, linhas vazias) e serialização em
bytes; comparação com bytes crus e com ids int32 (4 bytes por token).

| Split | Estados | Redução vs bruto | Redução vs int32 | Idempotência | Veredito |
| --- | --- | --- | --- | --- | --- |
| random | 2000 | 1,29% | 17,33% | 2000/2000 | refutada |
| novel_premises | 2000 | 3,00% | 20,39% | 2000/2000 | refutada |

O critério de ≥ 20% de memória não é atingido contra bytes crus (1–3%) e falha
no split `random` mesmo contra ids int32 (17,3%). A parte de não-inferioridade
de sucesso exigiria treino (T2), mas o critério de memória já reprova no T0.
Refutada.

## Atualização do registro de hipóteses

`research/lean/hypotheses.json`: LH-024 `rejected`, LH-001 `rejected`,
LH-093 `inconclusive`, todos com `evidence` apontando para este registro.
Resultados negativos preservados; nenhum artefato foi apagado.

## Gate da fase

| Critério | Resultado |
| --- | --- |
| LH-093 avaliada com custos medidos | sim (3 conjuntos) |
| LH-024 avaliada com recall medido | sim (2 splits) |
| LH-001 avaliada com memória medida | sim (2 splits) |
| Condições de falsificação aplicadas | sim (2 refutadas, 1 inconclusiva) |
| Nenhum conjunto selado aberto | sim |
| Sem treino, GPU ou serviço pago | sim |

## Limitações

- O roteador usa custos agregados de execuções anteriores, não um pipeline
  integrado em tempo real; a ordem simbólica→modelo é fixa.
- O filtro por tipo é aproximado por identificadores, não unificação real no
  Lean; um filtro semântico pode ter recall maior (trabalho futuro).
- A comparação de memória do LH-001 usa um tokenizador BPE de referência, não o
  tokenizador ByT5; para ByT5 (byte-level) a memória em bytes coincide com o
  tamanho em bytes.
- Subconjuntos pequenos (16–32 metas; 20 provas) e máquina com carga alta.

## Decisão

Gate aprovado. LTP-05 encerrada. A próxima fase (LTP-06, microteste T0/T1 de
LH-001…LH-100) **não** foi iniciada.

## Prompt para iniciar LTP-06 (não executar agora)

```text
Trabalhe no projeto /home/iiii/PESSOAL-PROJETOS-ALEXANDRE/quantun-ia como
executor do programa Lean Tiny Prover. Antes de agir, leia README.md,
docs/LEAN_RESEARCH_PROGRAM.md, research/lean/hypotheses.json,
research/lean/datasets.json, research/lean/baselines.json,
research/lean/acquisitions/*.json, research/lean/ltp/LTP-00.md,
research/lean/ltp/LTP-01.md, research/lean/ltp/LTP-02.md,
research/lean/ltp/LTP-03.md, research/lean/ltp/LTP-04.md,
research/lean/ltp/LTP-05.md e .local/PROMPT-CONTINUAR.md. Rode git status
--short, make lean-plan-check e make check. Verifique Lean/Lake/elan, GPU/VRAM,
RAM e disco. Execute SOMENTE LTP-06: rodar o microteste T0/T1 de LH-001…LH-100,
uma única alteração causal por hipótese, com smoke, limite de custo, resultado
(válida, refutada, inconclusiva ou bloqueada), log, seed, duração, pico de
RAM/VRAM e número de verificações; nenhuma hipótese pode ser descartada em
silêncio e resultados negativos não podem ser apagados; não abra conjuntos de
teste selados, não treine modelos caros e não use serviços pagos. Registre tudo
em research/lean/ltp/LTP-06.md e atualize o status das hipóteses apenas com
evidência; pare antes de LTP-07.
```

## Errata (registrada em LTP-10)

O pareamento entre estados de prova e teoremas no smoke32 estava deslocado em
um índice: o marcador `dbg_trace` era lido do stdout, mas o estado o precedia,
de modo que o modelo recebia o estado do teorema seguinte. Correção: parser
unificado que aceita marcadores no stdout ou no stderr e associa cada bloco ao
seu marcador. Medição corrigida em `research/lean/runs/LTP-10/`:

| Conjunto | Antes (LTP-05) | Corrigido (LTP-10) |
| --- | --- | --- |
| smoke32: modelo direto | 19/32 fechadas | **21/32 fechadas** |
| smoke32: veredito do roteador | refutada | **não refutada** (27,0 vs 22,4 por 100 s) |
| val: veredito do roteador | refutada | refutada (inalterado) |
| união: veredito do roteador | não refutada | não refutada (4,47 vs 4,15) |

Os artefatos originais do LTP-05 permanecem preservados; os números acima são a
referência.
