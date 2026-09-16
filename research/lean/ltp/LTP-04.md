# LTP-04 — LH-070 (hash imutável) e LH-069 (replay limpo)

- **Data**: 2026-09-15
- **Fase**: LTP-04 (única fase aberta; nenhuma fase seguinte iniciada)
- **Dependência**: LTP-00 a LTP-03 concluídas
- **Orçamento**: CPU; ~250 chamadas Lean no ambiente nativo; sem treino e sem GPU
- **Gate**: falha se qualquer mutação de afirmação passar pelo gate e se o replay
  limpo não for determinístico
- **Decisão**: **gate passou**

## Objetivo

Implementar LH-070 (hash imutável da afirmação, textual e semântico) e LH-069
(reexecução em ambiente limpo com determinismo comprovado) sobre o LeanDojo
Benchmark 4 val, no ambiente nativo `Lean v4.10.0-rc1 + mathlib 29dcec07`.

## Entregas

| Entrega | Artefato |
| --- | --- |
| Hash semântico | `build_semantic_source`, `parse_canonical_type`, `canonical_type_sha256` em `scripts/lean_verify.py` |
| Hash textual | `statement_sha256` (inalterado desde LTP-00) |
| Semântico em escala | `scripts/ltp04_semantic.py` + `research/lean/runs/LTP-04/semantic_results.json` e `semantic_hashes.json` |
| Replay limpo | `scripts/ltp04_replay.py` + `replay_results.json` |
| Rechecagem ReProver | `scripts/ltp04_reprover_recheck.py` + `reprover_recheck.json` |
| Extração corrigida | `scripts/ltp03_extract.py` + `extraction_report_v2.json` |
| Testes | `tests/test_lean_verify.py` (24 testes no total) |

## LH-070 — hash imutável da afirmação

Método: o cabeçalho original é preservado; a declaração é renomeada para um alvo
fixo, recebe `sorry` e o tipo é impresso com `set_option pp.all true` e
`#check @ltp_target`; o texto canônico (espaços colapsados, nome removido) é
hasheado com sha256. O hash textual continua sendo a primeira camada; o
semântico cobre a forma elaborada.

| Métrica | Resultado |
| --- | --- |
| Casos (16 por split) | 32 |
| Hash determinístico em dois runs limpos | **32/32 (100%)** |
| Mutações testadas | 6 |
| Mutações rejeitadas | **6/6** |
| Mutações com hash divergente (compilaram) | 3 (`add_to_mul`, `le_to_lt`, `le_to_lt`) |
| Mutações rejeitadas por falha de elaboração | 3 (`numeral_0_to_1` ×2, `le_to_lt`) |
| Parede total | 310 s |

## LH-069 — reexecução em ambiente limpo

Dois passes independentes, cada um com diretório e processo novos; comparação de
veredito, razão, axiomas e hash semântico caso a caso.

| Classe | Casos | Resultado |
| --- | --- | --- |
| Prova de candidato aceita (ReProver, `LieSubmodule.coe_toSubmodule_mk`) | 1 | aceita nos dois passes |
| Provas originais do mathlib (2 por split) | 4 | aceitas nos dois passes |
| Mutação de afirmação com prova válida | 1 | rejeitada nos dois passes |
| Provas inválidas (`exact (0 : Nat)`) | 2 | rejeitadas nos dois passes |
| Replay idêntico entre passes | — | **sim (8/8, sem divergências)** |
| Parede total | — | 53 s |

## Errata do LTP-03 (correções aplicadas)

1. **Extração**: aceitava o primeiro `:= by` dentro de provas term-mode/`calc`,
   truncando afirmações. Agora só aceita o primeiro `:=` de nível zero seguido
   de `by`; 6 casos de `random` e 2 de `novel_premises` foram excluídos
   (39→32 e 35→32 tentativas).
2. **Opções**: `autoImplicit`/`maxHeartbeats` eram inseridas antes da declaração,
   quebrando cabeçalhos terminados em `open ... in` ou atributos como
   `@[aesop safe apply]`. Agora as opções entram logo após os imports.
3. **Pacotes**: `autoImplicit false` só é forçado para arquivos `Mathlib/`;
   arquivos de `Batteries` (3 casos) dependem do `autoImplicit` padrão.
4. **`@[to_additive]`**: atributo no fim do cabeçalho conflitava com o renomeio;
   a linha é removida do contexto de teste.
5. **Falso positivo do modo portfólio**: declarações irmãs com o mesmo enunciado
   e `@[reassoc (attr := simp)]` registravam lemas de `simp` que fechavam
   tentativas posteriores. O caso `CategoryTheory.ShortComplex.SnakeInput.w₁₃_τ₁`
   era marcado como fechado por `aesop`, mas falha isoladamente. O modo portfólio
   foi substituído pelo modo `single` (uma tática por chamada).

## Números corrigidos (substituem as tabelas correspondentes do LTP-03)

Táticas simbólicas em modo `single`, 8 metas por split, ambiente nativo:

| Split | metas | simp | aesop | omega | linarith | nlinarith | portfólio pass@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | **0/8** |
| novel_premises | 8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | **0/8** |

Rechecagem do ReProver tacgen (táticas geradas reavaliadas contra as declarações
corrigidas; 4 casos term-mode excluídos): `random` 1/14 fechou e 3/14 válidas
incompletas; `novel_premises` 0/14 fechou e 6/14 válidas incompletas; 3 casos
mudaram de veredito em relação ao LTP-03. Geração e demais métricas do LTP-03
permanecem válidas (não dependem da extração). O BM25 não foi afetado.

## Gate

| Critério | Resultado |
| --- | --- |
| Hash textual e semântico implementados | sim |
| Hash semântico determinístico em ambiente limpo | 32/32 |
| 100% das mutações rejeitadas (LH-070) | 6/6 |
| Provas aceitas recompilam em replay limpo | 5/5 |
| Replay determinístico (LH-069) | idêntico em 8/8 |
| Nenhum conjunto selado aberto | sim |

## Limitações

- O hash semântico aceita ruído de declarações anteriores do cabeçalho desde que
  o tipo do alvo seja impresso; 1 caso (`min_mul_distrib'`) tem erros de
  cabeçalho e hash estável.
- O replay de provas originais do mathlib serve de controle; a classe de provas
  de candidato tem 1 caso (o único fechado pelo ReProver corrigido).
- Subconjuntos pequenos (8 a 32 metas por split); nada foi aberto do test.
- O hash semântico cobre a forma impressa com `pp.all`; não é uma comparação de
  termos de prova nem substitui o hash textual.

## Decisão

Gate aprovado. LTP-04 encerrada. A próxima fase (LTP-05, pilotos LH-093, LH-024
e LH-001) **não** foi iniciada.

## Prompt para iniciar LTP-05 (não executar agora)

```text
Trabalhe no projeto /home/iiii/PESSOAL-PROJETOS-ALEXANDRE/quantun-ia como
executor do programa Lean Tiny Prover. Antes de agir, leia README.md,
docs/LEAN_RESEARCH_PROGRAM.md, research/lean/hypotheses.json,
research/lean/datasets.json, research/lean/baselines.json,
research/lean/acquisitions/*.json, research/lean/ltp/LTP-00.md,
research/lean/ltp/LTP-01.md, research/lean/ltp/LTP-02.md,
research/lean/ltp/LTP-03.md, research/lean/ltp/LTP-04.md,
research/lean/ltp/LTP-03-protocol.json e .local/PROMPT-CONTINUAR.md. Rode git
status --short, make lean-plan-check e make check. Verifique Lean/Lake/elan,
GPU/VRAM, RAM e disco. Execute SOMENTE LTP-05: implementar os pilotos LH-093
(roteador simbólico-neural), LH-024 (filtro por tipo antes da busca) e LH-001
(bytes canônicos) como microimplementações T0/T1 com condição de falsificação
explícita, usando o LeanDojo Benchmark 4 val e o ambiente nativo; não abra
conjuntos de teste selados, não treine modelos e não use serviços pagos.
Registre objetivo, entregas, orçamento, gate, comandos, hashes, métricas,
limitações e decisão em research/lean/ltp/LTP-05.md; promova artefatos apenas se
os gates das três hipóteses forem avaliados e pare antes de LTP-06.
```
