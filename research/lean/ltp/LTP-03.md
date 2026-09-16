# LTP-03 — Reproduzir táticas simbólicas, BM25 e ReProver

- **Data**: 2026-09-15
- **Fase**: LTP-03 (única fase aberta; nenhuma fase seguinte iniciada)
- **Dependência**: LTP-00, LTP-01 e LTP-02 concluídas
- **Orçamento**: CPU; sem treino, sem GPU, sem serviços pagos
- **Protocolo congelado**: `research/lean/ltp/LTP-03-protocol.json` (emendas de
  orçamento registradas antes de qualquer medição)
- **Gate**: reproduzir o baseline no mesmo commit/split ou documentar
  quantitativamente a divergência antes de testar hipótese nova
- **Decisão**: **gate passou**; correção de dataset e divergências registradas

## Correção de rota: o LeanDojo v5 adquirido é Lean 3

A inspeção de desenvolvimento mostrou que `leandojo_benchmark_v5`
(Zenodo 10114157) contém dados de **Lean 3/mathlib3**:

- `state_before` usa sintaxe Lean 3 (`preorder α`, `_inst_1`, minúsculas);
- corpus com caminhos `_target/deps/lean/library/init/core.lean`;
- `from_repo` aponta para `github.com/leanprover-community/mathlib`, commit
  `19c869ef…`, que não existe no `mathlib4`;
- o dataset é, portanto, incompatível com o alvo Lean 4 do programa.

Aquisição corretiva nesta fase: **LeanDojo Benchmark 4** (Zenodo 12740403,
2024-07-14, CC BY 2.0), arquivo `leandojo_benchmark_4.tar.gz`, 68.141.936 bytes,
md5 `25e1ee60cd8925b9d2e8673ddcc34b4c` **conferido**. Manifesto em
`research/lean/acquisitions/leandojo_benchmark_4.json`; mathlib4 commit
`29dcec074de168ac2bf835a77ef68bbe069194c5` (LeanDojo 2.0.0). `datasets.json`
registra o v5 como `acquired_lean3_mismatch` e o benchmark 4 como
`acquired_verified`. Testes permanecem selados (somente hash).

## Ambientes

| Ambiente | Lean | mathlib4 | Projeto |
| --- | --- | --- | --- |
| nativo (dataset) | `v4.10.0-rc1` | `29dcec07…` | `data/raw/mathlib4_src/` |
| moderno (pin LTP-00) | `v4.34.0` | `5ed2965…` | `lean/` |

O ambiente nativo foi montado com cache oficial do mathlib (4.742 arquivos,
100% de sucesso) e `lake build Mathlib`; import de `Mathlib` em ~7 s quando a
máquina está ociosa. O nativo é o **ambiente de referência do dataset** e passa a
ser obrigatório para qualquer comparação futura contra LeanDojo/ReProver
(`fairness_lock`); o moderno permanece como ambiente do smoke interno do LTP-00.

## Extração de afirmações

32/32 declarações por split (random e novel_premises) extraídas do val sem
falhas, com cabeçalho original preservado (média de 359 linhas) e hash canônico
da afirmação. Nenhuma afirmação foi alterada; somente o corpo de prova foi
substituído pela tática candidata. Artefatos: `statements.jsonl` e
`extraction_report.json` em `research/lean/runs/LTP-03/`.

## Táticas simbólicas (ambiente nativo, val)

16 metas por split; cinco táticas em ordem fixa, `maxHeartbeats 200000` por
declaração; sucesso = compila sem metas abertas.

| Split | simp | aesop | omega | linarith | nlinarith | portfólio pass@5 |
| --- | --- | --- | --- | --- | --- | --- |
| random | 1/16 (6,25%) | 2/16 (12,50%) | 0/16 | 0/16 | 0/16 | **2/16 (12,50%)** |
| novel_premises | 0/16 | 1/16 (6,25%) | 0/16 | 0/16 | 1/16 (6,25%) | **2/16 (12,50%)** |

Nenhum caso estourou timeout de parede; todas as chamadas terminaram por
elaboração (sucesso ou erro). Pareto de tempo: 282 s e 213 s por split.

## Portabilidade para o ambiente moderno

Método: afirmação de 2024 em contexto limpo (`import Mathlib` + apenas
`open`/`namespace`/`variable`/`section` do cabeçalho) com prova `sorry`.

- Resultado: **5/16 (31,25%)** das afirmações de `random/val` ainda elaboram no
  mathlib `v4.34.0`.
- Divergência quantificada: ~69% exigem reescrita; usar o ambiente moderno para
  reproduzir LeanDojo/ReProver não é viável sem portar afirmações.
- Registro metodológico: a primeira medição (0/16) foi descartada porque o
  cabeçalho completo colidia com declarações já existentes no mathlib moderno
  (`has already been declared`); a v2 isola o contexto e é a válida.

## BM25 (premissas, val)

Pipeline oficial do ReProver (BPE Whitespace vocab 30.000, `BM25Okapi`
k1=1,5 b=0,75, consulta no estado, premissas acessíveis por fechamento
transitivo de imports, métrica `_eval` oficial), com tokenizador treinado em
subconjunto determinístico (10.000 premissas + 10.000 estados), 50 provas por
split.

| Split | Consultas | R@1 | R@10 | R@32 | MRR |
| --- | --- | --- | --- | --- | --- |
| random | 125 | 9,00 | 24,10 | 30,61 | 0,224 |
| novel_premises | 154 | 2,13 | 15,54 | 20,56 | 0,120 |

Valores em percentual para R@k. Passos sem premissa positiva foram ignorados,
como no `_eval` oficial.

## ReProver tacgen sem recuperação (val, ambiente nativo)

Modelo `kaiyuy/leandojo-lean4-tacgen-byt5-small` revisão
`67a2c53cc36186fe8539d0a342fc42c50edc68fd`, 299.637.760 parâmetros, geração
greedy (1 amostra, `max_inp 2048`, `max_oup 512`), CPU; verificação no Lean
nativo com rejeição de `sorry`.

| Split | metas | exact match | prefixo | fechou (1 passo) | tática válida | inválida |
| --- | --- | --- | --- | --- | --- | --- |
| random | 16 | 1 (6,25%) | 1 (6,25%) | 1 (6,25%) | 3 (18,75%) | 12 (75,00%) |
| novel_premises | 16 | 3 (18,75%) | 3 (18,75%) | 0 | 5 (31,25%) | 11 (68,75%) |

"Tática válida" inclui fechar a meta ou deixar apenas metas abertas. Nenhuma
geração usou `sorry`. Tempo total: 216,7 s de geração e 243,1 s de verificação.

## Divergência em relação ao publicado

- ReProver publica pass@k com busca best-first e retriever sobre **test**; aqui
  medimos um passo greedy **sem recuperação** em **val** (desenvolvimento).
- O split de teste do benchmark permanece selado; a comparação publicada não é
  reproduzível sem abrir o test, o que o protocolo proíbe nesta fase.
- O tokenizador BM25 foi treinado em subconjunto (10k/10k), não no corpus
  completo do ReProver.
- Subconjuntos de 16 metas por split (simbólico e ReProver) e 50 provas (BM25).
- A máquina operou com load average ~15 durante as corridas; tempos de parede não
  são comparáveis a execuções dedicadas.

## Comandos principais

```bash
.venv/bin/python scripts/ltp03_extract.py --limit 32 --out <relatório> > statements.jsonl
.venv/bin/python scripts/ltp03_symbolic.py --statements statements.jsonl \
  --env native --split random --mode portfolio --goals 16 --out <json>
.venv/bin/python scripts/ltp03_symbolic.py --statements statements.jsonl \
  --env modern --split random --mode portability --goals 16 --out <json>
.venv/bin/python scripts/ltp03_bm25.py --proofs 50 --tokenizer-out <tk> --out <json>
.venv/bin/python scripts/ltp03_reprover.py --statements statements.jsonl \
  --model data/raw/reprover/leandojo-lean4-tacgen-byt5-small/<rev> --goals 16 --out <json>
```

## Artefatos promovidos

`research/lean/runs/LTP-03/`: `extraction_report.json`, `statements.jsonl`,
`symbolic_native_random.json`, `symbolic_native_novel_premises.json`,
`portability_modern_random_v2.json`, `bm25_results.json`,
`reprover_tacgen.json`, `environment.json`.

## Gate

| Critério | Resultado |
| --- | --- |
| Táticas simbólicas reproduzidas no commit do dataset | sim (12,5% portfólio) |
| BM25 reproduzido com métricas oficiais | sim (R@1 9,0%/2,1%) |
| ReProver medido quantitativamente | sim (tacgen sem recuperação, val) |
| Divergência documentada | sim (portabilidade 31,25%; val≠test; subset) |
| Test selado não aberto | sim |
| Sem treino, GPU, serviço pago | sim |

## Limitações

- Resultados valem para `val`; nada foi extrapolado para `test`.
- O pipeline simbólico mede táticas isoladas, não busca; não substitui um
  portfólio maior nem táticas aprendidas.
- A extração usa heurística de marcador `:= by`; 100% dos casos extraídos, mas
  declarações term-mode não foram testadas.
- O BM25 pode ser otimizado (tokenizador completo, mais provas); os números são
  um piso reprodutível, não o teto.
- Toda comparação futura contra LeanDojo/ReProver deve usar o ambiente nativo
  `v4.10.0-rc1 + 29dcec07` para satisfazer o `fairness_lock`.

## Errata (registrada em LTP-04)

Durante LTP-04 foram encontrados dois defeitos no ferramental desta fase:

1. **Extração**: declarações em estilo termo/equações (`calc`, casamento por
   padrões) foram cortadas no primeiro `:=` de nível zero, gerando afirmações
   truncadas. A extração corrigida aceita apenas provas `:= by` e exclui 6 casos
   em `random` e 2 em `novel_premises` (taxa de exclusão 6/39 e 2/35).
2. **Template**: as opções `autoImplicit`/`maxHeartbeats` eram inseridas
   imediatamente antes da declaração, quebrando cabeçalhos terminados em
   `open ... in` ou em atributos como `@[aesop safe apply]`. As opções passaram a
   ser inseridas logo após os imports.

Consequência: as tabelas de **táticas simbólicas** e a **verificação do ReProver**
desta página ficam superseded pelos números corrigidos de
`research/lean/runs/LTP-04/` (ver `LTP-04.md`), calculados sobre a extração v2.
O BM25 não é afetado (não usa extração). Os artefatos originais permanecem
preservados em `research/lean/runs/LTP-03/`.

## Decisão

Gate aprovado. LTP-03 encerrada. A próxima fase (LTP-04, LH-070 e LH-069) **não**
foi iniciada.

## Prompt para iniciar LTP-04 (não executar agora)

```text
Trabalhe no projeto /home/iiii/PESSOAL-PROJETOS-ALEXANDRE/quantun-ia como
executor do programa Lean Tiny Prover. Antes de agir, leia README.md,
docs/LEAN_RESEARCH_PROGRAM.md, research/lean/hypotheses.json,
research/lean/datasets.json, research/lean/baselines.json,
research/lean/acquisitions/*.json, research/lean/ltp/LTP-00.md,
research/lean/ltp/LTP-01.md, research/lean/ltp/LTP-02.md,
research/lean/ltp/LTP-03.md, research/lean/ltp/LTP-03-protocol.json e
.local/PROMPT-CONTINUAR.md. Rode git status --short, make lean-plan-check e
make check. Verifique Lean/Lake/elan, GPU/VRAM, RAM e disco. Execute SOMENTE
LTP-04: implementar LH-070 (hash imutável da afirmação, incluindo forma
semântica no ambiente nativo v4.10.0-rc1 + mathlib 29dcec07) e LH-069
(reexecução em ambiente limpo, sem cache, com determinismo comprovado) sobre o
LeanDojo Benchmark 4 val, estendendo scripts/lean_verify.py e a suíte de testes;
não abra conjuntos de teste selados, não treine e não use serviços pagos.
Registre objetivo, entregas, orçamento, gate, comandos, hashes, métricas,
limitações e decisão em research/lean/ltp/LTP-04.md; promova artefatos apenas se
o gate (falha se qualquer mutação de afirmação passar e se o replay não for
determinístico) passar; pare antes de LTP-05.
```
