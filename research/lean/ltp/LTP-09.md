# LTP-09 — Abertura única do teste selado

- **Data**: 2026-09-16
- **Fase**: LTP-09 (única fase aberta; LTP-10 não iniciada)
- **Dependência**: LTP-00 a LTP-08 concluídas
- **Orçamento**: T1 em CPU; 815 s de parede, pico de RSS 1,83 GB; sem treino,
  sem GPU e sem serviços pagos
- **Gate**: abrir o teste uma única vez, reportar pass@k, chamadas, tempo, pico
  e resolvedores exclusivos por finalista, sem reabrir depois do resultado
- **Decisão**: **gate aprovado**; finalistas avaliados no teste selado;
  `novel_premises/test.json` permaneceu intocado

## Abertura (uma única vez)

- Arquivo: `random/test.json` do LeanDojo Benchmark 4.
- sha256 **antes** da leitura:
  `9f84ac4cbd5ea9031ab335d172e7a5b1873e4ac773721e60cb1a4cc5b4e2a0ae`, idêntico
  ao registrado no manifesto de aquisição (LTP-01) — `hash_matches_manifest:
  true`.
- Subconjunto congelado: primeiros 16 teoremas com traços e extração by-style
  (17 tentativas), hashes das afirmações registrados; cópia selada em
  `.local/sealed/LTP-09/test_subset.json` (sha256
  `2ccaf1aa19bc9f041f0d98d305c8dfd5334da3a7d6c7a7e66592fee60266b12d`).
- Após a extração, `random/test.json` **não foi mais lido**; toda a avaliação
  usou a cópia selada. `novel_premises/test.json` não foi consultado.
- Artefatos versionados contêm somente agregados, hashes e nomes dos teoremas
  resolvidos exclusivamente (exigidos pelo protocolo); nenhuma prova ou
  afirmação do test foi publicada.

## Finalistas no teste selado

| Finalista | Métrica | Resultado |
| --- | --- | --- |
| LH-070 hash imutável | afirmações com hash semântico determinístico | **16/16** (após correção do harness) |
| LH-069 replay limpo | provas idênticas em dois passes limpos | **16/16** (14 aceitas) |
| LH-026 cache de objetivos | taxa de acerto (5 seeds, canônico) | **12,5%** (gate ≥ 5%) |

Observações:

- LH-070: a primeira medição deu 15/16 porque o renomeador do harness não
  tratava nomes Lean com `?` (caso `Heap.WF.tail?`); o bug foi corrigido
  (`rename_declaration`) e a medição refeita **na cópia selada**, sem reabrir o
  test: 16/16 hashable e determinístico. O resultado inicial fica registrado.
- LH-070: 3 mutações textuais aplicáveis foram geradas e 3 detectadas; a camada
  semântica de mutações foi confirmada no val (LTP-04).
- LH-069: 2 das 16 provas originais não foram aceitas (rejeitadas de forma
  idêntica nos dois passes); o replay é determinístico em 16/16 casos.

## Linhas de referência (mesmo protocolo)

| Linha | pass@k | Chamadas | Tempo | Exclusivo do teste |
| --- | --- | --- | --- | --- |
| Táticas simbólicas (portfólio, k=5) | **1/16** | 77 | 352,9 s | `Finmap.erase_toFinmap` |
| ReProver tacgen sem recuperação (k=1) | **1/16** | 16 | 75,1 s | `CategoryTheory.Equivalence.invFunIdAssoc_hom_app` |

- Nenhum teorema foi resolvido pelas duas linhas (`solved_by_both: []`).
- Nenhum provador neural treinado por este programa existe ainda: os finalistas
  são mecanismos de validade/cobertura, e as linhas acima dão contexto de
  pass@k, não uma alegação de vitória.
- Pico de RSS da execução completa: 1.827.504.128 bytes; parede total 814,9 s.

## Comandos

```bash
.venv/bin/python scripts/ltp09_sealed_test.py --out research/lean/runs/LTP-09/sealed_test.json
.venv/bin/python scripts/ltp09_exclusive.py --out research/lean/runs/LTP-09/exclusive.json
```

## Gate da fase

| Critério | Resultado |
| --- | --- |
| Teste aberto uma única vez | sim (hash conferido antes e depois) |
| Mesmas afirmações hasheadas e mesmo protocolo | sim |
| pass@k, chamadas, tempo e pico reportados | sim |
| Resolvedores exclusivos por linha | sim (1 e 1, sem interseção) |
| Teste não reaberto após o resultado | sim |
| `novel_premises/test.json` selado | sim |
| Sem treino/GPU/serviço pago | sim |

## Limitações

- Subconjunto de 16 teoremas; resultados não se estendem ao test completo.
- Correção de harness pós-abertura documentada; a correção não alterou dados do
  test, apenas o renomeio de uma declaração.
- Os finalistas medem determinismo, hashing e cache — não pass@k de um provador
  neural; a comparação pareada com ReProver segue pendente de treino (T2).
- O split `novel_premises/test` permanece selado para LTP-10.

## Decisão

Gate aprovado. LTP-09 encerrada. A próxima fase (LTP-10, generalização em
ProofNet Lean 4 e miniCTX; PutnamBench como stress) **não** foi iniciada.

## Prompt para iniciar LTP-10 (não executar agora)

```text
Trabalhe no projeto /home/iiii/PESSOAL-PROJETOS-ALEXANDRE/quantun-ia como
executor do programa Lean Tiny Prover. Antes de agir, leia README.md,
docs/LEAN_RESEARCH_PROGRAM.md, research/lean/hypotheses.json,
research/lean/datasets.json, research/lean/baselines.json,
research/lean/acquisitions/*.json, research/lean/ltp/LTP-00.md a LTP-09.md e
.local/PROMPT-CONTINUAR.md. Rode git status --short, make lean-plan-check e
make check. Verifique Lean/Lake/elan, GPU/VRAM, RAM e disco. Execute SOMENTE
LTP-10: avaliar generalização em ProofNet Lean 4 e miniCTX v2 após auditoria de
licença e aquisição pinada, mantendo o protocolo justo (mesmas afirmações
hasheadas, timeout, pass@k e chamadas) e usando PutnamBench apenas como stress
test; abra no máximo uma vez cada conjunto selado necessário e não use serviços
pagos nem treine acima do orçamento T1. Registre objetivo, entregas, orçamento,
gate, comandos, hashes, métricas, limitações e decisão em
research/lean/ltp/LTP-10.md e encerre o funil com o resultado final do
programa.
```
