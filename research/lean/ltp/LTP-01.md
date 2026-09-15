# LTP-01 — Adquirir LeanDojo Benchmark v5

- **Data**: 2026-09-15
- **Fase**: LTP-01 (única fase aberta; nenhuma fase seguinte iniciada)
- **Dependência**: LTP-00 concluída (`research/lean/ltp/LTP-00.md`, gate aprovado)
- **Orçamento**: T0; download de 47,3 MB, extração de 610 MB, sem treino e sem
  pesos de modelo
- **Gate**: checksum publicado confere; revisão/DOI pinados; licença auditada e
  registrada; papéis dos splits registrados; relatório de contaminação v1
  produzido; nenhum conjunto selado aberto; nenhum peso baixado.
- **Decisão**: **gate passou**.

## Objetivo

Adquirir o LeanDojo Benchmark v5 (treino e desenvolvimento) a partir do registro
Zenodo 10114157, com proveniência, licença e integridade verificadas, sem abrir
conjuntos de teste selados e sem baixar pesos neurais.

## Entregas

| Entrega | Artefato |
| --- | --- |
| Manifesto de aquisição | `research/lean/acquisitions/leandojo_benchmark_v5.json` |
| Snapshot público do Zenodo | `research/lean/acquisitions/zenodo_10114157.metadata.json` |
| Gerador reprodutível | `scripts/build_lean_acquisition_manifest.py` |
| Validação estrutural | `scripts/validate_lean_research.py` + `tests/test_lean_research.py` |
| Dados locais (não versionados) | `data/raw/leandojo/` |

## Proveniência e licença

- Registro: https://zenodo.org/records/10114157
- DOI da versão: `10.5281/zenodo.10114157`; DOI do conceito:
  `10.5281/zenodo.8016385`; revisão 4, criado em 2023-11-11.
- Licença do dataset: CC BY 2.0; licenças transitivas de mathlib e Lean vêm
  embutidas em `licenses/`.
- `metadata.json` interno: LeanDojo `1.4.0`, extraído de
  `leanprover-community/mathlib` commit
  `19c869efa56bbb8b500f2724c0b77261edbfa28c` (2023-11-11).
- Snapshot da API do Zenodo com sha256
  `3f059855e4183bcb08675d24b1c1f968948c7df3c47992470991dc50728187f`.

## Integridade

| Item | Valor |
| --- | --- |
| Arquivo | `leandojo_benchmark_v5.tar.gz` |
| Tamanho | 47.260.016 bytes (confere com o publicado) |
| md5 publicado | `4b256200618d4668b12a9cfe8c4df4d3` |
| md5 observado | `4b256200618d4668b12a9cfe8c4df4d3` |
| sha256 observado | `4c3df80440630c5c9785b511491c8ee09c4d2da31de085b205b3768666e510a0` |
| Checksum verificado | sim (`archive.checksum_verified = true`) |

Comandos executados:

```bash
curl -sSfL -o data/raw/leandojo/leandojo_benchmark_v5.tar.gz \
  "https://zenodo.org/api/records/10114157/files/leandojo_benchmark_v5.tar.gz/content"
md5sum data/raw/leandojo/leandojo_benchmark_v5.tar.gz
tar -xzf data/raw/leandojo/leandojo_benchmark_v5.tar.gz -C data/raw/leandojo/extracted
.venv/bin/python scripts/build_lean_acquisition_manifest.py \
  --id leandojo_benchmark_v5 \
  --root data/raw/leandojo/extracted/leandojo_benchmark \
  --archive data/raw/leandojo/leandojo_benchmark_v5.tar.gz \
  --zenodo-metadata <snapshot bruto da API> \
  --metadata-snapshot-out research/lean/acquisitions/zenodo_10114157.metadata.json \
  --out research/lean/acquisitions/leandojo_benchmark_v5.json
```

## Inventário por split

Dois splits de avaliação, `random` e `novel_premises`, cada um com 94.734
teoremas de treino e 2.000 de validação. Arquivos de teste foram apenas
hasheados; o conteúdo não foi inspecionado.

| Arquivo | Tamanho | Registros | sha256 |
| --- | --- | --- | --- |
| `corpus.jsonl` | 39.750.037 | — | `24c034bd1483f6edb05f429c3cc8fefdd064a1a0a4eb1ce350554e686c3540d8` |
| `random/train.json` | 285.461.169 | 94.734 | `8b20dee433e116231c99e96e79c50af729c3e868c59d206c9be4f386af1f6013` |
| `random/val.json` | 8.109.975 | 2.000 | `49fddb5476d1dd206bd6ee7caad1285a1937ea6192f3e87ea8fd962316074b49` |
| `random/test.json` | 5.894.234 | selado | `445eb61d25b98b34b029d720f8ed56851e93366d62ccae2f712441f63edec0ed` |
| `novel_premises/train.json` | 260.105.204 | 94.734 | `55fa25168fbfbdc8a5e0baa7819e574ea32880de1f5ab48300ce8299681767e8` |
| `novel_premises/val.json` | 20.038.599 | 2.000 | `524bac7fc7cb4547d20c426a8451a7424be101ace84ce093a429104c4fccde3b` |
| `novel_premises/test.json` | 19.321.575 | selado | `4d2b7fd134d00b9787d0a6c26d35321bc327daf0eafb806b0768c341a8e2681b` |

Os dados extraídos ocupam 610 MB em `data/raw/leandojo/extracted/`, coberto
pelo `.gitignore` (`data/raw/`). Somente o manifesto e o snapshot público são
versionados.

## Relatório de contaminação v1

- Nenhum conjunto selado foi tocado (`sealed_sets_touched: []`).
- Arquivos de teste do LeanDojo: hash e tamanho registrados,
  `content_inspected: false`, bloqueados para seleção de arquitetura,
  hiperparâmetros e hipóteses.
- miniF2F, ProofNet Lean 4, miniCTX v2 e PutnamBench: nenhum arquivo local
  encontrado; status de registro `planned` ou `planned_blocked_on_license`;
  auditoria de sobreposição marcada `pending_until_acquisition`.
- Deduplicação por afirmação exige o índice canônico de LH-070 e será executada
  em LTP-03/LTP-04 quando os demais benchmarks forem adquiridos.

## Risco de compatibilidade registrado

O dataset foi extraído de mathlib de 2023-11-11 (`19c869ef…`), enquanto o
ambiente de verificação pinado em LTP-00 usa mathlib `v4.34.0`
(`5ed2965…`, 2026-09-15). Afirmações do LeanDojo v5 podem não compilar
literalmente contra o mathlib atual. A decisão de alinhamento de versões para
comparação justa pertence a LTP-03 e deve ser pré-registrada antes de qualquer
medição: ou o baseline ReProver roda em ambiente próprio pinado (e o
`fairness_lock` exige replicar o ambiente para o candidato), ou o conjunto é
revalidado contra o snapshot moderno. Nenhuma das duas opções foi executada
nesta fase.

## Gate

| Critério | Resultado |
| --- | --- |
| md5 confere com o publicado | sim |
| Revisão/DOI pinados | sim (revisão 4, DOIs registrados) |
| Licença auditada e registrada | sim (CC BY 2.0 + licenças embutidas) |
| Papel dos splits registrado | sim (treino/desenvolvimento; test selado) |
| Relatório de contaminação v1 | sim, com pendências explicitadas |
| Nenhum conjunto selado aberto | sim |
| Nenhum peso de modelo baixado | sim |

## Limitações

- A auditoria de sobreposição entre benchmarks ainda não foi executada; depende
  das aquisições seguintes.
- O conteúdo dos arquivos de teste do LeanDojo não foi inspecionado por decisão
  de protocolo; contagens de registros de teste ficam ausentes de propósito.
- O dataset corresponde a um snapshot de mathlib de 2023; a compatibilidade com
  o ambiente pinado de LTP-00 não foi testada nesta fase.
- A extração local não é versionada; a reprodutibilidade depende do checksum e
  do manifesto registrados.

## Decisão

Gate aprovado. LTP-01 encerrada. A próxima fase (LTP-02, aquisição do ReProver
300M) **não** foi iniciada.

## Prompt para iniciar LTP-02 (não executar agora)

```text
Trabalhe no projeto /home/iiii/PESSOAL-PROJETOS-ALEXANDRE/quantun-ia como
executor do programa Lean Tiny Prover. Antes de agir, leia README.md,
docs/LEAN_RESEARCH_PROGRAM.md, research/lean/hypotheses.json,
research/lean/datasets.json, research/lean/baselines.json,
research/lean/acquisitions/leandojo_benchmark_v5.json,
research/lean/ltp/LTP-00.md, research/lean/ltp/LTP-01.md e
.local/PROMPT-CONTINUAR.md. Rode git status --short, make lean-plan-check e
make check. Verifique Lean/Lake/elan, GPU/VRAM, RAM e disco. Execute SOMENTE
LTP-02: auditar a licença MIT e a revisão dos pesos
kaiyuy/leandojo-lean4-tacgen-byt5-small e
kaiyuy/leandojo-lean4-retriever-byt5-small no Hugging Face, registrar revisão
imutável, baixar os ~1,2 GB do gerador (e o retriever, se a licença permitir),
verificar checksums, contar parâmetros totais e ativos localmente e conferir se
o total permanece <=300M, sem treinar, sem abrir conjuntos de teste selados e
sem usar serviços pagos. Se a licença ou o tamanho não forem aceitáveis,
registre o bloqueio e pare. Registre versões, comandos, hardware, hashes,
métricas, limitações e decisão em research/lean/ltp/LTP-02.md, promova apenas o
manifesto de aquisição se o gate passar e pare antes de LTP-03.
```
