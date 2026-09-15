# LTP-00 — Congelar o laboratório formal

- **Data**: 2026-09-15
- **Fase**: LTP-00 (primeira fase aberta; nenhuma fase seguinte iniciada)
- **Orçamento**: T0 (checagem em CPU, sem treino, sem download de pesos)
- **Gate**: 100% das provas válidas aceitas em ambiente limpo e 100% das provas
  deliberadamente inválidas rejeitadas.
- **Decisão**: **gate passou** na run canônica `20260915T182954Z`.

## Objetivo

Instalar e pinar Lean 4/mathlib, criar um projeto Lake separado, construir o
verificador adversarial (executor sandboxado, hash de afirmação, scanner de
proibições e auditoria formal de axiomas) e congelar um smoke de 32 teoremas sem
sobreposição com miniF2F, ProofNet, miniCTX, PutnamBench ou LeanDojo.

## Entregas

| Entrega | Artefato |
| --- | --- |
| Lean 4 via elan, pinado | `lean/lean-toolchain` = `leanprover/lean4:v4.34.0` |
| Projeto Lake separado | `lean/lakefile.toml`, `lean/lake-manifest.json` |
| mathlib pinado por commit | `v4.34.0` = `5ed2965256430c3649e86755f9576b54eca72435` |
| Executor sandboxado | `scripts/lean_verify.py` (`run_lean`) |
| Hash canônico da afirmação | `normalize_statement`, `statement_sha256` |
| Scanner de proibições | `scan_forbidden` (ignora comentários e strings) |
| Auditoria formal de axiomas | `parse_axioms` sobre `#print axioms` |
| Smoke de 32 teoremas | `research/lean/smoke/smoke32.json` |
| Suíte adversarial | `research/lean/smoke/adversarial.json` |
| Registro de hardware e pico | `research/lean/runs/LTP-00/environment.json` |
| Run canônica promovida | `research/lean/runs/LTP-00/` |
| Histórico de falhas | `research/lean/runs/LTP-00/history/` |

## Ambiente congelado

- Lean: `4.34.0`, commit `293d5d0c0c3f3dded4688b3ccd6a33939ac5102b`.
- Lake: `5.0.0-src+293d5d0`; elan: `4.2.4 (227caca13 2026-08-25)`.
- elan installer (release v4.2.4) sha256:
  `42b94d4244e8353142c456ec0e4ca6528fd898a6c604d4059f494e706e431f63`.
- mathlib: `rev` e `inputRev` iguais ao commit acima no `lake-manifest.json`.
- `lean-toolchain` sha256 `8733782dc070a99b312039cda424f601b80f3be6f6f512627da5ba25adc27632`.
- `lakefile.toml` sha256 `0a3b34abba71e3b70083a30e3d0dc3dc77ef344d91324ac722a541b3ef017026`.
- `lake-manifest.json` sha256 `e3456b9f9435c6a8a30ac91e749edbdb17137f8d916d15b8ecaa8406e794a7ed`.
- Hardware: i7-13620H (16 threads), 33.337.929.728 bytes de RAM
  (~31 GiB), ~106 GiB livres em disco, RTX 4060 Laptop 8 GiB, driver
  580.173.02, CUDA 13.0.
- Otimização: cache oficial do mathlib (`lake exe cache get`) + build local do
  módulo raiz (2 arquivos ausentes no cache). Nenhum peso de modelo foi baixado.

## Protocolo de verificação

Política congelada em `smoke32.json` antes de observar resultados:

- imports fixos: `Mathlib.Tactic`;
- `set_option autoImplicit false` e `maxHeartbeats 200000` no template;
- timeout de parede padrão de 120 s por caso;
- axiomas permitidos: `propext`, `Classical.choice`, `Quot.sound`;
- tokens proibidos no candidato: `sorry`, `admit`, `sorryAx`, `axiom`,
  `constant`, `unsafe`, `run_tac`, `run_cmd`, `set_option`, `partial`,
  `#eval`, `#exec`;
- sandbox: processo novo por caso, cwd temporário próprio, namespace de rede
  (`unshare -rn`), `RLIMIT_CPU`, `RLIMIT_AS` e kill por RSS acima de 8 GiB;
- aceitação exige exit code 0, `#print axioms` presente e subconjunto permitido;
  `native_decide` não é bloqueado no texto, mas cai na auditoria de axiomas.

Fluxo do gate: passe primário e replay integral em diretório e processos novos,
comparando veredito e razão caso a caso.

## Suíte e resultados

107 casos: 32 válidos selados + 4 controles positivos de scanner (3 com
comentários, 1 com string contendo tokens proibidos) + 32 `sorry` + 32 tática
errada + 6 casos especiais de rejeição (axioma em preâmbulo, `sorryAx`,
`admit`, `native_decide`, redefinição do alvo, prova vazia) + 1 adulteração de
afirmação.

Run canônica `20260915T182954Z`:

- 36/36 casos `accept` aceitos; 71/71 casos `reject` rejeitados; razões
  esperadas conferidas (criteria todas verdadeiras no `decision.json`);
- razões observadas: `forbidden_token` 35, `lean_error` 34, `extra_axiom` 1,
  `statement_hash_mismatch` 1;
- replay limpo idêntico em todos os 107 casos;
- 71 chamadas ao Lean por passe; parede total 259,3 s (replay 259,7 s);
- latência por chamada: p50 3,60 s, p95 4,85 s; pico de RSS 3,45 GB (~3,2 GiB);
- suite smoke32 sha256 `1ea007afb5d1b1ebde6bf7043242758d826b69719b818ddfc3c9f1781d0cb276`;
- suíte adversarial sha256 `eccaa0044cdea93a66315c74a6f21f9419928335ed2ea13f23b496de8aa1f46d`;
- git HEAD `9918975`; 619 alterações locais pré-existentes preservadas.

## Histórico de falhas (não apagado)

1. `20260915T175819Z` — gate falhou: o fixture `ADV-TIMEOUT` usava função
   identicamente zero, então o `decide` errava a afirmação em vez de consumir
   wall-clock. O critério do gate não foi alterado; o fixture foi substituído
   por teste sintético de timeout no harness (`test_run_lean_enforces_wall_timeout`).
2. `20260915T180959Z` — gate passou, mas `run_pass` apagava `run.json`,
   `environment.json` e `statements.sha256` do próprio `run_dir`; corrigido com
   remoção seletiva e teste de regressão `test_run_pass_preserves_run_metadata`.
3. `20260915T181929Z` — gate passou, mas `environment.json` registrava erro de
   toolchain ao capturar `lean --version` fora do projeto Lake; corrigido com
   `cwd` do projeto.

Detalhes em `research/lean/runs/LTP-00/history/`.

## Comandos

```bash
make lean-plan-check                 # valida registros do programa
make check                           # valida suítes + testes unitários do harness
.venv/bin/python scripts/lean_verify.py seal-suite
.venv/bin/python scripts/lean_verify.py validate-suite
.venv/bin/python scripts/lean_verify.py environment --out env.json
.venv/bin/python scripts/lean_verify.py build-case --case-id SMOKE-001-VALID --out c.lean
.venv/bin/python scripts/lean_verify.py scan --file c.lean
make lean-gate                       # gate completo com replay limpo
```

## Limitações

- O hash da afirmação é textual canônico (NFC, comentários removidos, espaços
  colapsados); a forma semântica fica para LH-070.
- O scanner é rejeição antecipada; a garantia de ausência de atalhos vem do
  exit code e do relatório `#print axioms`.
- O sandbox é namespace de rede + rlimits + cwd temporário; não é uma sandbox
  de sistema completa (sem seccomp).
- Não há medição de energia nesta fase.
- O timeout de parede é backstop de processo; com `maxHeartbeats 200000`,
  elaborações descontroladas viram `lean_error` antes de consumir minutos.
- Os 32 teoremas são internos e não substituem miniF2F ou outros benchmarks.
- Nenhum peso de modelo foi baixado, então não há checksums de pesos; o campo
  fica explicitamente vazio nesta fase.

## Decisão

Gate aprovado. LTP-00 encerrada. A próxima fase (LTP-01, aquisição do LeanDojo
v5) **não** foi iniciada.

## Prompt para iniciar LTP-01 (não executar agora)

```text
Trabalhe no projeto /home/iiii/PESSOAL-PROJETOS-ALEXANDRE/quantun-ia como
executor do programa Lean Tiny Prover. Antes de agir, leia README.md,
docs/LEAN_RESEARCH_PROGRAM.md, research/lean/hypotheses.json,
research/lean/datasets.json, research/lean/baselines.json,
research/lean/ltp/LTP-00.md e .local/PROMPT-CONTINUAR.md. Rode git status
--short, make lean-plan-check e make check. Verifique Lean/Lake/elan, GPU/VRAM,
RAM e disco. Execute SOMENTE LTP-01: adquirir o LeanDojo Benchmark v5 (47,3 MB públicos)
com URL, revisão/commit, licença auditada, checksum verificado, tamanho, papel
do split e relatório de contaminação, conforme os requisitos de aquisição de
research/lean/datasets.json, sem abrir conjuntos de teste selados e sem baixar
pesos de modelo. Confirme o checksum publicado antes de aceitar o arquivo.
Registre versões, comandos, hardware, hashes, métricas, limitações e decisão em
research/lean/ltp/LTP-01.md, promova o manifesto apenas se os requisitos de
aquisição forem cumpridos e pare antes de LTP-02.
```
