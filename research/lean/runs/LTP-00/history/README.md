# Histórico de execuções do gate LTP-00

A run canônica promovida é `.local/runs/lean/LTP-00/20260915T182954Z`
(copiada para esta pasta sem o diretório `scratch/`). As execuções abaixo
ficam registradas, nunca apagadas.

| run | resultado | motivo |
| --- | --- | --- |
| `20260915T175819Z` | gate falhou | fixture `ADV-TIMEOUT` estava quebrado: a função definida era identicamente zero, então o `decide` errava a afirmação em vez de consumir o wall-clock. Critério do gate inalterado; fixture substituído por teste sintético de timeout. |
| `20260915T180959Z` | gate passou, artefatos incompletos | `run_pass` apagava o `run_dir` inteiro antes do passe primário, removendo `run.json`, `environment.json`, `statements.sha256` e `cases.json`. Corrigido com remoção seletiva de `scratch/candidates/proofs/transcripts` e teste de regressão. |
| `20260915T181929Z` | gate passou, metadados incompletos | `environment.json` capturava versões de `lean`/`lake` fora do projeto Lake, registrando erro de toolchain. Corrigido com `cwd` do projeto e run canônica repetida. |
| `20260915T182954Z` | gate passou | run canônica promovida; 36/36 aceitas, 71/71 rejeitadas, replay limpo idêntico. |

Os três primeiros diretórios guardam `decision.json` e `metrics.json` (e
`environment.json`, quando existia) para auditoria.
