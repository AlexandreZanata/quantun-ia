# CI da pesquisa Lean

O repositório possui um único workflow ativo:
`.github/workflows/ci.yml`.

## Check obrigatório

O GitHub exibirá um único check:

`Lean Research CI / Formal Research Gate`

Esse check:

1. valida os 100 cartões de hipótese;
2. valida manifestos de datasets, baselines e aquisições;
3. valida as suítes smoke/adversarial e executa seus testes unitários;
4. instala o toolchain Lean/mathlib pinado em `lean/`;
5. executa o gate adversarial completo com replay limpo.

O CI não instala PyTorch, PennyLane, Docker, LaTeX, API/JWT, MLflow ou o antigo
pipeline de publicação. Também não baixa o dataset LeanDojo nem os pesos
ReProver: integridade e proveniência dessas aquisições são verificadas por seus
manifestos versionados, enquanto os arquivos grandes permanecem ignorados pelo
Git.

## Execução local

Validação rápida:

```bash
make check
```

Gate formal completo:

```bash
make lean-gate
```

O segundo comando exige o projeto pinado em `lean/` e o cache mathlib.

## Branch protection

Depois que o workflow entrar na branch remota, remova os nomes dos checks
legados das regras de proteção e torne obrigatório apenas
`Lean Research CI / Formal Research Gate`.

