# quantun-ia

## Direção ativa: Lean Tiny Prover

Em 2026-09-15 o projeto abriu uma nova linha de matemática formal: testar 100
hipóteses falsificáveis para construir um provador Lean 4 especialista com até
100M parâmetros (nano) ou, no máximo, 300M (micro). O objetivo não é maximizar
pass@k a qualquer custo, mas encontrar a fronteira entre provas certificadas,
parâmetros e compute fim a fim.

Comece por:

1. [`docs/LEAN_RESEARCH_PROGRAM.md`](docs/LEAN_RESEARCH_PROGRAM.md);
2. [`research/lean/hypotheses.json`](research/lean/hypotheses.json);
3. [`research/lean/datasets.json`](research/lean/datasets.json);
4. [`research/lean/baselines.json`](research/lean/baselines.json);
5. `make lean-plan-check`.

O CI ativo está documentado em [`docs/CI.md`](docs/CI.md) e executa um único
gate formal, sem os pipelines antigos de QML, API, Docker, paper ou publicação.

O Lean certifica cada prova, não a alegação de superioridade do modelo. Essa
alegação continua sujeita a teste selado, comparação pareada, correção estatística
e orçamento idêntico. A linha QML abaixo permanece como histórico validado do
reset, não como agenda ativa.

Projeto de pesquisa em aprendizado de máquina quântico, reiniciado em 2026-09-15
para separar evidência reproduzível de exploração, demonstrações e resultados
negativos.

## Estado científico

Não há, neste repositório, prova de vantagem quântica. Quatro estudos permanecem
ativos como **evidência interna reproduzível**, todos com dados públicos e execução
multi-seed:

- `exp_011`: no Wisconsin Breast Cancer, a QNN perdeu para o baseline clássico
  pareado em 5,6 pontos percentuais;
- `exp_012`: em MNIST 0 vs 1 reduzido por PCA, amplitude encoding superou angle
  encoding, ainda sem comparação clássica forte;
- `exp_024`: no Wisconsin Breast Cancer, o híbrido ficou 0,5 ponto percentual
  abaixo da regressão logística;
- `exp_025`: em Pima Diabetes, o híbrido ficou 1,0 ponto percentual abaixo da
  regressão logística.

Esses resultados são limitados a um único protocolo de holdout. `exp_024` e
`exp_025` também não possuem o pré-registro externo prometido. Portanto, eles não
devem ser descritos como confirmação externa, validação clínica ou vantagem
quântica.

## Onde começar

1. Leia [`docs/RESEARCH_RESET.md`](docs/RESEARCH_RESET.md).
2. Leia [`docs/PROTOCOL.md`](docs/PROTOCOL.md).
3. Consulte [`research/evidence_registry.json`](research/evidence_registry.json).
4. Execute `make check`.

O navegador contém apenas um monitor de evidências em modo leitura:

```bash
streamlit run dashboard/app.py
```

Os experimentos e pesos retirados do fluxo ativo estão preservados localmente em
`.local/archive/2026-09-15-research-reset/`. O snapshot Git anterior à limpeza é
`9918975`, permitindo recuperação mesmo sem o arquivo local.

## Regra de alegação

Resultado positivo, paridade interna e passagem de teste de software não são
sinônimos de comprovação científica. Uma nova alegação exige pré-registro anterior
ao primeiro resultado, dados reais rastreáveis, baseline clássico forte no mesmo
split, avaliação confirmatória selada, correção estatística, análise de potência e
replicação em pelo menos um segundo dataset.
