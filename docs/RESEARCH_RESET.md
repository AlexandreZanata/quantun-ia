# Reset científico — 2026-09-15

## Decisão

O projeto foi reduzido a quatro estudos com dados públicos reais e execução
multi-seed. Eles são mantidos como evidência interna reproduzível, não como prova
externa. Nenhum resultado atual demonstra vantagem quântica.

Foram retirados do fluxo ativo 109 diretórios experimentais. O conjunto inclui
resultados rejeitados, inconclusivos, estudos somente sintéticos, testes de API e
serving, ingestão de dados, sínteses que não reexecutam modelos e resultados com
gates internos permissivos. Preservar um resultado negativo é boa prática; por
isso nada foi apagado de forma irrecuperável.

## Critério de retenção

Um estudo legado permaneceu ativo somente quando satisfez todos estes mínimos:

- dataset público e não sintético;
- hipótese escrita e resultado versionado;
- mais de uma semente;
- comparação quantitativa identificável;
- teste estatístico e tamanho de efeito registrados;
- conclusão limitada ao protocolo realmente executado.

O critério ainda não equivale a confirmação científica externa. Para isso faltam,
conforme o caso, pré-registro válido, múltiplos splits ou nested CV, teste selado,
replicação em outro dataset, execução em hardware quântico e reprodução
independente.

## Evidência que permaneceu

- `exp_011`: evidência contra vantagem da QNN angle no Wisconsin Breast Cancer.
- `exp_012`: comparação de encodings em MNIST PCA, sem baseline clássico primário.
- `exp_024`: híbrido dentro da faixa de paridade, mas significativamente abaixo da
  regressão logística no protocolo executado.
- `exp_025`: repetição da ausência de vantagem no Pima Diabetes.

Os números e limitações canônicos estão em
`research/evidence_registry.json`. O navegador lê somente esse registro.

## O que foi arquivado

O arquivo local está em `.local/archive/2026-09-15-research-reset/` e contém:

- `legacy_experiments/`: 109 experimentos fora dos critérios;
- `browser_pages/`: páginas de treino, clínicas e “Lab” removidas da navegação;
- `legacy_weights/`: 10 pacotes de pesos que não sustentam alegação ativa;
- `legacy_model_cards/`, `legacy_docs/` e `legacy_outputs/`;
- `experiment_coupled_tests/`: testes presos ao catálogo experimental antigo.

Recuperação alternativa: `git show 9918975:<caminho>` ou um worktree temporário
nesse commit. O arquivo local não deve ser versionado.

## Pesos

O checkpoint `dist/serve_models/quantum_nano_bc/` foi mantido como baseline legado
do `exp_024`, nunca como modelo clínico ou prova de vantagem. Os demais pesos foram
arquivados localmente. Checksums SHA-256 ficam em
`research/WEIGHTS.sha256` para detectar alteração acidental.

## Próxima linha recomendada

> Nota de 2026-09-15: esta recomendação Q-Shift foi substituída, por decisão do
> pesquisador, pelo programa ativo `docs/LEAN_RESEARCH_PROGRAM.md`. O texto
> abaixo é preservado como histórico da decisão anterior, não como próximo passo.

Recomenda-se abandonar a sequência de “receitas” quânticas e executar um único
programa confirmatório: **Q-Shift**, uma avaliação pré-registrada de adaptadores
residuais quânticos sob mudança temporal real, com implantação clássica.

A pergunta primária é: um adaptador residual quântico treinado em simulador e,
depois, em hardware real melhora a degradação temporal e a calibração em relação a
um adaptador clássico pareado, sob o mesmo split, orçamento e número de parâmetros?

O primeiro alvo deve ser ACYD milho com corte temporal; GoBug cronológico serve
como replicação fora do domínio. Os baselines obrigatórios são regressão logística,
HistGradientBoosting/XGBoost, MLP pareada e adaptador residual clássico. Os controles
negativos são rótulo embaralhado, tempo embaralhado, circuito sem emaranhamento e
features aleatórias pareadas.

Alegação positiva somente se o desfecho primário pré-registrado vencer o baseline
forte após correção de multiplicidade no alvo selado, mantiver direção na réplica e
não depender de uma única semente. Caso contrário, a contribuição publicável é o
benchmark de falsificação e o resultado negativo.

## Base científica da mudança

- Bowles, Ahmed e Schuld mostram que desenho experimental e seleção de baseline
  mudam fortemente conclusões de QML e que modelos clássicos simples frequentemente
  vencem: https://arxiv.org/abs/2403.07059
- O trabalho sobre generalização de QNNs mostra que memorizar rótulos aleatórios é
  possível, justificando controles de randomização e avaliação fora da distribuição:
  https://www.nature.com/articles/s41467-024-45882-z
- O framework de vantagem prática em modelos generativos exige comparação clássica
  explícita e não apenas ganho contra ruído:
  https://www.nature.com/articles/s42005-024-01552-6
- O programa de reprodutibilidade do NeurIPS reforça código, checklist e reprodução
  como parte da evidência experimental:
  https://www.jmlr.org/papers/v22/20-303.html
- Modelos “shadow” motivam usar o recurso quântico no treino e manter implantação
  clássica, evitando chamar simulação barata de vantagem operacional:
  https://www.nature.com/articles/s41467-024-49877-8
- Um benchmark recente de QML sob envenenamento reforça que robustez deve ser
  comparada com controles clássicos e múltiplas execuções; Q-Shift transfere a
  pergunta para drift temporal real e um teste confirmatório selado:
  https://www.nature.com/articles/s41467-026-70420-4
