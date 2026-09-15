# Programa Lean Tiny Prover

## Decisão de pesquisa

A nova linha investiga se um provador neural especializado em Lean 4 pode ficar
substancialmente menor e mais econômico sem perder a maior parte da capacidade de
um baseline micro consolidado.

A pergunta primária é:

> Um sistema com no máximo 100 milhões de parâmetros totais consegue atingir pelo
> menos 85% dos teoremas resolvidos pelo ReProver de aproximadamente 300 milhões,
> usando no máximo um terço da memória de pesos e pelo menos três vezes mais
> teoremas válidos por unidade de compute?

A pergunta secundária permite candidatos de até 300 milhões de parâmetros:

> Um sistema micro consegue ser não inferior ao ReProver (margem proposta de
> -2 pontos percentuais) e simultaneamente duplicar teoremas válidos por GPU-hora?

Essas margens são propostas para pré-registro. Devem ser congeladas antes da
primeira avaliação selada e não podem ser ajustadas depois de observar resultados.

## O que o Lean valida — e o que não valida

O kernel Lean valida que um termo de prova tem o tipo da afirmação formalizada no
ambiente fixado. Esse é o gate de correção de cada prova.

Lean não demonstra que:

- a afirmação formal representa corretamente um enunciado em linguagem natural;
- uma arquitetura é superior a outra;
- o conjunto de teste não vazou para o treino;
- o ganho é estatisticamente estável ou eficiente em hardware.

Por isso a fase inicial aceita somente afirmações Lean já formalizadas. A
autoformalização fica fora do escopo até existir um protocolo humano ou um
benchmark com equivalência formal pareada.

Uma prova conta somente se:

- o hash da afirmação for idêntico ao original;
- compilar em ambiente limpo com versões Lean e mathlib pinadas;
- não contiver `sorry`, `admit` ou axiomas extras não autorizados;
- respeitar timeout, número de chamadas e orçamento de amostragem;
- puder ser reproduzida a partir do artefato salvo.

## Escala e hardware

Classes operacionais:

- simbólico: zero parâmetros aprendidos;
- nano: até 100M parâmetros totais;
- micro: acima de 100M e até 300M;
- pequeno: acima de 300M e até 1,5B;
- referência externa: acima de 1,5B.

O treino padrão fica limitado a 300M. Modelos de 4B/7B só podem ser usados como
teto de qualidade, inferência quantizada opcional ou professor offline. Nunca
serão usados para alegar vitória pareada de hardware.

A sonda local desta etapa encontrou 31 GiB de RAM e 120 GiB livres, mas o driver
NVIDIA não estava acessível. A fase de aquisição não deve baixar pesos grandes
antes de confirmar GPU, VRAM, CUDA, espaço, licença e checksum.

## Baselines obrigatórios

1. Táticas nativas/simbólicas: `simp`, `aesop`, `omega`, `linarith` e
   `nlinarith`, cada uma com orçamento explícito.
2. Recuperação não neural: filtro de tipos + BM25 no snapshot mathlib.
3. ReProver/ByT5-small sem recuperação, nominalmente 300M.
4. ReProver com retriever ByT5-small, reportando parâmetros totais e ativos.
5. DeepSeek-Prover-V2-7B como teto externo opcional.
6. Pythagoras-Prover-4B somente como referência recente pendente de reprodução e
   auditoria de licença.

Não foi localizado um provador Lean especialista consolidado abaixo de 100M. Isso
é uma lacuna de baseline e uma oportunidade de pesquisa; não deve ser preenchida
com um modelo próprio rotulado indevidamente como referência consolidada.

O registro canônico está em `research/lean/baselines.json`.

## Dados abertos e papéis

- LeanDojo Benchmark v5: treino e desenvolvimento de tática/recuperação.
- miniF2F: validação de desenvolvimento e teste final selado.
- ProofNet Lean 4: avaliação secundária de matemática universitária, somente após
  auditoria da licença da porta.
- miniCTX v2: avaliação fora de distribuição com contexto de projeto.
- PutnamBench: stress test final; nunca entra na seleção das 100 hipóteses.
- Lean Workbook: treino sintético opcional em quarentena, com deduplicação contra
  todas as avaliações.

Cada aquisição exige URL, revisão/commit, licença, checksum, tamanho, papel do
split e relatório de contaminação. O manifesto está em
`research/lean/datasets.json`. Nesta etapa nenhum corpus ou peso foi baixado.

## As 100 hipóteses

O registro `research/lean/hypotheses.json` contém exatamente 100 hipóteses,
organizadas em dez famílias:

- representação e tokenização;
- arquitetura compacta;
- recuperação de premissas;
- busca e planejamento;
- objetivos de treinamento;
- compressão e destilação;
- feedback formal;
- dados e currículo;
- memória e especialização;
- sistemas e roteamento.

Cada hipótese contém intervenção, mecanismo esperado, condição explícita de
falsificação, classe de custo e alvo de parâmetros. “Bizarra” significa ousada e
testável, não vaga. Toda hipótese deve receber ao menos uma implementação mínima
executável e um resultado registrado; só as sobreviventes recebem treino caro.

## Funil experimental 100 → 30 → 10 → 3

### Fase 0 — congelar o laboratório formal

Entregas:

- instalar Lean 4 via elan e criar projeto Lake separado;
- pinar Lean, mathlib e harness por commit;
- implementar executor sandboxado, hash de afirmação e scanner de proibições;
- registrar hardware e medir consumo de pico;
- criar conjunto smoke de 32 teoremas sem sobreposição com testes finais.

Gate: 100% das provas aceitas recompilam em ambiente limpo e 100% das provas
deliberadamente inválidas são rejeitadas.

### Fase 1 — adquirir e reproduzir baselines

Baixar primeiro o LeanDojo pequeno e o ReProver. Pesos de 4B/7B permanecem
opcionais.

Gate: reproduzir o baseline no mesmo commit/split ou documentar quantitativamente
a divergência antes de testar hipótese nova.

### Fase 2 — microteste de todas as 100 hipóteses

Cada LH-001…LH-100 recebe:

- branch/configuração isolada;
- uma única alteração causal;
- smoke em 32 teoremas;
- limite T0/T1;
- resultado válido, refutado, inconclusivo ou bloqueado;
- log, seed, duração, pico de RAM/VRAM e número de verificações.

Não é permitido descartar silenciosamente uma hipótese. As 30 melhores avançam
por Pareto: sucesso Lean, custo, parâmetros e novidade de teoremas resolvidos.

### Fase 3 — triagem das 30

Executar 256 teoremas de desenvolvimento, três seeds e orçamento idêntico.
Controlar falsas descobertas com Benjamini–Hochberg em q=0,10, deixando claro que
a fase é exploratória. Selecionar no máximo 10 sem consultar o teste.

### Fase 4 — confirmação das 10

Executar conjunto de validação congelado, cinco seeds e intervalos pareados por
bootstrap sobre teoremas. Usar teste de McNemar para sucesso binário pareado e
reportar tamanho de efeito. Aplicar Holm em alfa 0,05 às alegações confirmatórias.

Promover no máximo três configurações Pareto. Combinações de hipóteses só podem
ser testadas depois de efeitos individuais; a combinação vira nova hipótese
pré-registrada, não uma edição retroativa.

### Fase 5 — teste selado e generalização

Uma única abertura de miniF2F-test para as três finalistas. Depois, ProofNet Lean 4
e miniCTX em snapshots independentes. PutnamBench é apenas stress test.

Se nenhum candidato passar, o resultado publicável é o mapa completo de 100
refutações/inconclusões, com custo e artefatos.

## Métricas

Métrica primária:

- proporção de teoremas com ao menos uma prova aceita pelo Lean sob orçamento
  fixo, reportada como pass@k com k explícito.

Métricas de eficiência obrigatórias:

- teoremas resolvidos por hora de GPU e por hora total;
- energia por teorema quando o sensor estiver disponível;
- parâmetros totais, ativos e bytes de pesos;
- pico de RAM e VRAM;
- tokens gerados, nós expandidos, chamadas ao Lean e tempo de verificação;
- latência mediana e p95;
- tamanho do termo de prova;
- conjunto de teoremas resolvidos exclusivamente por cada sistema.

Perplexidade e validade sintática são diagnósticos, nunca desfechos principais.

## Regra de comparação justa

Toda linha comparada deve compartilhar:

- hashes idênticos das afirmações;
- mesma versão de Lean/mathlib;
- mesmo split e política de imports;
- mesmo timeout e máximo de chamadas ao verificador;
- mesmo pass@k e política de amostragem;
- mesma classe de hardware, ou normalização explicitamente separada.

Reportar duas fronteiras de Pareto:

- qualidade × parâmetros;
- qualidade × custo fim a fim.

Uma vitória contra um teto 7B não substitui a comparação com ReProver, táticas
simbólicas e BM25. Uma vitória em parâmetros que consome muito mais busca também
não é automaticamente eficiência.

## Artefatos por execução

Cada run deve salvar:

- `run.json` com configuração imutável;
- `environment.json` com commits e hardware;
- `statements.sha256`;
- candidatos brutos e respostas Lean;
- provas aceitas em `.lean`;
- `metrics.json`;
- checksums dos pesos;
- decisão do gate e justificativa.

Resultados e pesos promovidos entram na árvore versionada/DVC. Runs exploratórias
ficam em `.local/runs/lean/` até passarem o gate.

## Ordem prática recomendada

1. Fase 0: Lean, Lake, mathlib pinado e verificador adversarial.
2. Adquirir somente LeanDojo v5 e ReProver 300M.
3. Reproduzir táticas simbólicas, BM25 e ReProver.
4. Implementar primeiro LH-070 (hash da afirmação), LH-069 (replay limpo),
   LH-093 (roteador simbólico-neural), LH-024 (filtro por tipo) e LH-001
   (bytes canônicos).
5. Executar as 100 microimplementações antes de combinar ideias.
6. Só depois treinar um estudante 20M/50M/100M e comparar na fronteira Pareto.

Essa ordem começa com mecanismos de validade e baselines, depois explora
compressão. Ela reduz a chance de gastar semanas treinando um modelo cujo ganho
vem de vazamento, alteração da afirmação ou orçamento desigual.

## Fontes primárias consultadas em 2026-09-15

- Lean 4 e documentação: https://docs.lean-lang.org/theorem_proving_in_lean4/
- LeanDojo/ReProver: https://github.com/lean-dojo/ReProver
- LeanDojo Benchmark: https://zenodo.org/records/10114157
- miniF2F: https://github.com/facebookresearch/miniF2F
- ProofNet: https://github.com/zhangir-azerbayev/ProofNet
- miniCTX: https://cmu-l3.github.io/minictx/
- PutnamBench: https://github.com/trishullab/PutnamBench
- Lean Workbook: https://huggingface.co/datasets/internlm/Lean-Workbook
- DeepSeek-Prover-V2-7B: https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B
- Pythagoras-Prover-4B: https://huggingface.co/Pythagoras-LM/Pythagoras-Prover-4B

