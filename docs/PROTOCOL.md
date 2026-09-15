# Protocolo obrigatório para nova pesquisa

Inspirado no controle de microfases do projeto `abu-mosca`.

## Antes de executar

1. Defina uma única hipótese primária, desfecho, direção, margem mínima relevante,
   baselines e controles negativos.
2. Faça análise de potência e fixe sementes, splits, orçamento de CPU/GPU/QPU e
   regra de parada.
3. Gere um cartão de pré-registro e seu SHA-256 antes de produzir qualquer resultado.
4. Separe desenvolvimento, validação e teste confirmatório. O teste fica selado.
5. Registre fonte, licença, versão, checksum e data de acesso de cada dataset.

## Durante a execução

- Execute primeiro fixture e smoke; smoke nunca conta como evidência.
- Não ajuste hiperparâmetros no teste confirmatório.
- Compare no mesmo split e orçamento contra baseline forte e baseline pareada.
- Registre todas as execuções, falhas e exclusões; não apague resultado negativo.
- Meça qualidade, calibração, tempo, memória, número de parâmetros e custo de shots.
- Uma microfase por sessão. Não comece a próxima ao terminar a atual.

## Gate confirmatório

Uma hipótese só passa quando:

- o desfecho primário pré-registrado passa no conjunto selado;
- intervalo de confiança e tamanho de efeito sustentam a margem definida;
- multiplicidade foi corrigida;
- controles negativos se comportam como esperado;
- o efeito mantém direção em um segundo dataset temporal;
- o pacote reproduzível contém ambiente, hashes, seeds, métricas e limitações;
- uma alegação quântica operacional inclui validação em hardware quântico real.

Falha no gate encerra a hipótese. Não trocar métrica, split ou narrativa depois do
resultado. Uma análise pós-hoc deve receber novo ID, rótulo exploratório e novo alvo
confirmatório.

## Registro de conclusão

Cada microfase concluída registra: data, executor, entradas com hashes, comando,
ambiente, recursos, resultado, decisão, limitações e commit. Gates que exigem revisão
humana preparam o pacote e param sem autoaprovação.

