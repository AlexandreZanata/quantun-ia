# LTP-02 — Adquirir ReProver 300M e recontar parâmetros/checksum

- **Data**: 2026-09-15
- **Fase**: LTP-02 (única fase aberta; nenhuma fase seguinte iniciada)
- **Dependência**: LTP-01 concluída (`research/lean/ltp/LTP-01.md`, gate aprovado)
- **Orçamento**: T0; download de ~2,07 GB, inspeção estática em CPU, sem treino,
  sem inferência, sem serviço pago
- **Gate**: licença MIT auditada; revisão imutável pinada; sha256 de todo arquivo
  LFS confere com o publicado pelo Hugging Face; recontagem local de parâmetros
  compatível com o nominal e dentro do teto micro (≤300M para o gerador);
  nenhum conjunto de teste selado tocado; nada publicado
- **Decisão**: **gate passou**

## Objetivo

Auditar licença e revisão dos pesos ReProver (gerador de táticas e retriever de
premissas), baixar os arquivos nas revisões imutáveis, verificar checksums LFS e
recontar os parâmetros localmente a partir dos headers `safetensors`, sem
carregar tensores e sem treinar.

## Entregas

| Entrega | Artefato |
| --- | --- |
| Manifesto do gerador | `research/lean/acquisitions/reprover_tacgen_byt5_small.json` |
| Manifesto do retriever | `research/lean/acquisitions/reprover_retriever_byt5_small.json` |
| Snapshots públicos do HF | `research/lean/acquisitions/leandojo-lean4-*.metadata.json` |
| Gerador de manifesto de pesos | `scripts/build_hf_weight_manifest.py` |
| Validação por tipo (`dataset`/`weights`) | `scripts/validate_lean_research.py` |
| Dados locais (não versionados) | `data/raw/reprover/` |

## Auditoria de licença e revisão

| Repositório | Revisão (sha) | Licença | Última modificação |
| --- | --- | --- | --- |
| `kaiyuy/leandojo-lean4-tacgen-byt5-small` | `67a2c53cc36186fe8539d0a342fc42c50edc68fd` | MIT | 2024-07-16 |
| `kaiyuy/leandojo-lean4-retriever-byt5-small` | `ebccee9d0fb1b043db797a7493c30dfde6aa8ce2` | MIT | 2024-07-16 |

Evidência: `cardData.license = mit` e tag `license:mit` na API de modelos do HF.
Snapshots das respostas com sha256
`70e72b114c4b597fe564c173fb4e9fd901e2ae7e3b2a4093c9d1f0b142964b99` (tacgen) e
`47a2ae452bf930ac35d69a3957acf2ef4c38e7a6b9e2a958718839695d4c3c28` (retriever).

## Integridade dos arquivos

Todos os arquivos foram baixados de
`https://huggingface.co/<repo>/resolve/<revisão>/<arquivo>`. O único arquivo LFS
é `model.safetensors`; os demais não possuem hash publicado pelo HF e ficam com
sha256 local registrado sem comparação.

| Arquivo | Tamanho | sha256 LFS observado |
| --- | --- | --- |
| tacgen `model.safetensors` | 1.198.571.496 | `769fc3b498785e2492b35da4ee3e92a9a48a386807f2885258d6a4340aa2473c` |
| retriever `model.safetensors` | 870.643.240 | `0ce9e98141afa1d0f4e9f57c716c70798e3dd23efec570275182ff105e08ee9e` |

`checksums_verified = true` nos dois manifestos.

## Recontagem de parâmetros

Método: soma dos produtos dos `shape` no header `safetensors`, verificando que
`8 + tamanho_do_header + maior_offset` é igual ao tamanho do arquivo
(`layout_consistent = true`). Nenhum tensor foi materializado.

| Modelo | Classe | Configuração | Parâmetros armazenados | Nominal | Ativos |
| --- | --- | --- | --- | --- | --- |
| tacgen | `T5ForConditionalGeneration` | 12 enc. + 4 dec., d_model 1472, vocab 384, F32 | **299.637.760** | 300M | 299.637.760 |
| retriever | `T5EncoderModel` | 12 enc., d_model 1472, vocab 384, F32 | **217.657.472** | 200M | 217.657.472 |

Observações:

- `tie_word_embeddings = false` nos dois configs; não há peso compartilhado
  omitido da contagem.
- O gerador respeita o teto micro (≤300M) com folga de 362.240 parâmetros.
- O retriever real tem 217,7M, não 200M; o nominal de `baselines.json` deve ser
  lido como aproximação publicada e a contagem local é a autoritativa.
- O sistema ReProver completo é 299.637.760 + 217.657.472 = **517.295.232**
  parâmetros armazenados quando gerador e retriever são usados juntos. Toda
  comparação futura deve declarar se mede o gerador isolado ou o sistema.

## Comandos

```bash
curl -s "https://huggingface.co/api/models/kaiyuy/leandojo-lean4-tacgen-byt5-small?blobs=true" -o /tmp/...json
for FILE in .gitattributes README.md added_tokens.json config.json \
  generation_config.json model.safetensors special_tokens_map.json tokenizer_config.json; do
  curl -sSfL -o "data/raw/reprover/<repo>/<rev>/$FILE" \
    "https://huggingface.co/<repo>/resolve/<rev>/$FILE"
done
sha256sum data/raw/reprover/*/<rev>/model.safetensors
.venv/bin/python scripts/build_hf_weight_manifest.py --id ... --repo ... \
  --root data/raw/reprover/<repo>/<rev> --api-snapshot ... \
  --metadata-snapshot-out research/lean/acquisitions/<repo>.metadata.json \
  --out research/lean/acquisitions/<id>.json --role "..."
```

## Gate

| Critério | Resultado |
| --- | --- |
| Licença MIT auditada e registrada | sim, nos dois repositórios |
| Revisão imutável pinada (sha 40 hex) | sim |
| sha256 dos arquivos LFS confere | sim, nos dois `model.safetensors` |
| Recontagem local de parâmetros | sim, 299.637.760 e 217.657.472 |
| Gerador dentro do teto micro (≤300M) | sim |
| Nenhum conjunto selado tocado | sim (`sealed_tests_touched: false`) |
| Nada publicado ou enviado | sim |
| Nenhum serviço pago | sim |

## Limitações

- Nenhuma inferência foi executada; a recontagem cobre parâmetros armazenados,
  não FLOPs, latência ou memória de ativação.
- Arquivos não LFS não têm hash publicado pelo HF; seus sha256 locais ficam
  registrados sem comparação externa.
- A compatibilidade do tokenizer com o LeanDojo v5 e com o ambiente Lean pinado
  não foi testada; isso pertence a LTP-03.
- O retriever nominal de 200M difere 8,8% da contagem local; a contagem local
  passa a ser a referência.
- Os 517,3M do sistema completo não são um modelo único; comparações de
  parâmetros devem separar gerador e sistema.

## Decisão

Gate aprovado. LTP-02 encerrada. A próxima fase (LTP-03, reprodução de táticas
simbólicas, BM25 e ReProver no mesmo protocolo) **não** foi iniciada.

## Prompt para iniciar LTP-03 (não executar agora)

```text
Trabalhe no projeto /home/iiii/PESSOAL-PROJETOS-ALEXANDRE/quantun-ia como
executor do programa Lean Tiny Prover. Antes de agir, leia README.md,
docs/LEAN_RESEARCH_PROGRAM.md, research/lean/hypotheses.json,
research/lean/datasets.json, research/lean/baselines.json,
research/lean/acquisitions/*.json, research/lean/ltp/LTP-00.md,
research/lean/ltp/LTP-01.md, research/lean/ltp/LTP-02.md e
.local/PROMPT-CONTINUAR.md. Rode git status --short, make lean-plan-check e
make check. Verifique Lean/Lake/elan, GPU/VRAM, RAM e disco. Execute SOMENTE
LTP-03: pré-registrar e reproduzir, no mesmo protocolo, as táticas simbólicas
(simp/aesop/omega/linarith/nlinarith com orçamento explícito), o baseline BM25
com filtro de tipo e o ReProver 300M, usando o LeanDojo v5 já adquirido e o
ambiente Lean/mathlib pinado em LTP-00; decida e documente o alinhamento de
versões de mathlib antes de medir; não abra conjuntos de teste selados, não
treine nada e não use serviços pagos. Registre versões, comandos, hardware,
hashes de afirmações, timeout, pass@k, chamadas ao verificador, métricas,
limitações e decisão em research/lean/ltp/LTP-03.md; promova artefatos apenas se
o gate (reproduzir o baseline no mesmo commit/split ou documentar
quantitativamente a divergência) passar; pare antes de LTP-04.
```
