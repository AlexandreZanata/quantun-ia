# Results — EXP 099: Masked climate SSL pretrain (ACYD maize)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`

## Validation gate

- Train rows: **151,956** | Val rows: **13,566**
- ResidualNano params: **840,321**
- Scratch AUC: **0.8110** (12 supervised epochs)
- SSL fine-tune AUC: **0.8143** (pretrain 8 + fine-tune 12)
- Pretrain MSE: **0.1601**
- HistGB (honesty) AUC: **0.8203**
- SSL vs scratch: **0.33 pp** (gate ≥ +0.5)
- Elapsed: **25.994s**

## Verdict
**rejected** — Phase D D-T5 masked climate SSL pretrain.

## Limitations
- Pretext masks seasonal weather aggregates (features 9–36), not raw weekly tensors.
- Matched supervised epochs; SSL spends additional pretrain epochs.
- Agro research benchmark — not operational planting advice.
