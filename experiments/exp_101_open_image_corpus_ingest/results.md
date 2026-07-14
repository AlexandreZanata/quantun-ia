# Results — EXP 101: Open image corpus ingest

**Run date:** 2026-07-14  
**Hardware:** CPU synthesis (verifies workstation downloads)
**Profile:** `publication`

## Validation gate

- Packs complete: **3/3**
- Splits ready: **3/3**
- Smoke shapes: **True**
- Elapsed: **2.264s**

| Pack | Complete | Stats | Shape OK |
|------|----------|-------|----------|
| cifar10 | True | True | True |
| fashion_mnist | True | True | True |
| flowers102 | True | True | True |

## Verdict
**Hypothesis confirmed** — Phase G P0 accept pack gate.

## Limitations
- Caption packs (G-T3) not included.
- Raw blobs gitignored; checksums in download_stats.json.
