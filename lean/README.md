# Projeto Lean pinado

Projeto Lake separado do programa Lean Tiny Prover.

- Toolchain: `leanprover/lean4:v4.34.0` (`lean-toolchain`).
- mathlib: `v4.34.0`, commit `5ed2965256430c3649e86755f9576b54eca72435`
  (`lake-manifest.json`).
- Build: `~/.elan/bin/lake build Mathlib` após `lake exe cache get`.
- O harness adversarial fica em `../scripts/lean_verify.py` e verifica
  candidatos com `lake env lean` resolvido uma única vez (binário + `LEAN_PATH`).

Comandos úteis, a partir da raiz do repositório:

```bash
export PATH="$HOME/.elan/bin:$PATH"
cd lean && lake build Mathlib
cd lean && lake env lean arquivo.lean
make lean-gate
```

O diretório `.lake/` não é versionado.
