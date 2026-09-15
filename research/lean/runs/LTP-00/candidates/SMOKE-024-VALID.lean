import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ p q r : Prop, (p → q) → (q → r) → p → r := by
  intro p q r hpq hqr hp
  exact hqr (hpq hp)
#print axioms ltp_target
