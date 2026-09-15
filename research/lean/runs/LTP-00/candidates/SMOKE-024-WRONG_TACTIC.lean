import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ p q r : Prop, (p → q) → (q → r) → p → r := by
  exact (0 : Nat)
#print axioms ltp_target
