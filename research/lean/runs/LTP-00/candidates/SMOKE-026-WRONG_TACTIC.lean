import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ p q : Prop, p → q → p ∧ q := by
  exact (0 : Nat)
#print axioms ltp_target
