import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ x : ℝ, x^2 + 1 > 0 := by
  exact (0 : Nat)
#print axioms ltp_target
