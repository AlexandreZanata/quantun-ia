import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ x : ℝ, x ≤ x + 1 := by
  intro x
  linarith
#print axioms ltp_target
