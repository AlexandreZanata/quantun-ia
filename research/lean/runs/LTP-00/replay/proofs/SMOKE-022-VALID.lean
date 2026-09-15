import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ a : Int, a^2 ≥ 0 := by
  intro a
  nlinarith [sq_nonneg a]
#print axioms ltp_target
