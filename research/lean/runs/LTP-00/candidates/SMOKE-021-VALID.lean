import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ a : Int, 0 ≤ |a| := by
  intro a
  exact abs_nonneg a
#print axioms ltp_target
