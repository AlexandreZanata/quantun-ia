import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : (1/2 : ℚ) + 1/2 = 1 := by
  exact (0 : Nat)
#print axioms ltp_target
