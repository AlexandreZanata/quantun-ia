import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : (1/2 : ℚ) + 1/2 = 1 := by
  norm_num
#print axioms ltp_target
