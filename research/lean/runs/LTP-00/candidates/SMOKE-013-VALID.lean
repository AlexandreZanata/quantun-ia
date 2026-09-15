import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : (123456 * 654321) % 10 = 6 := by
  norm_num
#print axioms ltp_target
