import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : (123456 * 654321) % 10 = 6 := by
  native_decide
#print axioms ltp_target
