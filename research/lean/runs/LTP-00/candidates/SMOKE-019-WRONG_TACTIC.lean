import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : (-5 : Int) + 8 = 3 := by
  exact (0 : Nat)
#print axioms ltp_target
