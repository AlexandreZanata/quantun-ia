import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : (2 : Nat) + 2 = 4 := by
  -- sorry
  -- admit
  -- axiom
  -- native_decide
  norm_num
#print axioms ltp_target
