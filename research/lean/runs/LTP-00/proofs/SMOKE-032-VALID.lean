import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : List.reverse (List.reverse [1, 2, 3]) = [1, 2, 3] := by
  decide
#print axioms ltp_target
