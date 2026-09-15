import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ a b : Int, (a - b) + b = a := by
  intro a b
  ring
#print axioms ltp_target
