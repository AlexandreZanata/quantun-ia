import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ a : Int, a * 2 = a + a := by
  intro a
  ring
#print axioms ltp_target
