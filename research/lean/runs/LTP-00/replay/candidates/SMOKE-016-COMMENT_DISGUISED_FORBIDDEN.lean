import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ a : Int, a + 0 = a := by
  -- sorry
  -- admit
  -- axiom
  -- native_decide
  intro a
  ring
#print axioms ltp_target
