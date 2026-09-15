import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ a b : Nat, a + b = b + a := by
  intro a b
  omega
#print axioms ltp_target
