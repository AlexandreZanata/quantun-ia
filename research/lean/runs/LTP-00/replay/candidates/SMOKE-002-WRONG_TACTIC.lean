import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ n : Nat, n + 0 = n := by
  exact (0 : Nat)
#print axioms ltp_target
