import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ m n : Nat, m + n - n = m := by
  exact (0 : Nat)
#print axioms ltp_target
