import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ n : Nat, n % 2 = 0 ∨ n % 2 = 1 := by
  exact (0 : Nat)
#print axioms ltp_target
