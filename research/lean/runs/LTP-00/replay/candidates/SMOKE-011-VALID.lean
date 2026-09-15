import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ n : Nat, n < n + 5 := by
  intro n
  omega
#print axioms ltp_target
