import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ n : Nat, (n + 1)^2 = n^2 + 2*n + 1 := by
  intro n
  ring
#print axioms ltp_target
