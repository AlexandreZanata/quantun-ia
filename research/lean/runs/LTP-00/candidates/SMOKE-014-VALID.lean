import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ n : Nat, 2 * n = n + n := by
  intro n
  ring
#print axioms ltp_target
