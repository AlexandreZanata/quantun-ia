import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ m n : Nat, m + n - n = m := by
  intro m n
  omega
#print axioms ltp_target
