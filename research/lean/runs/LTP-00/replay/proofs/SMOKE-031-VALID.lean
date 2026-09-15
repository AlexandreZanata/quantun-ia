import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ l : List Nat, 0 ≤ l.length := by
  intro l
  exact Nat.zero_le _
#print axioms ltp_target
