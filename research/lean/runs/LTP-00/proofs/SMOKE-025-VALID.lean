import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ p : Prop, p → ¬¬p := by
  intro p hp hnp
  exact hnp hp
#print axioms ltp_target
