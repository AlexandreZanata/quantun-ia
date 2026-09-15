import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ p : Prop, p ∨ ¬p := by
  intro p
  exact Classical.em p
#print axioms ltp_target
