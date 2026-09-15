import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ p q : Prop, p → q → p ∧ q := by
  intro p q hp hq
  exact ⟨hp, hq⟩
#print axioms ltp_target
