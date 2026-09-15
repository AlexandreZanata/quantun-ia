import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ p q : Prop, p ∧ q → q ∧ p := by
  intro p q h
  rcases h with ⟨hp, hq⟩
  exact ⟨hq, hp⟩
#print axioms ltp_target
