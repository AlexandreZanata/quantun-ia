import Mathlib.Tactic
set_option autoImplicit false
set_option maxHeartbeats 200000
theorem ltp_target : ∀ n : Nat, n * 1 = n := by
  have _h : String := "sorry admit axiom native_decide"
  simp
#print axioms ltp_target
