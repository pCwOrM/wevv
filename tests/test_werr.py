"""
Unit tests for the werr System-One Fractal Decision Engine.
"""
import unittest
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from werr import (
    WerrEngine,
    NoulQuestion,
    ChoiceQuestion,
    ScoreQuestion,
    NoulAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    create_security_guard,
    create_smart_router,
    create_risk_evaluator
)


class TestWerrEngine(unittest.TestCase):

    def setUp(self):
        self.engine = WerrEngine(resolution=32, max_iter=30)
        self.sample_state = {
            "user_id": 4291,
            "role": "admin",
            "is_internal": True,
            "latency": 45.2,
            "error_count": 0
        }

    def test_zero_memory_contract(self):
        """Verify the engine strictly adheres to the 0-byte matrix tensor contract."""
        response = self.engine.decide(
            state=self.sample_state,
            questions={"check": NoulQuestion(instructions="Is valid?")}
        )
        self.assertEqual(response.memory_tensor_bytes, 0)
        self.assertEqual(response.coordinate_bytes, 24)

    def test_noul_primitive(self):
        """Test binary/boolean decision evaluation."""
        q = NoulQuestion(instructions="Authorize action?", threshold=0.5)
        resp = self.engine.decide(state=self.sample_state, questions={"auth": q})

        ans = resp.answers["auth"]
        self.assertIsInstance(ans, NoulAnswer)
        self.assertEqual(ans.type, "noul")
        self.assertTrue(0.0 <= ans.noul <= 1.0)
        self.assertIsInstance(ans.decision, bool)
        self.assertTrue(0.0 <= ans.confidence <= 1.0)
        # Helper methods
        self.assertEqual(resp.noul("auth"), ans.noul)
        self.assertEqual(resp.boolean("auth"), ans.decision)

    def test_choice_primitive_quadrant(self):
        """Test categorical choice with <= 4 options (4-Quadrant mode)."""
        criteria = {
            "fast": "Fast lane",
            "normal": "Normal lane",
            "slow": "Slow lane"
        }
        q = ChoiceQuestion(instructions="Route request", criteria=criteria)
        resp = self.engine.decide(state=self.sample_state, questions={"route": q})

        ans = resp.answers["route"]
        self.assertIsInstance(ans, ChoiceAnswer)
        self.assertEqual(ans.type, "choice")
        self.assertIn(ans.choice, criteria.keys())
        self.assertAlmostEqual(sum(ans.probabilities.values()), 1.0, places=2)
        self.assertTrue(0.0 <= ans.confidence <= 1.0)
        self.assertEqual(resp.choice("route"), ans.choice)

    def test_choice_primitive_quadtree(self):
        """Test categorical choice with > 4 options (Quadtree 2^p mode)."""
        criteria = {f"opt_{i}": f"Option description {i}" for i in range(8)}
        q = ChoiceQuestion(instructions="Select from 8 options", criteria=criteria)
        resp = self.engine.decide(state=self.sample_state, questions={"choice8": q})

        ans = resp.answers["choice8"]
        self.assertIn(ans.choice, criteria.keys())
        self.assertAlmostEqual(sum(ans.probabilities.values()), 1.0, places=2)

    def test_score_primitive(self):
        """Test scalar ordinal score evaluation."""
        criteria = ["Very Low", "Low", "Medium", "High", "Critical"]
        q = ScoreQuestion(instructions="Rate priority", criteria=criteria)
        resp = self.engine.decide(state=self.sample_state, questions={"priority": q})

        ans = resp.answers["priority"]
        self.assertIsInstance(ans, ScoreAnswer)
        self.assertEqual(ans.type, "score")
        self.assertTrue(0.0 <= ans.score <= len(criteria) - 1)
        self.assertAlmostEqual(sum(ans.probabilities.values()), 1.0, places=2)
        self.assertEqual(resp.score("priority"), ans.score)

    def test_bundled_parallel_execution(self):
        """Test multiple questions evaluated in a single forward pass."""
        questions = {
            "allow": NoulQuestion(instructions="Allow access?"),
            "lane": ChoiceQuestion(instructions="Select lane", criteria={"a": "Lane A", "b": "Lane B"}),
            "rank": ScoreQuestion(instructions="Rank request", criteria=["Low", "Med", "High"])
        }
        resp = self.engine.decide(state=self.sample_state, questions=questions)

        self.assertEqual(len(resp.answers), 3)
        self.assertIn("allow", resp.answers)
        self.assertIn("lane", resp.answers)
        self.assertIn("rank", resp.answers)
        self.assertTrue(resp.latency_ms > 0.0)

    def test_presets(self):
        """Verify pre-calibrated gate presets."""
        guard = create_security_guard(resolution=32)
        router = create_smart_router(resolution=32)
        evaluator = create_risk_evaluator(resolution=32)

        for p_name, eng in [("guard", guard), ("router", router), ("eval", evaluator)]:
            resp = eng.decide(self.sample_state, {"n": NoulQuestion("Test")})
            self.assertIsInstance(resp.answers["n"], NoulAnswer, f"Preset {p_name} failed")

    def test_turkish_semantic_support(self):
        """Verify that Turkish semantic terms for roles and questions evaluate correctly."""
        # 1. Turkish Admin
        resp_admin = self.engine.decide(
            state={"rol": "Yönetici", "hata_sayisi": 0, "hiz": 2.5},
            questions={
                "izin": NoulQuestion(instructions="Bu isteğe geçiş izni verilsin mi?"),
                "rota": ChoiceQuestion(instructions="Yönlendir", criteria={"dogrudan": "Doğrudan", "engelle": "Engelle"}),
                "risk": ScoreQuestion(instructions="Risk", criteria=["Düşük", "Orta", "Yüksek"])
            }
        )
        self.assertTrue(resp_admin.boolean("izin"))
        self.assertEqual(resp_admin.choice("rota"), "dogrudan")

        # 2. Turkish Attacker
        resp_att = self.engine.decide(
            state={"rol": "Saldırgan Bot", "hata_sayisi": 15, "hiz": 100.0},
            questions={
                "izin": NoulQuestion(instructions="İşlem onaylansın mı?"),
                "rota": ChoiceQuestion(instructions="Yönlendir", criteria={"dogrudan": "Doğrudan", "engelle": "Engelle"}),
                "risk": ScoreQuestion(instructions="Risk", criteria=["Düşük", "Orta", "Yüksek"])
            }
        )
        self.assertFalse(resp_att.boolean("izin"))
        self.assertEqual(resp_att.choice("rota"), "engelle")


class TestMultiDomainRouting(unittest.TestCase):
    """
    Tests the Multi-Domain Auto-Seed Router and heterogeneous domain gates.
    """
    def setUp(self):
        from werr.router import AutoSeedRouter
        from werr.gates import (
            APISecurityGate,
            FinancialRiskGate,
            IoTSafetyGate,
            EcommerceFraudGate,
            GameCombatGate
        )
        self.router = AutoSeedRouter()
        self.fin_gate = FinancialRiskGate()
        self.iot_gate = IoTSafetyGate()
        self.fraud_gate = EcommerceFraudGate()
        self.combat_gate = GameCombatGate()

    def test_router_domain_detection(self):
        # 1. Financial
        d, conf, _ = self.router.detect_domain(
            query="Approve credit facility for applicant?",
            state={"income": 50000, "debt_ratio": 0.25}
        )
        self.assertEqual(d, "financial_risk")
        self.assertTrue(conf >= 0.8)

        # 2. IoT Safety
        d, conf, _ = self.router.detect_domain(
            query="Hazard alert: evacuate building?",
            state={"temp_c": 75.0, "smoke_detected": True}
        )
        self.assertEqual(d, "iot_safety")
        self.assertTrue(conf >= 0.8)

        # 3. E-Commerce
        d, conf, _ = self.router.detect_domain(
            query="Flag suspicious transaction on checkout?",
            state={"order_amount": 1200.0, "velocity_1h": 5}
        )
        self.assertEqual(d, "ecommerce_fraud")
        self.assertTrue(conf >= 0.8)

        # 4. Game Combat
        d, conf, _ = self.router.detect_domain(
            query="Tactical reflex: engage enemy?",
            state={"ammo": 40, "enemy_count": 1}
        )
        self.assertEqual(d, "game_combat")
        self.assertTrue(conf >= 0.8)

    def test_financial_gate_decisions(self):
        # Good profile -> Approve
        resp_good = self.fin_gate.evaluate_state_and_questions(
            state={"income": 85000, "debt_ratio": 0.20, "requested_amount": 10000, "credit_score": 750},
            questions={"loan": NoulQuestion("Approve credit facility?")}
        )
        self.assertTrue(resp_good.boolean("loan"))

        # Toxic profile -> Reject
        resp_bad = self.fin_gate.evaluate_state_and_questions(
            state={"income": 18000, "debt_ratio": 0.75, "requested_amount": 25000, "credit_score": 510, "late_payments": 3},
            questions={"loan": NoulQuestion("Approve credit facility?")}
        )
        self.assertFalse(resp_bad.boolean("loan"))

    def test_iot_gate_decisions(self):
        # Safe -> No alert
        resp_safe = self.iot_gate.evaluate_state_and_questions(
            state={"temp": 22.0, "smoke_detected": False, "gas_ppm": 10.0},
            questions={"hazard": NoulQuestion("Hazard alert detected?")}
        )
        self.assertFalse(resp_safe.boolean("hazard"))

        # Fire Emergency -> Alert
        resp_fire = self.iot_gate.evaluate_state_and_questions(
            state={"temp": 82.0, "smoke_detected": True, "gas_ppm": 550.0},
            questions={"hazard": NoulQuestion("Hazard alert detected?")}
        )
        self.assertTrue(resp_fire.boolean("hazard"))

    def test_engine_auto_route_seamless(self):
        engine = WerrEngine()
        # Call with auto_route=True on IoT emergency state
    def test_engine_domain_mode_multi(self):
        engine = WerrEngine(domain_mode="multi")
        self.assertEqual(engine.domain_mode, "multi")
        resp = engine.decide(
            state={"temp": 88.0, "smoke_detected": True},
            questions={"alert": NoulQuestion("Hazard alert detected?")}
        )
        self.assertTrue(resp.boolean("alert"))

    def test_engine_domain_mode_none(self):
        engine = WerrEngine(domain_mode="none", mode="pure_fractal")
        self.assertEqual(engine.domain_mode, "none")
        resp = engine.decide(
            state={"status": "active", "valid": True},
            questions={"ok": NoulQuestion("Is operation valid?")}
        )
        self.assertTrue(resp.boolean("ok"))

    def test_jev_wire_adapter(self):
        from werr.adapters import JevWireAdapter
        adapter = JevWireAdapter(domain_mode="none")
        sample_task = {
            "id": "sample-task-01",
            "question": {
                "type": "choice",
                "instructions": "Select appropriate action",
                "criteria": {"allow": "permit access", "deny": "block connection"}
            },
            "state": {"status": "authorized", "clear": True},
            "expected": "allow"
        }
        res = adapter.decide(sample_task)
        self.assertIn("predicted", res)
        self.assertIn("probs", res)
        self.assertLess(res["latency_ms"], 50.0)

    def test_8_state_orthogonal_parameter_matrix(self):
        """Verify all 8 boolean combinations of (enable_domain, enable_lexical, enable_resonance)."""
        for d_on in (False, True):
            for l_on in (False, True):
                for r_on in (False, True):
                    eng = WerrEngine(
                        enable_domain=d_on,
                        enable_lexical=l_on,
                        enable_resonance=r_on,
                        resolution=24,
                        max_iter=24
                    )
                    if not d_on:
                        # Rule 1: When Domain is OFF, domain dictionaries are locked OFF
                        self.assertFalse(eng.enable_lexical)
                        self.assertFalse(eng.enable_resonance)
                        self.assertFalse(eng.effective_domain_active)
                    elif not l_on and not r_on:
                        # Rule 2: [1,0,0] auto-guards to Universal Cusp
                        self.assertFalse(eng.effective_domain_active)
                    else:
                        # Rule 3: [1,1,0], [1,0,1], [1,1,1] activate domain + selected dictionaries
                        self.assertTrue(eng.effective_domain_active)
                        self.assertEqual(eng.enable_lexical, l_on)
                        self.assertEqual(eng.enable_resonance, r_on)


if __name__ == "__main__":
    unittest.main()
