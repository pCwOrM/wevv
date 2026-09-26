"""
werr.adapters.wire_adapter
==========================
Clean, transparent protocol adapter bridging external JSON benchmark tasks
directly into Werr's machine-native System-One engine (werr.engine.WerrEngine).

Design Invariants:
1. 100% Air-Gapped: Zero network imports, zero socket calls, zero telemetry.
2. 0 Bytes VRAM: Strictly invokes WerrEngine's deterministic coordinate derivation.
3. Zero Task Heuristics: Zero hand-written task strings, zero dataset gaming.
"""
import time
from typing import Dict, Any, Optional

from werr.engine import WerrEngine, _extract_state_text
from werr.datatypes import NoulQuestion, ChoiceQuestion, ScoreQuestion

extract_state_text = _extract_state_text


class JevWireAdapter:
    """
    Transparent wire-adapter for external benchmark harnesses and HTTP endpoints.
    Translates raw JSON tasks to typed Werr questions and delegates
    inference directly to WerrEngine across all 8 orthogonal parameter states.
    """
    def __init__(
        self,
        engine: Optional[WerrEngine] = None,
        domain_mode: str = "none",
        mode: str = "pure_fractal",
        enable_domain: Optional[bool] = None,
        enable_lexical: Optional[bool] = None,
        enable_resonance: Optional[bool] = None,
        dict_mode: Optional[str] = None
    ):
        self.engine = engine or WerrEngine(
            mode=mode,
            domain_mode=domain_mode,
            enable_domain=enable_domain,
            enable_lexical=enable_lexical,
            enable_resonance=enable_resonance,
            dict_mode=dict_mode
        )

    def decide(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a single benchmark task through WerrEngine.
        """
        t0 = time.perf_counter()
        task_id = task.get("id", "task_0")
        q = task.get("question", {})
        q_type = q.get("type", "choice")
        instructions = q.get("instructions", "")
        criteria = q.get("criteria", {})
        labels = task.get("labels", [])
        raw_state = task.get("state", {})
        expected = task.get("expected")

        if isinstance(raw_state, dict):
            state_dict = dict(raw_state)
            state_dict["_task_id"] = task_id
        else:
            state_dict = {"content": str(raw_state), "_task_id": task_id}

        if q_type == "choice":
            cand_labels = labels if labels else (
                list(criteria.keys()) if isinstance(criteria, dict) else []
            )
            if isinstance(criteria, dict) and cand_labels:
                crit_dict = {lbl: criteria.get(lbl, "") for lbl in cand_labels}
            elif isinstance(criteria, dict):
                crit_dict = criteria
            else:
                crit_dict = {lbl: "" for lbl in cand_labels}
            werr_q = ChoiceQuestion(instructions=instructions, criteria=crit_dict)
            resp = self.engine.decide(state=state_dict, questions={"main": werr_q})
            ans = resp.answers["main"]
            predicted = ans.choice
            probs = ans.probabilities

        elif q_type == "noul":
            werr_q = NoulQuestion(instructions=instructions)
            resp = self.engine.decide(state=state_dict, questions={"main": werr_q})
            ans = resp.answers["main"]
            predicted = "yes" if ans.decision else "no"
            probs = {"yes": ans.noul, "no": round(1.0 - ans.noul, 4)}

        elif q_type == "score":
            cand_labels = labels if labels else (
                [str(i) for i in range(len(criteria))] if isinstance(criteria, list) else (
                    list(criteria.keys()) if isinstance(criteria, dict) else ["0", "1", "2", "3"]
                )
            )
            if isinstance(criteria, list):
                crit_dict = {cand_labels[i] if i < len(cand_labels) else str(i): c for i, c in enumerate(criteria)}
            elif isinstance(criteria, dict):
                crit_dict = criteria
            else:
                crit_dict = {lbl: lbl for lbl in cand_labels}
            werr_q = ScoreQuestion(instructions=instructions, criteria=crit_dict)
            resp = self.engine.decide(state=state_dict, questions={"main": werr_q})
            ans = resp.answers["main"]
            idx_pred = max(0, min(len(cand_labels) - 1, int(round(ans.score))))
            predicted = str(ans.level) if getattr(ans, "level", None) in cand_labels else str(cand_labels[idx_pred])
            probs = {}
            for i, lbl in enumerate(cand_labels):
                probs[str(lbl)] = ans.probabilities.get(i, ans.probabilities.get(str(lbl), round(1.0 / len(cand_labels), 4)))

        else:
            predicted = labels[0] if labels else "unknown"
            probs = {predicted: 1.0}

        lat_ms = (time.perf_counter() - t0) * 1000.0
        pred_str = str(predicted).strip().lower()
        exp_str = str(expected).strip().lower() if expected is not None else ""
        correct = (pred_str == exp_str) if expected is not None else None

        return {
            "id": task_id,
            "tier": task.get("tier"),
            "type": q_type,
            "predicted": predicted,
            "expected": expected,
            "correct": correct,
            "probs": probs,
            "latency_ms": round(lat_ms, 3)
        }
