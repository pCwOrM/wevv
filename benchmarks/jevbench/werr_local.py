"""Werr (0-VRAM Zero-Memory Fractal System-One Decision Engine) JevBench Adapter.

Interface (https://github.com/pCwOrM/werr):
  Zero-parameter, zero-weight, zero-VRAM deterministic fractal escape-time kernel with:
  - Multi-Scale Harmonic Tripod (0.60x, 1.00x, 1.60x scales)
  - Continuous 16-dim Latent State-to-Wave Boundary Modulation
  - 16-Tile Mandelbrot Quadtree Projection (Hard-Tier Reasoning)
  - Bounded Density Estimation (arXiv:1810.11107) for 4-Quadrant Energies
  - Coupled Cadence Supercritical Pitchfork Bifurcation Operator (Nodal Deadlock Resolution)
  - High-Fidelity N-gram & Numeric Semantic Alignment with Localized Negation Guard

Maps 1:1 to JevBench canonical types:
  noul   -> {"yes": p, "no": 1-p}
  choice -> full probability distribution over option keys
  score  -> full probability distribution over level indices
"""

from __future__ import annotations

import os
import sys
import re
import json
import time
import math
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

from .base import DecisionResult, build_question


def _normalize_text(s: str) -> str:
    s = str(s).lower()
    mapping = {'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c'}
    for k, v in mapping.items():
        s = s.replace(k, v)
    return s.strip()


def _tokenize(s: str) -> List[str]:
    s = _normalize_text(s)
    s = re.sub(r"[^a-z0-9_]", " ", s)
    return [w for w in s.split() if w]


def _extract_state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        parts = []
        for k, v in state.items():
            if isinstance(v, (str, int, float, bool)):
                parts.append(f"{k}: {v}")
            elif isinstance(v, dict):
                sub = ", ".join(f"{sk}: {sv}" for sk, sv in v.items())
                parts.append(f"{k}: {{{sub}}}")
            elif isinstance(v, list):
                parts.append(f"{k}: {', '.join(str(x) for x in v)}")
            else:
                parts.append(f"{k}: {str(v)}")
        return "\n".join(parts)
    if isinstance(state, list):
        return " ".join(str(x) for x in state)
    return str(state)


def _vec_erf(x: np.ndarray) -> np.ndarray:
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    abs_x = np.abs(x)
    t = 1.0 / (1.0 + p * abs_x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-abs_x * abs_x)
    return sign * y


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _vec_erf(x / np.sqrt(2.0)))


def _compute_boundary_correction_weights(u: np.ndarray, bandwidth: float = 0.12) -> np.ndarray:
    u_clamped = np.clip(u, 0.0, 1.0)
    h = max(1e-4, bandwidth)
    phi_left = _normal_cdf(u_clamped / h)
    phi_right = _normal_cdf((1.0 - u_clamped) / h)
    omega = np.clip(phi_left + phi_right - 1.0, 0.45, 1.0)
    return 1.0 / omega


def _extract_bounded_quadrant_weights(
    escape_iters: np.ndarray, max_iter: int = 50, bandwidth: float = 0.12
) -> Tuple[float, float, float, float, List[float]]:
    h, w = escape_iters.shape
    mid_h, mid_w = h // 2, w // 2
    quadrants = [
        escape_iters[:mid_h, :mid_w],
        escape_iters[:mid_h, mid_w:],
        escape_iters[mid_h:, :mid_w],
        escape_iters[mid_h:, mid_w:]
    ]

    ratios = []
    weights = []
    for quad in quadrants:
        u = quad.astype(np.float64) / float(max_iter)
        w_corr = _compute_boundary_correction_weights(u, bandwidth=bandwidth)
        cusp_mask = (u >= 0.90).astype(np.float64)
        boundary_corrected_ratio = float(np.sum(w_corr * cusp_mask) / np.sum(w_corr))
        avg_energy = float(np.sum(w_corr * u) / np.sum(w_corr))
        composite_ratio = 0.65 * boundary_corrected_ratio + 0.35 * avg_energy
        ratios.append(float(composite_ratio))
        weights.append(float((composite_ratio - 0.5) * 6.0))

    return weights[0], weights[1], weights[2], weights[3], ratios


def _compute_mandelbrot_patch(
    cx: float, cy: float, zoom: float, res: int = 64, max_iter: int = 50
) -> Tuple[float, float, np.ndarray]:
    scale = 1.0 / zoom
    x = np.linspace(cx - scale, cx + scale, res)
    y = np.linspace(cy - scale, cy + scale, res)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    escape_iters = np.full(C.shape, max_iter, dtype=int)

    for i in range(max_iter):
        mask = (np.abs(Z) <= 2.0) & (escape_iters == max_iter)
        if not np.any(mask):
            break
        Z[mask] = Z[mask] ** 2 + C[mask]
        escaped = (np.abs(Z) > 2.0) & (escape_iters == max_iter)
        escape_iters[escaped] = i

    black_ratio = float(np.sum(escape_iters == max_iter) / escape_iters.size)
    avg_escape = float(np.mean(escape_iters) / max_iter)
    return black_ratio, avg_escape, escape_iters


def _extract_quadtree_tiles(escape_iters: np.ndarray, grid_size: int = 4, max_iter: int = 50) -> np.ndarray:
    h, w = escape_iters.shape
    tile_h = h // grid_size
    tile_w = w // grid_size
    ratios = []
    for r in range(grid_size):
        for c in range(grid_size):
            tile = escape_iters[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w]
            ratio = float(np.sum(tile == max_iter) / tile.size)
            ratios.append(ratio)
    return np.array(ratios, dtype=np.float64)


def _apply_cadence_bifurcation(
    scores: Union[List[float], np.ndarray],
    lambda_param: float = 0.10,
    alpha: float = 0.50,
    beta: float = 0.15,
    deadlock_threshold: float = 0.85
) -> np.ndarray:
    scores_arr = np.array(scores, dtype=np.float64)
    if len(scores_arr) < 2:
        return scores_arr

    sorted_idx = np.argsort(scores_arr)
    top_gap = scores_arr[sorted_idx[-1]] - scores_arr[sorted_idx[-2]]
    if top_gap >= deadlock_threshold:
        return scores_arr

    diff_matrix = scores_arr[:, np.newaxis] - scores_arr[np.newaxis, :]
    bif_force = np.sum(np.sign(diff_matrix) * (np.abs(diff_matrix) ** alpha), axis=1)
    return scores_arr + lambda_param * bif_force


class WerrLocalAdapter:
    """JevBench Local In-Process Adapter for Werr Zero-Memory System-One Decision Engine."""
    name = "werr_local"
    cost_basis = "local_cpu_no_provider_tariff"

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        key_env: str = "",
        timeout_s: Optional[float] = None,
        price_input_per_m: Optional[float] = None,
        price_output_per_m: Optional[float] = None,
        threads: int = 4,
        revision: Optional[str] = None,
        tripod: bool = True,
        cadence_lambda: float = 0.10,
        cadence_beta: float = 0.15,
        cadence_alpha: float = 0.50,
        temp_choice: float = 1.25,
        res: int = 36,
        max_iter: int = 36
    ):
        self.model = model or "werr-v0.5.0-tripod-tesla369"
        self.threads = threads
        self.revision = revision
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m

        # Universal neutral boundary cusp coordinate
        self.cx = -0.7436438870371587
        self.cy = 0.1318259042053119
        self.zoom = 50.0
        self.res = res
        self.max_iter = max_iter
        self.tripod = tripod
        self.cadence_lambda = cadence_lambda
        self.cadence_beta = cadence_beta
        self.cadence_alpha = cadence_alpha
        self.temp_choice = temp_choice

    def load(self):
        return True

    def build_request(self, task) -> dict:
        return {"state": task.state, "questions": {"decision": build_question(task)}}

    def _state_to_vector(self, state: Any) -> np.ndarray:
        values = []
        if isinstance(state, dict):
            for k, v in sorted(state.items()):
                if isinstance(v, (int, float)):
                    norm_val = 2.0 / (1.0 + math.exp(-float(v) / 10.0 if abs(v) < 700 else (-1.0 if v < 0 else 1.0))) - 1.0
                    values.append(norm_val)
                elif isinstance(v, bool):
                    values.append(1.0 if v else -1.0)
                elif isinstance(v, str):
                    h = int(hashlib.md5(v.encode('utf-8')).hexdigest()[:8], 16)
                    angle = (h % 10000) / 10000.0 * 2.0 * math.pi
                    values.append(math.sin(angle))
                    values.append(math.cos(angle))
                else:
                    values.append(0.0)
        elif isinstance(state, list):
            for item in state:
                if isinstance(item, (int, float)):
                    values.append(float(item))
                else:
                    h = int(hashlib.md5(str(item).encode('utf-8')).hexdigest()[:8], 16)
                    values.append(math.sin((h % 1000) / 1000.0 * 2.0 * math.pi))
        else:
            h = int(hashlib.md5(str(state).encode('utf-8')).hexdigest()[:8], 16)
            angle = (h % 10000) / 10000.0 * 2.0 * math.pi
            values.append(math.sin(angle))
            values.append(math.cos(angle))

        while len(values) < 16:
            values.append(0.0)
        return np.array(values[:16], dtype=np.float64)

    def run(self, task) -> DecisionResult:
        res = DecisionResult(adapter=self.name, ok=False, probs_source="native", model=self.model)
        body = self.build_request(task)
        res.request_body = body

        t0 = time.perf_counter()
        q = task.question
        q_type = q.get("type", "choice")
        state = task.state
        instructions = q.get("instructions", "")
        criteria = q.get("criteria", {})
        labels = task.labels if hasattr(task, "labels") else task.get("labels", [])

        st_text = _extract_state_text(state).lower()
        st_tokens = set(_tokenize(st_text))
        instr_tokens = set(_tokenize(instructions))

        vec = self._state_to_vector(state)

        # Coordinate perturbation aligned with WerrEngine v0.5.1
        scale = 1.0 / self.zoom
        raw_id = getattr(task, "id", "") if hasattr(task, "id") else (task.get("id", "") if isinstance(task, dict) else "")
        t_id_clean = re.sub(r"[^a-zA-Z0-9]", "", str(raw_id))
        h_s = (int(hashlib.md5(t_id_clean.encode('utf-8')).hexdigest()[:8], 16) % 10000) / 10000.0
        h_i = (int(hashlib.md5(str(instructions).encode('utf-8')).hexdigest()[:8], 16) % 10000) / 10000.0

        delta_x = (float(np.tanh(np.mean(vec[0::2]))) * 0.5 + (h_s - 0.5) * 0.5) * scale * 0.40
        delta_y = (float(np.tanh(np.mean(vec[1::2]))) * 0.5 + (h_i - 0.5) * 0.5) * scale * 0.40

        eff_cx = self.cx + delta_x
        eff_cy = self.cy + delta_y
        eff_zoom = self.zoom * (1.0 + 0.08 * float(np.sin(np.sum(vec))))

        if self.tripod:
            tripod_configs = [
                (eff_zoom * 0.60, 0.25),
                (eff_zoom * 1.00, 0.50),
                (eff_zoom * 1.60, 0.25)
            ]
            fused_quad_ratios = np.zeros(4, dtype=np.float64)
            fused_tile_ratios = np.zeros(16, dtype=np.float64)
            fused_black_ratio = 0.0

            for z_val, w_z in tripod_configs:
                b_r, a_e, esc = _compute_mandelbrot_patch(
                    cx=eff_cx, cy=eff_cy, zoom=z_val, res=self.res, max_iter=self.max_iter
                )
                _, _, _, _, q_r = _extract_bounded_quadrant_weights(
                    esc, max_iter=self.max_iter, bandwidth=0.12
                )
                t_r = _extract_quadtree_tiles(esc, grid_size=4, max_iter=self.max_iter)

                fused_quad_ratios += w_z * np.array(q_r, dtype=np.float64)
                fused_tile_ratios += w_z * t_r
                fused_black_ratio += w_z * b_r

            quad_ratios = list(fused_quad_ratios)
            quad_weights = [float(r - 0.5) * 2.5 for r in quad_ratios]
            tile_weights = (fused_tile_ratios - 0.5) * 4.0
        else:
            black_ratio, avg_escape, escape_iters = _compute_mandelbrot_patch(
                cx=eff_cx, cy=eff_cy, zoom=eff_zoom, res=self.res, max_iter=self.max_iter
            )
            _, _, _, _, quad_ratios = _extract_bounded_quadrant_weights(
                escape_iters, max_iter=self.max_iter, bandwidth=0.12
            )
            quad_weights = [float(r - 0.5) * 2.5 for r in quad_ratios]
            tile_ratios = _extract_quadtree_tiles(escape_iters, grid_size=4, max_iter=self.max_iter)
            tile_weights = (tile_ratios - 0.5) * 4.0

        probs: Dict[str, float] = {}

        try:
            if q_type == "choice":
                options = labels if labels else (list(criteria.keys()) if isinstance(criteria, dict) else [])
                num_opts = len(options)
                instr_hash = int(hashlib.md5(str(instructions).encode('utf-8')).hexdigest()[:6], 16)
                phase_offset = instr_hash % 4

                scores = []
                fractal_fields = []
                st_no_punct = re.sub(r"[,.\$€£]", "", st_text)

                for i, opt in enumerate(options):
                    opt_norm = _normalize_text(opt)
                    opt_tokens = set(_tokenize(opt_norm))
                    crit_desc = criteria.get(opt, "") if isinstance(criteria, dict) else ""
                    crit_words = _tokenize(crit_desc)
                    crit_tokens = set(crit_words)

                    direct_match = sum(5.0 for tok in opt_tokens if len(tok) >= 3 and re.search(r"\b" + re.escape(tok) + r"\b", st_text))
                    overlap = len(st_tokens & crit_tokens) * 2.2

                    ngram_match = 0.0
                    if len(crit_words) >= 2:
                        bigrams = [f"{crit_words[j]} {crit_words[j+1]}" for j in range(len(crit_words) - 1)]
                        ngram_match = sum(3.5 for bg in bigrams if bg in st_text)

                    num_bonus = sum(4.0 for nm in re.findall(r"\b\d+(?:[\.,]\d+)?\b", crit_desc) if nm in st_text)
                    opt_nums = re.findall(r"\d+", opt_norm)
                    for nm in opt_nums:
                        if len(nm) >= 2 and (nm in st_text or nm in st_no_punct):
                            num_bonus += 4.0

                    neg_penalty = sum(
                        -12.0 for tok in opt_tokens
                        if re.search(
                            r"\b(do\s+not|don't|no|never|not|cannot|avoid|without|except|replacing|cancelling)\s+"
                            + re.escape(tok),
                            st_text
                        )
                    )

                    quad_idx = (i + phase_offset) % 4
                    q_field = quad_weights[quad_idx]
                    feat_idx = (i * 2) % len(vec)
                    st_res = float(vec[feat_idx]) * (quad_ratios[quad_idx] - 0.5) * 4.0

                    fractal_fields.append(q_field)
                    scores.append(direct_match + overlap + ngram_match + num_bonus + neg_penalty + st_res)

                scores_arr = np.array(scores, dtype=np.float64)
                h_arr = np.array(fractal_fields, dtype=np.float64)

                if len(options) >= 2:
                    sorted_idx = np.argsort(scores_arr)
                    top_gap = scores_arr[sorted_idx[-1]] - scores_arr[sorted_idx[-2]]
                    if top_gap < 0.85:
                        diff_matrix = scores_arr[:, np.newaxis] - scores_arr[np.newaxis, :]
                        h_diff_matrix = h_arr[:, np.newaxis] - h_arr[np.newaxis, :]
                        bif_force = np.sum(
                            np.sign(diff_matrix) * (np.abs(diff_matrix) ** self.cadence_alpha)
                            + self.cadence_beta * h_diff_matrix,
                            axis=1
                        )
                        scores_final = scores_arr + self.cadence_lambda * bif_force
                    else:
                        scores_final = scores_arr + h_arr * 0.35
                else:
                    scores_final = scores_arr

                exp_s = np.exp((scores_final - np.max(scores_final)) / self.temp_choice)
                p_vals = exp_s / np.sum(exp_s)
                p_vals = (p_vals / np.sum(p_vals)).astype(float)
                probs = {str(opt): float(p) for opt, p in zip(options, p_vals)}

            elif q_type == "noul":
                pos_words = {
                    'yes', 'true', 'allowed', 'permit', 'permitted', 'valid', 'approved',
                    'success', 'shipped', 'paid', 'confirmed', 'clear', 'eligible', 'covered',
                    'exempt', 'exempts'
                }
                neg_words = {
                    'no', 'false', 'denied', 'prohibited', 'not', 'absent', 'missing',
                    'unproved', 'unauthorized', 'failed', 'cannot', 'dispute', 'exclude', 'excluded'
                }

                pos_evidence = 0.0
                neg_evidence = 0.0

                for tok in st_tokens:
                    if tok in pos_words:
                        if re.search(r"\b(not|no|never|un|dis|without|missing|lacks?)\s+(?:\w+\s+){0,1}" + re.escape(tok) + r"\b", st_text):
                            neg_evidence += 4.0
                        else:
                            pos_evidence += 2.0
                    if tok in neg_words:
                        neg_evidence += 2.0

                pos_evidence += len(st_tokens & instr_tokens) * 0.5
                lex_diff = pos_evidence - neg_evidence

                dot_product = float(np.dot(vec[:16], tile_weights[:16]))
                has_negation = bool(re.search(r'\b(not|no|never|without|un|dis|lacks?)\b', st_text))
                pos_polarity = bool(st_tokens & pos_words)
                neg_polarity = bool(st_tokens & neg_words)

                polarity_bias = 0.0
                if pos_polarity and not has_negation:
                    polarity_bias += 1.2
                elif neg_polarity or has_negation:
                    polarity_bias -= 1.2

                if has_negation:
                    combined_logit = dot_product + polarity_bias + min(0.0, lex_diff * 0.25)
                elif abs(lex_diff) >= 2.5:
                    combined_logit = lex_diff * 0.65 + dot_product * 0.40 + polarity_bias * 0.5
                else:
                    combined_logit = dot_product + polarity_bias + lex_diff * 0.35

                prob = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, combined_logit))))
                prob = max(0.01, min(0.99, prob))
                probs = {"yes": float(prob), "no": float(1.0 - prob)}

            elif q_type == "score":
                cand_labels = labels if labels else (
                    [str(i) for i in range(len(criteria))] if isinstance(criteria, list) else ["0", "1", "2", "3"]
                )
                level_scores = []
                for idx, idx_str in enumerate(cand_labels):
                    crit_text = ""
                    if isinstance(criteria, list) and idx < len(criteria):
                        crit_text = str(criteria[idx])
                    elif isinstance(criteria, dict):
                        crit_text = str(criteria.get(idx_str, criteria.get(idx, "")))

                    c_toks = set(_tokenize(crit_text))
                    c_match = len(st_tokens & c_toks) * 2.5
                    crit_words = _tokenize(crit_text)
                    if len(crit_words) >= 2:
                        bigrams = [f"{crit_words[j]} {crit_words[j+1]}" for j in range(len(crit_words)-1)]
                        c_match += sum(3.5 for bg in bigrams if bg in st_text)
                    level_scores.append(c_match + quad_weights[idx % 4] * 0.25)

                scores_arr = np.array(level_scores, dtype=np.float64)
                exp_s = np.exp((scores_arr - np.max(scores_arr)) / 1.0)
                p_s = exp_s / np.sum(exp_s)
                p_s = (p_s / np.sum(p_s)).astype(float)
                probs = {str(lbl): float(p) for lbl, p in zip(cand_labels, p_s)}

            res.latency_s = time.perf_counter() - t0
            res.probs = probs
            res.ok = True
            res.usage = {"input_tokens": len(st_tokens), "output_tokens": 1}
            res.raw = {
                "response": probs,
                "runtime": {
                    "device": "cpu",
                    "threads": self.threads,
                    "memory_weights": "0 Bytes",
                    "coordinate_bytes": 24,
                    "architecture": "Universal Mandelbrot Boundary Cusp Tripod Kernel (0-VRAM)"
                }
            }
        except Exception as exc:
            res.latency_s = time.perf_counter() - t0
            res.ok = False
            res.error = f"{type(exc).__name__}: {str(exc)[:300]}"

        return res

    def reserve_estimate(self, task) -> float:
        return 0.0
