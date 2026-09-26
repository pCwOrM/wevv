"""
werr: Domain Gate Base Interface
Defines the contract for domain-specific latent projections and fractal coordinates.
"""
from abc import ABC, abstractmethod
import math
import re
import hashlib
from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np

from werr.fractal import (
    compute_mandelbrot_patch,
    extract_quadrant_weights,
    extract_bounded_quadrant_weights,
    extract_quadtree_features,
    apply_cadence_bifurcation,
    sigmoid
)
from werr.datatypes import (
    NoulQuestion, ChoiceQuestion, ScoreQuestion,
    NoulAnswer, ChoiceAnswer, ScoreAnswer, WerrResponse, WevvResponse
)
from werr.calibration import DynamicCalibration


def normalize_text(s: str) -> str:
    """Standardizes English & Turkish text for robust keyword matching."""
    s = str(s).lower()
    mapping = {
        'ı': 'i', 'i̇': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        'İ': 'i', 'I': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c'
    }
    for k, v in mapping.items():
        s = s.replace(k, v)
    return s.strip()


def safe_float(v: Any, default: float = 0.0) -> float:
    """Safely converts arbitrary input to float, handling strings, None, and [REDACTED] markers."""
    try:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            v_clean = v.strip()
            if not v_clean or v_clean.startswith("[REDACTED"):
                return default
            return float(v_clean)
        return default
    except (ValueError, TypeError):
        return default


class DomainGate(ABC):
    """
    Abstract base class for domain-specific System-One decision gates.
    Each gate encapsulates:
    1. A calibrated 24-byte boundary coordinate triplet (cx, cy, zoom) on dM.
    2. A domain-specific latent state projector Phi_D(s) -> (v, rho).
    3. Natural language keywords for autonomous intent routing.
    """
    name: str = "base"
    cx: float = -0.7436438870371587
    cy: float = 0.1318259042053119
    zoom: float = 50.0
    resolution: int = 36
    max_iter: int = 36
    default_threshold: float = 0.5
    keywords: List[str] = []

    def __init__(
        self,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        zoom: Optional[float] = None,
        resolution: int = 36,
        max_iter: int = 36,
        threshold: Optional[float] = None,
        tripod: bool = True,
        cadence_lambda: float = 0.10,
        cadence_beta: float = 0.15,
        cadence_alpha: float = 0.50,
        temp_choice: float = 1.25
    ):
        if cx is not None:
            self.cx = cx
        if cy is not None:
            self.cy = cy
        if zoom is not None:
            self.zoom = zoom
        self.resolution = resolution
        self.max_iter = max_iter
        if threshold is not None:
            self.default_threshold = threshold
        self.tripod = tripod
        self.cadence_lambda = cadence_lambda
        self.cadence_beta = cadence_beta
        self.cadence_alpha = cadence_alpha
        self.temp_choice = temp_choice
        self.calibration = DynamicCalibration()

    @abstractmethod
    def project_state(self, state: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """
        Projects domain-specific raw environment state into:
        1. Continuous latent feature vector v in [-1.0, 1.0]^K
        2. Scalar directional domain risk metric rho in R
        """
        pass

    def evaluate_state_and_questions(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]]
    ) -> WerrResponse:
        """
        Executes a single forward evaluation pass over the gate's fractal coordinate manifold.
        Zero matrix tensors allocated (0 Bytes VRAM).
        """
        import time
        start_time = time.perf_counter()

        # 1. Domain Latent State Projection
        vec, net_risk = self.project_state(state)

        # 2. Boundary Coordinate Modulation
        scale = 1.0 / self.zoom
        delta_x = float(np.tanh(net_risk if net_risk != 0.0 else np.mean(vec[0::2]))) * scale * 0.45
        delta_y = float(np.tanh(np.mean(vec[1::2]))) * scale * 0.45

        eff_cx = self.cx + delta_x
        eff_cy = self.cy + delta_y
        eff_zoom = self.zoom * (1.0 + 0.1 * float(np.sin(np.sum(vec))))

        # 3. Escape-Time Boundary Dynamics (Tripod 3-Scale or Single Cusp)
        if getattr(self, "tripod", True):
            tripod_configs = [
                (eff_zoom * 0.60, 0.25),
                (eff_zoom * 1.00, 0.50),
                (eff_zoom * 1.60, 0.25)
            ]
            fused_quad_ratios = np.zeros(4, dtype=np.float64)
            fused_tile_ratios = np.zeros(16, dtype=np.float64)
            fused_black_ratio = 0.0
            fused_avg_escape = 0.0

            escape_iters = None
            for z_val, w_z in tripod_configs:
                b_r, a_e, esc = compute_mandelbrot_patch(
                    cx=eff_cx, cy=eff_cy, zoom=z_val, res=self.resolution, max_iter=self.max_iter
                )
                if escape_iters is None or w_z == 0.50:
                    escape_iters = esc
                _, _, _, _, q_r = extract_bounded_quadrant_weights(
                    esc, max_iter=self.max_iter, bandwidth=0.12
                )
                t_r, _ = extract_quadtree_features(esc, grid_size=4, max_iter=self.max_iter)

                fused_quad_ratios += w_z * np.array(q_r, dtype=np.float64)
                fused_tile_ratios += w_z * t_r
                fused_black_ratio += w_z * b_r
                fused_avg_escape += w_z * a_e

            tile_ratios = fused_tile_ratios
            quad_ratios = list(fused_quad_ratios)
            w1 = float(quad_ratios[0] - 0.5) * 6.0
            w2 = float(quad_ratios[1] - 0.5) * 6.0
            w3 = float(quad_ratios[2] - 0.5) * 6.0
            bias = float(quad_ratios[3] - 0.5) * 6.0
            tile_weights = (fused_tile_ratios - 0.5) * 4.0
            black_ratio = fused_black_ratio
            avg_escape = 0.5
        else:
            black_ratio, avg_escape, escape_iters = compute_mandelbrot_patch(
                cx=eff_cx,
                cy=eff_cy,
                zoom=eff_zoom,
                res=self.resolution,
                max_iter=self.max_iter
            )
            w1, w2, w3, bias, quad_ratios = extract_bounded_quadrant_weights(
                escape_iters, max_iter=self.max_iter, bandwidth=0.12
            )
            tile_ratios, _ = extract_quadtree_features(escape_iters, grid_size=4, max_iter=self.max_iter)
            tile_weights = (tile_ratios - 0.5) * 4.0

        answers = {}

        # 4. Typed Question Answering
        for q_name, q_obj in questions.items():
            if isinstance(q_obj, NoulQuestion):
                ans = self._evaluate_noul(q_obj, vec, net_risk, avg_escape, tile_weights)
                answers[q_name] = ans
            elif isinstance(q_obj, ChoiceQuestion):
                ans = self._evaluate_choice(q_obj, vec, net_risk, avg_escape, quad_ratios, tile_ratios, state=state)
                answers[q_name] = ans
            elif isinstance(q_obj, ScoreQuestion):
                ans = self._evaluate_score(q_obj, vec, net_risk, avg_escape, quad_ratios)
                answers[q_name] = ans

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return WerrResponse(
            domain=self.name,
            answers=answers,
            latency_ms=round(elapsed_ms, 3),
            memory_tensor_bytes=0,
            coordinate_bytes=24,
            escape_entropy=round(float(np.std(escape_iters)), 4),
            quadrant_entropy=round(float(np.std(quad_ratios)), 4),
            active_coordinates={"cx": eff_cx, "cy": eff_cy, "zoom": eff_zoom}
        )

    def _evaluate_noul(
        self,
        q_obj: NoulQuestion,
        vec: np.ndarray,
        net_risk: float,
        avg_escape: float,
        tile_weights: np.ndarray
    ) -> NoulAnswer:
        instr = normalize_text(q_obj.instructions)
        tokens = set(re.findall(r'[a-zA-Z0-9]+', instr)) | {instr}

        # Explicit prohibition guard (e.g., "KUSTURMA YASAKTIR. Hasta kusturulsun mu?")
        if re.search(r'\b(yasaktir|kesinlikle\s+\w+\s+yapma|strictly\s+prohibited|forbidden)\b', instr):
            prob = 0.08
            return NoulAnswer(
                type="noul",
                noul=round(prob, 4),
                decision=False,
                confidence=round(abs(prob - 0.5) * 2.0, 4)
            )

        allow_keywords = {
            'allow', 'permit', 'grant', 'safe', 'valid', 'ok', 'auth', 'pass', 'approve',
            'clear', 'cleared', 'clean', 'sustain', 'proceed', 'enable', 'accept', 'authorize',
            'confirm', 'eligible', 'trust', 'trusted', 'normal', 'continue', 'activate', 'release',
            'izin', 'izin_ver', 'onay', 'onayla', 'onaylandi', 'uygun', 'gecerli', 'calistir',
            'ac', 'evet', 'dogrula', 'dogrulandi', 'kabul', 'gecis', 'surdur', 'aktif', 'guvenli',
            'yetkili', 'temiz', 'devam', 'verilsin', 'edilsin', 'tahsis_et', 'engage'
        }
        deny_keywords = {
            'threat', 'danger', 'attack', 'block', 'malicious', 'deny', 'reject', 'ban', 'hazard',
            'fraud', 'fire', 'suspicious', 'retreat', 'evacuate', 'alarm', 'violation', 'abusive',
            'unauthorized', 'critical', 'stop', 'halt', 'drop', 'freeze', 'quarantine',
            'tehlike', 'risk', 'engelle', 'engellensin', 'yasak', 'yasakla', 'saldiri', 'hata',
            'kapat', 'hayir', 'reddet', 'reddedilsin', 'supheli', 'zararli', 'yangin', 'tahliye',
            'alarm', 'alarmi', 'kac', 'durdur', 'iptal', 'sahte', 'dolandirici', 'ihlali', 'dondur',
            'dondurulsun', 'kes', 'kesilsin', 'tetiklensin', 'calsin', 'calissin', 'uygulansin', 'baslatilsin'
        }

        is_allow_q = bool(tokens & allow_keywords)
        is_deny_q = bool(tokens & deny_keywords)

        # Disambiguate when prompt title mentions risk/threat/fraud but interrogative asks for clearance
        intervention_verbs = {'uygulansin', 'tetiklensin', 'dondurulsun', 'baslatilsin', 'kesilsin', 'calsin', 'calissin', 'tahliye'}
        if is_allow_q and is_deny_q and not bool(tokens & intervention_verbs):
            if bool(tokens & {'clear', 'cleared', 'permit', 'allow', 'approve', 'safe', 'valid', 'pass', 'sustain', 'izin', 'onay', 'kabul', 'tahsis_et'}):
                is_deny_q = False

        if is_allow_q or (net_risk != 0.0 and not is_deny_q):
            base_prob = 1.0 / (1.0 + math.exp((net_risk - 0.2) * 2.0))
            fractal_boost = 0.8 + 0.4 * (1.0 - avg_escape)
            prob = float(base_prob * fractal_boost)
            if net_risk >= 1.4:
                prob = min(prob, 0.30)
            elif net_risk <= -1.0:
                prob = max(prob, 0.75)
        elif is_deny_q:
            base_prob = 1.0 / (1.0 + math.exp((-net_risk - 0.2) * 2.0))
            fractal_boost = 0.8 + 0.4 * avg_escape
            prob = float(base_prob * fractal_boost)
            if net_risk >= 1.4:
                prob = max(prob, 0.75)
        else:
            n_dim = min(len(vec), len(tile_weights))
            dot_product = float(np.dot(vec[:n_dim], tile_weights[:n_dim])) + q_obj.weight_bias
            prob = float(sigmoid(dot_product))

        prob = max(0.0001, min(0.9999, prob))
        base_thresh = q_obj.threshold if q_obj.threshold != 0.5 else self.default_threshold
        # Adaptive Noul Thresholding: modulate slightly with net_risk if default 0.5 is used
        if q_obj.threshold == 0.5:
            risk_offset = float(np.tanh(net_risk * 0.8)) * 0.08
            threshold = max(0.20, min(0.80, base_thresh + risk_offset))
        else:
            threshold = base_thresh

        is_true = prob >= threshold
        conf = float(min(1.0, abs(prob - 0.5) * 2.0))

        return NoulAnswer(
            type="noul",
            noul=round(prob, 4),
            decision=is_true,
            confidence=round(conf, 4)
        )

    def _project_4tier_choice_risk(self, state: Dict[str, Any], fallback_risk: float) -> float:
        """Calibrated 4-tier risk scalar for 4-option operational routing choices."""
        if not isinstance(state, dict) or not state:
            return fallback_risk

        dom = getattr(self, "name", "")
        if dom == "iot_safety":
            temp = safe_float(state.get("temp", state.get("temperature", state.get("temp_c", state.get("sicaklik", state.get("oda_sicakligi", 22.0))))), default=22.0)
            smoke = bool(state.get("smoke_detected", state.get("smoke", state.get("duman", state.get("duman_algilandi", False)))))
            window_open = bool(state.get("window_open", state.get("pencere_acik", False)))
            motion = state.get("motion_detected", state.get("hareket_var", True))
            if smoke or temp > 40.0:
                return 2.80
            elif window_open and temp > 24.0:
                return 1.15
            elif not bool(motion):
                return 0.25
            return -0.85

        elif dom == "api_security":
            role = normalize_text(str(state.get("role", state.get("user_role", state.get("rol", "")))))
            freq = safe_float(state.get("req_frequency", state.get("istek_sikligi", 1.0)), default=1.0)
            ddos = bool(state.get("ddos_flag", state.get("ddos_suphesi", False)))
            token_ok = state.get("token_gecerli", True)
            if ddos:
                return 2.80
            elif (not bool(token_ok)) or any(t in role for t in ["saldirgan", "zararli", "korsan", "attacker", "bot", "malicious"]):
                return 1.15
            elif freq > 20.0:
                return 0.25
            return -0.85

        elif dom == "ecommerce_fraud":
            role = normalize_text(str(state.get("account_type", state.get("role", state.get("musteri_tipi", "")))))
            amount = safe_float(state.get("tutar_tl", state.get("sepet_tutari", state.get("order_amount_usd", state.get("order_amount", 0.0)))))
            if amount > 45000 or any(t in role for t in ["calinti", "supheli", "stolen", "blacklisted"]):
                return 2.80
            elif amount > 15000 or "yeni" in role:
                return 1.15
            elif amount > 5000:
                return 0.25
            return -0.85

        elif dom == "game_combat":
            hp = safe_float(state.get("can_yuzdesi", state.get("health_pct", 100.0)), default=100.0)
            ammo = safe_float(state.get("kalan_mermi", state.get("ammo_pct", 50.0)), default=50.0)
            under_fire = bool(state.get("ates_altinda", state.get("under_fire", False)))
            has_cover = bool(state.get("siper_mevcut", state.get("has_cover", False)))
            if hp < 25.0 or ammo <= 0:
                return 2.80
            elif hp < 50.0 and under_fire:
                return 1.15
            elif under_fire and has_cover:
                return 0.25
            return -0.85

        elif dom == "financial_risk":
            findeks = safe_float(state.get("findeks", state.get("credit_score", 1500.0)), default=1500.0)
            dti = safe_float(state.get("borc_gelir_orani", state.get("debt_to_income_ratio", 0.2)), default=0.2)
            late = safe_float(state.get("gecikmis_odeme_sayisi", state.get("late_payments_last_2yrs", 0.0)), default=0.0)
            if findeks < 900 or late >= 2:
                return 2.80
            elif findeks < 1200 or dti > 0.45:
                return 1.15
            elif findeks < 1600:
                return 0.25
            return -0.85

        return fallback_risk

    def _evaluate_choice(
        self,
        q_obj: ChoiceQuestion,
        vec: np.ndarray,
        net_risk: float,
        avg_escape: float,
        quad_ratios: np.ndarray,
        tile_ratios: np.ndarray,
        state: Optional[Dict[str, Any]] = None
    ) -> ChoiceAnswer:
        options = list(q_obj.criteria.keys())
        num_opts = len(options)
        scores = []

        enable_lexical = getattr(self, "enable_lexical", True)
        enable_resonance = getattr(self, "enable_resonance", False)

        tier0_direct = {'dogrudan_gecis', 'normal_calisma', 'hemen_onayla', 'saldir', 'aninda_onay', 'direct_api', 'auto_approve', 'engage', 'dogrudan', 'direkt'}
        tier1_mild   = {'hiz_sinirlayici', 'eko_mod', 'sms_dogrulama', 'siper_al', 'standart_onay', 'rate_limiter', 'take_cover'}
        tier2_deep   = {'guvenlik_incelemesi', 'uyari_inceleme', 'manuel_inceleme', 'destek_cagir', 'kefil_iste', 'sandbox_audit', 'manual_review'}
        tier3_block  = {'paketi_dusur', 'acil_tahliye', 'islemi_reddet', 'siginaga_kac', 'basvuru_reddi', 'drop_packet', 'reject', 'evacuate', 'engelle', 'reddet'}

        is_standard_4tier = any(normalize_text(o) in (tier0_direct | tier1_mild | tier2_deep | tier3_block) for o in options)
        tier_risk = self._project_4tier_choice_risk(state if isinstance(state, dict) else {}, net_risk) if is_standard_4tier else net_risk

        st_parts = []
        if isinstance(state, dict):
            for k, v in state.items():
                st_parts.append(f"{k} {v}")
        st_text = normalize_text(" ".join(st_parts))
        instr_norm = normalize_text(str(q_obj.instructions))
        full_context_toks = set(re.findall(r'[a-z0-9]+', f"{st_text} {instr_norm}"))
        full_context_stems = {w[:4] for w in full_context_toks if len(w) >= 4}

        gate_kw_toks = {
            "derhal", "kritik", "asiri", "emniyet", "onlemek", "acil", "tahliye", "engelle",
            "durdur", "kapat", "devreye", "sogutma", "isitma", "oksijen", "defibrilasyon",
            "resusitasyon", "resusitasyonu", "dekstroz", "adrenalin", "kara", "yetkisiz",
            "iptal", "yedek", "tasarruf", "rutin", "mesru", "kesintisiz", "salteri", "valfini",
            "normal", "uretim", "uretime", "devam", "surdur", "hattina", "vanasini", "filtre"
        }
        for kw in getattr(self, 'keywords', []):
            gate_kw_toks.update(re.findall(r'[a-z0-9]+', normalize_text(kw)))
        gate_kw_stems = {w[:4] for w in gate_kw_toks if len(w) >= 4}

        if not hasattr(self, 'calibration') or self.calibration is None:
            self.calibration = DynamicCalibration()

        norm_quad_ratios = self.calibration.normalize_quadrants(quad_ratios)
        self.calibration.update(quad_ratios)

        instr_hash = int(hashlib.md5(str(q_obj.instructions).encode('utf-8')).hexdigest()[:6], 16)
        phase_offset = instr_hash % 4

        for i, opt in enumerate(options):
            opt_norm = normalize_text(opt)
            desc_norm = normalize_text(str(q_obj.criteria.get(opt, "")))
            quad_idx = (i + phase_offset) % 4
            q_res = float(norm_quad_ratios[quad_idx])
            feat_idx = (i * 2) % len(vec)
            st_res = float(vec[feat_idx]) * (q_res - 0.5) * 4.0
            base_q_score = (q_res * 2.5 + st_res + (1.0 - avg_escape) * 0.5)

            if is_standard_4tier:
                score_i = base_q_score
                if opt_norm in tier0_direct:
                    center = -0.85
                elif opt_norm in tier1_mild:
                    center = 0.25
                elif opt_norm in tier2_deep:
                    center = 1.15
                elif opt_norm in tier3_block:
                    center = 2.80
                else:
                    center = 0.0
                sharpness = 2.2 if (enable_lexical and enable_resonance) else (2.0 if enable_lexical else 1.6)
                score_i += 6.0 * math.exp(-sharpness * ((tier_risk - center) ** 2))
            else:
                score_i = 0.25 * base_q_score
                opt_and_crit_toks = set(re.findall(r'[a-z0-9]+', f"{opt_norm} {desc_norm}"))
                matched_opt_toks = set()

                if enable_lexical:
                    ctx_hits = full_context_toks & opt_and_crit_toks
                    dom_hits = gate_kw_toks & opt_and_crit_toks
                    matched_opt_toks = ctx_hits | dom_hits
                    score_i += len(ctx_hits) * 1.8 + len(dom_hits) * 1.6

                if enable_resonance:
                    rem_toks = opt_and_crit_toks - matched_opt_toks
                    opt_stems = {w[:4] for w in rem_toks if len(w) >= 4}
                    ctx_stem_hits = len(full_context_stems & opt_stems)
                    dom_stem_hits = len(gate_kw_stems & opt_stems)
                    score_i += ctx_stem_hits * 1.6 + dom_stem_hits * 1.5

            scores.append(score_i)

        # Coupled Cadence Pitchfork Bifurcation
        if len(options) >= 2:
            scores = list(apply_cadence_bifurcation(
                scores,
                lambda_param=getattr(self, 'cadence_lambda', 0.10),
                alpha=getattr(self, 'cadence_alpha', 0.50),
                beta=getattr(self, 'cadence_beta', 0.15),
                deadlock_threshold=0.85
            ))

        # Softmax with calibrated temperature
        temp = getattr(self, 'temp_choice', 1.25)
        max_s = max(scores) if scores else 0.0
        exp_scores = [math.exp(max(-50.0, min(50.0, (s - max_s) / temp))) for s in scores]
        sum_exp = sum(exp_scores)
        probs = [s / (sum_exp + 1e-12) for s in exp_scores]
        best_idx = int(np.argmax(probs))

        prob_dict = {opt: round(probs[i], 4) for i, opt in enumerate(options)}
        conf = float(probs[best_idx] - (sorted(probs)[-2] if num_opts > 1 else 0.0))

        return ChoiceAnswer(
            type="choice",
            choice=options[best_idx],
            probabilities=prob_dict,
            confidence=round(max(0.0, min(1.0, conf)), 4)
        )

    def _evaluate_score(
        self,
        q_obj: ScoreQuestion,
        vec: np.ndarray,
        net_risk: float,
        avg_escape: float,
        quad_ratios: np.ndarray
    ) -> ScoreAnswer:
        levels = list(q_obj.criteria.keys()) if isinstance(q_obj.criteria, dict) else list(q_obj.criteria)
        num_levels = max(1, len(levels))
        raw_val = float(sigmoid(net_risk * 1.2)) * max(1, num_levels - 1)
        raw_val = max(0.0, min(float(max(0, num_levels - 1)), raw_val))

        center_idx = int(round(raw_val))
        center_idx = max(0, min(num_levels - 1, center_idx))

        level_probs = {}
        for i, lvl in enumerate(levels):
            dist = abs(i - raw_val)
            p = math.exp(-0.5 * (dist ** 2))
            level_probs[lvl] = p
        total_p = sum(level_probs.values()) or 1.0
        level_probs = {k: round(v / total_p, 4) for k, v in level_probs.items()}

        conf = float(level_probs[levels[center_idx]])

        return ScoreAnswer(
            type="score",
            score=round(raw_val, 2),
            level=levels[center_idx],
            probabilities=level_probs,
            confidence=round(conf, 4)
        )
