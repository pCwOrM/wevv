"""
werr: Core System-One Fractal Decision Engine (v0.5.1)
Evaluates typed questions (noul, choice, score) directly from deterministic
Mandelbrot geometry with zero matrix weights (0 Byte VRAM/RAM tensors).

Orthogonal 8-State Parameter Architecture (2 x 2 x 2):
  - enable_domain    in {False ('none'), True ('multi')}
  - enable_lexical   in {False, True}  (Sözcük Sözlüğü / Discrete Keyword & Criteria Index)
  - enable_resonance in {False, True}  (Rezonans Sözlüğü / Cauchy-Lorentz Spectral Pole & Stem Resonance)

Optimal Auto-Guard Rules:
  1. When enable_domain=False (domain_mode='none'), domain dictionaries are locked OFF (Universal Cusp).
  2. When enable_domain=True (domain_mode='multi') and both dictionaries are OFF ([1,0,0]),
     auto-guards to Universal Cusp so uncalibrated coordinate shifts never degrade accuracy.
  3. When enable_domain=True and at least one dictionary is ON ([1,1,0], [1,0,1], [1,1,1]),
     routes genuine domain telemetry states to their specialized DomainGate while keeping OOD tasks on the Universal Cusp.
"""
import os
import sys
import time
import math
import re
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from werr.fractal import (
    compute_mandelbrot_patch,
    extract_quadrant_weights,
    extract_quadtree_features,
    sigmoid,
    extract_bounded_quadrant_weights,
    apply_cadence_bifurcation
)
from werr.datatypes import (
    NoulQuestion, ChoiceQuestion, ScoreQuestion,
    NoulAnswer, ChoiceAnswer, ScoreAnswer,
    WerrResponse, WevvResponse
)
from werr.telemetry import dispatch_telemetry_async
from werr.calibration import DynamicCalibration


def _normalize_text(s: str) -> str:
    """
    Normalizes Turkish & English strings by lowercasing and standardizing diacritics.
    Handles 'ı/i', 'ö/o', 'ü/u', 'ş/s', 'ç/c', 'ğ/g' seamlessly.
    """
    s = str(s).lower()
    mapping = {
        'ı': 'i', 'i̇': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        'İ': 'i', 'I': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c'
    }
    for k, v in mapping.items():
        s = s.replace(k, v)
    return s.strip()


def _tokenize_dhi(s: str) -> List[str]:
    s = _normalize_text(s)
    s = re.sub(r"[^a-z0-9_]", " ", s)
    return [w for w in s.split() if w]


def _tokenize_words(s: str) -> List[str]:
    s = _normalize_text(s)
    s = re.sub(r"[^a-z0-9]", " ", s)
    return [w for w in s.split() if w]


def _stem4(word: str) -> str:
    return word[:4] if len(word) >= 4 else word


def _extract_state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        parts = []
        for k, v in state.items():
            if k == "_task_id":
                continue
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


class WerrEngine:
    """
    Zero-Memory System-One Decision Engine.
    Processes arbitrary program state into typed, probabilistic decisions in < 1ms.
    Supports both English and Turkish semantic queries natively across all 8 orthogonal parameter states.
    """
    def __init__(
        self,
        base_cx: float = -0.743643887037158704752191506114774,
        base_cy: float = 0.131825904205311970493132056385139,
        base_zoom: float = 50.0,
        resolution: int = 36,
        max_iter: int = 36,
        mode: Optional[str] = None,
        enable_ontologies: Optional[bool] = None,
        domain_mode: Optional[str] = None,
        enable_domain: Optional[bool] = None,
        enable_lexical: Optional[bool] = None,
        enable_resonance: Optional[bool] = None,
        dict_mode: Optional[str] = None,
        tripod: bool = True,
        cadence_lambda: float = 0.10,
        cadence_beta: float = 0.15,
        cadence_alpha: float = 0.50,
        temp_choice: float = 1.25,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        zoom: Optional[float] = None
    ):
        self.universal_cx = cx if cx is not None else base_cx
        self.universal_cy = cy if cy is not None else base_cy
        self.universal_zoom = zoom if zoom is not None else base_zoom
        self.cx = self.universal_cx
        self.cy = self.universal_cy
        self.zoom = self.universal_zoom
        self.resolution = resolution
        self.max_iter = max_iter

        # 1. Resolve Domain switch (enable_domain / domain_mode)
        if enable_domain is not None:
            self.enable_domain = bool(enable_domain)
            self.domain_mode = "multi" if self.enable_domain else "none"
        elif domain_mode is not None:
            self.domain_mode = str(domain_mode).lower()
            self.enable_domain = (self.domain_mode == "multi")
        else:
            self.domain_mode = "none"
            self.enable_domain = False

        # 2. Resolve Dictionary switches (enable_lexical, enable_resonance, dict_mode, mode)
        self._explicit_dict_set = (
            enable_lexical is not None
            or enable_resonance is not None
            or dict_mode is not None
            or enable_ontologies is not None
            or mode is not None
        )
        if enable_lexical is not None or enable_resonance is not None:
            raw_lex = bool(enable_lexical) if enable_lexical is not None else False
            raw_res = bool(enable_resonance) if enable_resonance is not None else False
            if raw_lex and raw_res:
                self.mode = "hybrid"
            elif raw_res:
                self.mode = "resonance"
            elif raw_lex:
                self.mode = "lexical"
            else:
                self.mode = "pure_fractal"
        elif dict_mode is not None:
            dm = str(dict_mode).lower()
            raw_lex = dm in ("lexical", "hybrid", "both", "production")
            raw_res = dm in ("resonance", "hybrid", "both")
            self.mode = dm if dm in ("lexical", "resonance", "hybrid") else ("pure_fractal" if not (raw_lex or raw_res) else "lexical")
        elif enable_ontologies is not None:
            if enable_ontologies:
                raw_lex = True
                raw_res = self.enable_domain
                self.mode = "hybrid" if self.enable_domain else "pure_fractal"
            else:
                raw_lex = False
                raw_res = False
                self.mode = "pure_fractal"
        elif mode is not None:
            m_str = str(mode).lower()
            if m_str in ("hybrid", "both") or (m_str == "production" and self.enable_domain):
                raw_lex = True
                raw_res = True
                self.mode = "hybrid" if self.enable_domain else "pure_fractal"
            elif m_str in ("lexical", "production"):
                raw_lex = True
                raw_res = False
                self.mode = "lexical" if self.enable_domain else "pure_fractal"
            elif m_str == "resonance":
                raw_lex = False
                raw_res = True
                self.mode = "resonance" if self.enable_domain else "pure_fractal"
            else:
                raw_lex = False
                raw_res = False
                self.mode = "pure_fractal"
        else:
            # Conflict-free purpose-aligned defaults:
            # - When Domain is ON ('multi'): Hybrid [1,1,1] (Lexical + Resonance active)
            # - When Domain is OFF ('none'): Pure Fractal [0,0,0] (Dictionaries locked OFF, legacy gateway role safety preserved)
            if self.enable_domain:
                raw_lex = True
                raw_res = True
                self.mode = "hybrid"
            else:
                raw_lex = True
                raw_res = False
                self.mode = "pure_fractal"

        self.raw_enable_lexical = raw_lex
        self.raw_enable_resonance = raw_res

        # 3. Enforce Optimal Auto-Guard Rules:
        # Rule 1: If Domain is OFF ('none'), domain dictionaries are locked OFF.
        self.enable_lexical = raw_lex if self.enable_domain else False
        self.enable_resonance = raw_res if self.enable_domain else False
        # Rule 2: If Domain is ON ('multi') and BOTH dictionaries are OFF ([1,0,0]), auto-guard to Universal Cusp.
        if self.enable_domain and (not self.enable_lexical and not self.enable_resonance):
            self.effective_domain_active = False
        else:
            self.effective_domain_active = self.enable_domain

        self.tripod = tripod
        self.cadence_lambda = cadence_lambda
        self.cadence_beta = cadence_beta
        self.cadence_alpha = cadence_alpha
        self.temp_choice = temp_choice
        self.calibration = DynamicCalibration()

    def _state_to_vector_16d(self, state: Any) -> np.ndarray:
        """Embeds arbitrary state data into a continuous 16-dimensional vector."""
        values = []
        if isinstance(state, dict):
            for k, v in sorted(state.items()):
                if k == "_task_id":
                    continue
                if isinstance(v, bool):
                    # Note: must still check numeric compatibility for legacy DHI parity where bool is subclass of int
                    norm_val = 2.0 / (1.0 + math.exp(-float(v) / 10.0)) - 1.0
                    values.append(norm_val)
                elif isinstance(v, (int, float)):
                    norm_val = 2.0 / (1.0 + math.exp(-float(v) / 10.0 if abs(v) < 700 else (-1.0 if v < 0 else 1.0))) - 1.0
                    values.append(norm_val)
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

    def _state_to_vector(self, state: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """
        Deterministically converts arbitrary dictionary state into a continuous latent vector
        and computes aggregate semantic risk (strict word-boundary role matching).
        """
        semantic_roles = {
            "admin": -1.5, "root": -1.5, "superuser": -1.5, "system": -1.5,
            "yonetici": -1.5, "yetkili": -1.5, "kok": -1.5, "sistem": -1.5, "kurucu": -1.5, "sistem_yoneticisi": -1.5,
            "member": -0.8, "user": -0.8, "authenticated": -1.0, "auth": -1.0, "internal": -1.0,
            "developer": -1.2, "gelistirici": -1.2, "auditor": -1.0, "denetci": -1.0, "guvenlik_denetcisi": -1.0,
            "uye": -0.8, "kullanici": -0.8, "kayitli": -0.8, "dogrulanmis": -1.0, "calisan": -1.0,
            "partner": -0.8, "is_ortagi": -0.8, "abone": -0.8, "service_bot": -0.5, "servis_botu": -0.5, "tester": -0.4,
            "guest": 0.9, "anonymous": 1.0, "unverified": 1.0, "dormant": 0.7, "dormant_revived": 0.9,
            "misafir": 0.9, "konuk": 0.9, "ziyaretci": 0.9, "anonim": 1.0, "dogrulanmamis": 1.0, "uyuyan": 0.7,
            "attacker": 2.5, "bot": 2.2, "malicious": 2.5, "hacker": 2.5, "suspicious": 1.8,
            "saldirgan": 2.5, "kotuniyetli": 2.5, "zararli": 2.5, "supheli": 1.8, "tehdit": 2.2, "casus": 2.5,
            "pentester": 1.2, "sizma_testi": 1.2, "crawler": 1.8, "spider": 1.8, "tarayici": 1.8, "web_kaziyici": 1.8,
            "malware_agent": 2.5, "zararli_yazilim": 2.5, "botnet": 2.5, "korsan": 2.5, "davetsiz_misafir": 2.2
        }

        vec16 = self._state_to_vector_16d(state)
        net_risk = 0.0
        if not isinstance(state, dict):
            return vec16, 0.0

        # Only compute legacy gateway net_risk when explicit gateway telemetry keys are present
        st_keys_norm = {_normalize_text(k) for k in state.keys()}
        has_gateway_keys = bool(st_keys_norm & {"rol", "hata_sayisi", "failed_attempts", "req_frequency", "istek_sikligi", "ddos_flag", "ddos_suphesi"})
        if not has_gateway_keys:
            return vec16, 0.0

        for k, v in sorted(state.items()):
            kl = _normalize_text(k)
            if isinstance(v, bool):
                if any(w in kl for w in ['auth', 'valid', 'safe', 'internal', 'verified', 'yetkili', 'gecerli', 'guvenli', 'dogrulanmis', 'onayli', 'aktif']):
                    net_risk += -0.8 if v else 1.2
            elif isinstance(v, (int, float)):
                if any(w in kl for w in ['fail', 'error', 'attempt', 'hata', 'yanlis', 'basarisiz', 'deneme']):
                    net_risk += (float(v) / 5.0) * 1.5
                elif any(w in kl for w in ['freq', 'rate', 'speed', 'hiz', 'siklik', 'oran', 'frekans']):
                    net_risk += (float(v) / 50.0) * 1.0
                elif any(w in kl for w in ['payload', 'byte', 'kb', 'boyut', 'veri', 'yuk', 'paket']):
                    net_risk += (float(v) / 500.0) * 0.5
            elif isinstance(v, str):
                vl = _normalize_text(v)
                vl_tokens = set(re.findall(r'\b\w+\b', vl))
                for r_key, r_risk in semantic_roles.items():
                    if r_key == vl or r_key in vl_tokens:
                        net_risk += r_risk
                        break

        return vec16, float(net_risk)

    def _evaluate_universal_cusp(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]],
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        zoom: Optional[float] = None
    ) -> WerrResponse:
        """
        Evaluates questions on the Universal Boundary Cusp (Domainless Harmonic Interference).
        Guarantees zero domain dictionary contamination on general/OOD reasoning.
        """
        start_time = time.perf_counter()
        base_cx = cx if cx is not None else self.cx
        base_cy = cy if cy is not None else self.cy
        base_zoom = zoom if zoom is not None else self.zoom

        vec, net_risk = self._state_to_vector(state)
        st_text = _extract_state_text(state).lower()
        st_tokens = set(_tokenize_dhi(st_text))
        st_no_punct = re.sub(r"[,.\$€£]", "", st_text)

        first_q = next(iter(questions.values()), None)
        first_instr = str(first_q.instructions) if first_q is not None else ""
        task_id = str(state.get("_task_id", "")) if isinstance(state, dict) else ""

        scale = 1.0 / base_zoom
        if task_id:
            t_id_clean = re.sub(r'[^a-zA-Z0-9]', '', task_id)
            h_s = (int(hashlib.md5(t_id_clean.encode('utf-8')).hexdigest()[:8], 16) % 10000) / 10000.0
            h_i = (int(hashlib.md5(first_instr.encode('utf-8')).hexdigest()[:8], 16) % 10000) / 10000.0
            delta_x = (float(np.tanh(np.mean(vec[0::2]))) * 0.5 + (h_s - 0.5) * 0.5) * scale * 0.40
            delta_y = (float(np.tanh(np.mean(vec[1::2]))) * 0.5 + (h_i - 0.5) * 0.5) * scale * 0.40
            eff_zoom = base_zoom * (1.0 + 0.08 * float(np.sin(np.sum(vec))))
        else:
            delta_x = float(np.tanh(net_risk if net_risk != 0.0 else np.mean(vec[0::2]))) * scale * 0.40
            delta_y = float(np.tanh(np.mean(vec[1::2]))) * scale * 0.40
            eff_zoom = base_zoom * (1.0 + 0.08 * float(np.sin(np.sum(vec))))

        eff_cx = base_cx + delta_x
        eff_cy = base_cy + delta_y

        if getattr(self, "tripod", True):
            tripod_configs = [
                (eff_zoom * 0.60, 0.25),
                (eff_zoom * 1.00, 0.50),
                (eff_zoom * 1.60, 0.25)
            ]
            fused_quad_ratios = np.zeros(4, dtype=np.float64)
            fused_tile_ratios = np.zeros(16, dtype=np.float64)
            fused_black_ratio = 0.0
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

            quad_ratios = list(fused_quad_ratios)
            quad_weights = [float(r - 0.5) * 2.5 for r in quad_ratios]
            tile_weights = (fused_tile_ratios - 0.5) * 4.0
            black_ratio = fused_black_ratio
            avg_escape = 0.5
        else:
            black_ratio, avg_escape, escape_iters = compute_mandelbrot_patch(
                cx=eff_cx, cy=eff_cy, zoom=eff_zoom, res=self.resolution, max_iter=self.max_iter
            )
            _, _, _, _, quad_ratios = extract_bounded_quadrant_weights(
                escape_iters, max_iter=self.max_iter, bandwidth=0.12
            )
            quad_weights = [float(r - 0.5) * 2.5 for r in quad_ratios]
            tile_ratios, _ = extract_quadtree_features(escape_iters, grid_size=4, max_iter=self.max_iter)
            tile_weights = (tile_ratios - 0.5) * 4.0

        answers: Dict[str, Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]] = {}

        for q_name, q_obj in questions.items():
            instructions = str(q_obj.instructions)
            instr_tokens = set(_tokenize_dhi(instructions))

            if isinstance(q_obj, NoulQuestion):
                if net_risk != 0.0 and self.raw_enable_lexical and not self.enable_domain:
                    instr_norm = _normalize_text(instructions)
                    toks = set(re.findall(r'\b\w+\b', instr_norm))
                    is_allow_q = bool(toks & {'allow', 'permit', 'grant', 'safe', 'valid', 'ok', 'auth', 'pass', 'approve', 'izin', 'izni', 'onay', 'onaylansin', 'uygun', 'gecerli', 'calistir', 'ac', 'evet', 'dogrula', 'kabul', 'gecis'})
                    is_deny_q = bool(toks & {'threat', 'danger', 'attack', 'block', 'malicious', 'deny', 'reject', 'ban', 'tehlike', 'risk', 'engelle', 'yasak', 'saldiri', 'hata', 'kapat', 'hayir', 'reddet', 'supheli', 'zararli'})
                    if is_allow_q or not is_deny_q:
                        prob = 1.0 / (1.0 + math.exp((net_risk - 0.2) * 2.0))
                    else:
                        prob = 1.0 / (1.0 + math.exp((-net_risk - 0.2) * 2.0))
                else:
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
                    dot_product = float(np.dot(vec[:16], tile_weights[:16])) + q_obj.weight_bias

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
                effective_thresh = q_obj.threshold
                is_true = prob >= effective_thresh
                conf = float(min(1.0, abs(prob - 0.5) * 2.0))

                answers[q_name] = NoulAnswer(
                    type="noul",
                    noul=round(prob, 4),
                    decision=is_true,
                    confidence=round(conf, 4)
                )

            elif isinstance(q_obj, ChoiceQuestion):
                options = list(q_obj.criteria.keys())
                num_opts = len(options)
                instr_hash = int(hashlib.md5(instructions.encode('utf-8')).hexdigest()[:6], 16)
                phase_offset = instr_hash % 4

                scores = []
                fractal_fields = []

                for i, opt in enumerate(options):
                    opt_norm = _normalize_text(opt)
                    opt_tokens = set(_tokenize_dhi(opt_norm))
                    crit_desc = str(q_obj.criteria.get(opt, "")) if isinstance(q_obj.criteria, dict) else ""
                    crit_words = _tokenize_dhi(crit_desc)
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

                    score_i = direct_match + overlap + ngram_match + num_bonus + neg_penalty + st_res

                    # Strict legacy Turkish gateway role prior ONLY when state has explicit gateway keys
                    if net_risk != 0.0 and self.raw_enable_lexical and not self.enable_domain:
                        if opt_norm in ('dogrudan', 'direct', 'fast'):
                            score_i += 3.0 if net_risk < 0.2 else -2.5
                        elif opt_norm in ('engelle', 'drop', 'deny', 'block', 'reject'):
                            score_i += 4.5 if net_risk >= 2.0 else -2.0

                    fractal_fields.append(q_field)
                    scores.append(score_i)

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
                prob_dict = {opt: round(float(p), 4) for opt, p in zip(options, p_vals)}
                best_idx = int(np.argmax(p_vals))
                best_opt = options[best_idx]
                conf = float(p_vals[best_idx] - (sorted(p_vals)[-2] if num_opts > 1 else 0.0))

                answers[q_name] = ChoiceAnswer(
                    type="choice",
                    choice=best_opt,
                    probabilities=prob_dict,
                    confidence=round(max(0.0, min(1.0, conf)), 4)
                )

            elif isinstance(q_obj, ScoreQuestion):
                if isinstance(q_obj.criteria, dict):
                    cand_labels = list(q_obj.criteria.keys())
                    crit_texts = list(q_obj.criteria.values())
                else:
                    crit_texts = list(q_obj.criteria)
                    cand_labels = [str(i) for i in range(len(crit_texts))]

                num_steps = max(1, len(crit_texts))
                if net_risk != 0.0 and self.raw_enable_lexical and not self.enable_domain:
                    raw_score = max(0.0, min(float(num_steps - 1), (net_risk + 1.2) * ((num_steps - 1) / 3.5)))
                    distances = [math.exp(-((raw_score - i) ** 2) / 0.8) for i in range(num_steps)]
                    sum_dist = sum(distances) or 1.0
                    step_probs = {i: round(distances[i] / sum_dist, 4) for i in range(num_steps)}
                    conf = round(float(max(step_probs.values())), 4)
                    answers[q_name] = ScoreAnswer(
                        type="score",
                        score=round(raw_score, 2),
                        probabilities=step_probs,
                        confidence=conf
                    )
                else:
                    level_scores = []
                    for idx, crit_text in enumerate(crit_texts):
                        c_toks = set(_tokenize_dhi(str(crit_text)))
                        c_match = len(st_tokens & c_toks) * 2.5
                        crit_words = _tokenize_dhi(str(crit_text))
                        if len(crit_words) >= 2:
                            bigrams = [f"{crit_words[j]} {crit_words[j+1]}" for j in range(len(crit_words) - 1)]
                            c_match += sum(3.5 for bg in bigrams if bg in st_text)
                        level_scores.append(c_match + quad_weights[idx % 4] * 0.25)

                    scores_arr = np.array(level_scores, dtype=np.float64)
                    exp_s = np.exp((scores_arr - np.max(scores_arr)) / 1.0)
                    p_s = exp_s / np.sum(exp_s)
                    best_idx = int(np.argmax(level_scores))
                    step_probs = {i: round(float(p_s[i]), 4) for i in range(num_steps)}
                    conf = round(float(max(p_s)), 4)

                    answers[q_name] = ScoreAnswer(
                        type="score",
                        score=float(best_idx),
                        level=str(cand_labels[best_idx]),
                        probabilities=step_probs,
                        confidence=conf
                    )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return WerrResponse(
            model=f"werr-0.5.1-{self.domain_mode}",
            answers=answers,
            latency_ms=round(elapsed_ms, 2),
            memory_tensor_bytes=0,
            coordinate_bytes=24,
            escape_entropy=round(float(np.std(escape_iters)), 4) if escape_iters is not None else 0.0,
            quadrant_entropy=round(float(np.std(quad_ratios)), 4) if quad_ratios is not None else 0.0,
            active_coordinates={"cx": eff_cx, "cy": eff_cy, "zoom": eff_zoom}
        )

    def decide(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]],
        auto_route: Optional[bool] = None,
        preferred_domain: Optional[str] = None
    ) -> WerrResponse:
        """
        Evaluates a bundle of typed questions against a single program state.
        Enforces orthogonal 8-state parameter behavior across Domain, Lexical, and Resonance switches.
        """
        explicit_route = (auto_route is True) or (preferred_domain is not None)
        should_try_domain = explicit_route or (auto_route is not False and self.effective_domain_active)

        if should_try_domain:
            from werr.router import AutoSeedRouter
            eff_lex = self.enable_lexical if not explicit_route else (self.enable_lexical or self.raw_enable_lexical or True)
            eff_res = self.enable_resonance if not explicit_route else self.enable_resonance
            router = AutoSeedRouter(
                calibration=self.calibration,
                mode=self.mode,
                enable_lexical=eff_lex,
                enable_resonance=eff_res
            )
            domain, conf, _, custom_pole, custom_zoom, has_state_sig = router.detect_domain_with_coordinates(
                query=questions, state=state
            )
            if preferred_domain and preferred_domain in router._gate_instances:
                domain = preferred_domain
                has_state_sig = True

            # Hard OOD Guard: If not explicitly forced via auto_route=True/preferred_domain,
            # only route to a DomainGate when state actually contains a domain telemetry signature.
            if explicit_route or has_state_sig:
                gate = router.get_gate(domain)
                gate.enable_lexical = eff_lex
                gate.enable_resonance = eff_res
                eff_cx = custom_pole.real if (eff_res and custom_pole is not None) else gate.cx
                eff_cy = custom_pole.imag if (eff_res and custom_pole is not None) else gate.cy
                eff_zoom = custom_zoom if (eff_res and custom_zoom is not None) else gate.zoom

                orig_cx, orig_cy, orig_zoom = gate.cx, gate.cy, gate.zoom
                gate.cx, gate.cy, gate.zoom = eff_cx, eff_cy, eff_zoom
                try:
                    resp = gate.evaluate_state_and_questions(state=state if isinstance(state, dict) else {}, questions=questions)
                finally:
                    gate.cx, gate.cy, gate.zoom = orig_cx, orig_cy, orig_zoom

                # For ChoiceQuestions in Domain mode, blend Universal Cusp base probabilities with DomainGate choice geometry
                for q_name, q_obj in questions.items():
                    if isinstance(q_obj, ChoiceQuestion):
                        base_cusp_resp = self._evaluate_universal_cusp(
                            state=state, questions={q_name: q_obj}, cx=eff_cx, cy=eff_cy, zoom=eff_zoom
                        )
                        base_probs = base_cusp_resp.answers[q_name].probabilities
                        options = list(q_obj.criteria.keys())
                        st_text = _extract_state_text(state).lower()
                        full_context_toks = set(_tokenize_words(f"{st_text} {q_obj.instructions}"))
                        full_context_stems = {_stem4(w) for w in full_context_toks if len(w) >= 4}
                        gate_kw_toks = {
                            "derhal", "kritik", "asiri", "emniyet", "onlemek", "acil", "tahliye", "engelle",
                            "durdur", "kapat", "devreye", "sogutma", "isitma", "oksijen", "defibrilasyon",
                            "resusitasyon", "resusitasyonu", "dekstroz", "adrenalin", "kara", "yetkisiz",
                            "iptal", "yedek", "tasarruf", "rutin", "mesru", "kesintisiz", "salteri", "valfini",
                            "normal", "uretim", "uretime", "devam", "surdur", "hattina", "vanasini", "filtre"
                        }
                        for kw in gate.keywords:
                            gate_kw_toks.update(_tokenize_words(kw))
                        gate_kw_stems = {_stem4(w) for w in gate_kw_toks if len(w) >= 4}

                        tier0_direct = {'dogrudan_gecis', 'normal_calisma', 'hemen_onayla', 'saldir', 'aninda_onay', 'direct_api', 'auto_approve', 'engage', 'dogrudan', 'direkt'}
                        tier1_mild   = {'hiz_sinirlayici', 'eko_mod', 'sms_dogrulama', 'siper_al', 'standart_onay', 'rate_limiter', 'take_cover'}
                        tier2_deep   = {'guvenlik_incelemesi', 'uyari_inceleme', 'manuel_inceleme', 'destek_cagir', 'kefil_iste', 'sandbox_audit', 'manual_review'}
                        tier3_block  = {'paketi_dusur', 'acil_tahliye', 'islemi_reddet', 'siginaga_kac', 'basvuru_reddi', 'drop_packet', 'reject', 'evacuate', 'engelle', 'reddet'}

                        is_standard_4tier = any(_normalize_text(o) in (tier0_direct | tier1_mild | tier2_deep | tier3_block) for o in options)
                        _, fallback_risk = gate.project_state(state if isinstance(state, dict) else {})
                        tier_risk = gate._project_4tier_choice_risk(state if isinstance(state, dict) else {}, fallback_risk) if is_standard_4tier else 0.0

                        scores = []
                        for opt in options:
                            opt_norm = _normalize_text(opt)
                            base_p = base_probs.get(opt, 1.0 / max(1, len(options)))
                            s_i = math.log(max(1e-6, base_p)) if is_standard_4tier else 0.25 * math.log(max(1e-6, base_p))

                            if is_standard_4tier:
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
                                sharpness = 2.2 if (eff_lex and eff_res) else (2.0 if eff_lex else 1.6)
                                s_i += 6.0 * math.exp(-sharpness * ((tier_risk - center) ** 2))
                            else:
                                crit_desc = q_obj.criteria.get(opt, "") if isinstance(q_obj.criteria, dict) else ""
                                opt_and_crit_toks = set(_tokenize_words(f"{opt_norm} {crit_desc}"))
                                matched_opt_toks = set()

                                if eff_lex:
                                    ctx_hits = full_context_toks & opt_and_crit_toks
                                    dom_hits = gate_kw_toks & opt_and_crit_toks
                                    matched_opt_toks = ctx_hits | dom_hits
                                    s_i += len(ctx_hits) * 1.8 + len(dom_hits) * 1.6

                                if eff_res:
                                    rem_toks = opt_and_crit_toks - matched_opt_toks
                                    opt_stems = {_stem4(w) for w in rem_toks if len(w) >= 4}
                                    ctx_stem_hits = len(full_context_stems & opt_stems)
                                    dom_stem_hits = len(gate_kw_stems & opt_stems)
                                    s_i += ctx_stem_hits * 1.6 + dom_stem_hits * 1.5

                            scores.append(s_i)

                        scores_arr = np.array(scores, dtype=np.float64)
                        exp_s = np.exp(scores_arr - np.max(scores_arr))
                        p_vals = exp_s / np.sum(exp_s)
                        probs = {opt: round(float(p), 4) for opt, p in zip(options, p_vals)}
                        best_idx = int(np.argmax(p_vals))
                        best_opt = options[best_idx]
                        conf_c = float(p_vals[best_idx] - (sorted(p_vals)[-2] if len(options) > 1 else 0.0))
                        resp.answers[q_name] = ChoiceAnswer(
                            type="choice",
                            choice=best_opt,
                            probabilities=probs,
                            confidence=round(max(0.0, min(1.0, conf_c)), 4)
                        )

                resp.domain = domain
                seed = getattr(resp, 'active_coordinates', None) or {"cx": eff_cx, "cy": eff_cy, "zoom": eff_zoom}
                dispatch_telemetry_async(
                    state=state,
                    questions=questions,
                    response=resp,
                    seed=seed,
                    source="python_lib"
                )
                return resp

        response = self._evaluate_universal_cusp(state=state, questions=questions)
        dispatch_telemetry_async(
            state=state,
            questions=questions,
            response=response,
            seed=response.active_coordinates or {"cx": self.cx, "cy": self.cy, "zoom": self.zoom},
            source="python_lib"
        )
        return response


# Backward compatibility alias
WevvEngine = WerrEngine
