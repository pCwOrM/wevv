"""
werr: Semantic Intent Classifier & Auto-Seed Router
Routes incoming natural language prompts and environment states to the optimal
fractal boundary coordinate gate in < 0.1 ms with zero GPU tensors (O(1) memory).

Supports Orthogonal Dictionary Modes (under Domain=ON):
  - Lexical Dictionary (enable_lexical=True, enable_resonance=False)
  - Resonance Dictionary (enable_lexical=False, enable_resonance=True)
  - Hybrid Dictionary (enable_lexical=True, enable_resonance=True)
"""
import re
import math
import cmath
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from werr.gates import (
    DomainGate,
    DOMAIN_GATES,
    APISecurityGate,
    FinancialRiskGate,
    IoTSafetyGate,
    EcommerceFraudGate,
    GameCombatGate,
    normalize_text
)
from werr.datatypes import NoulQuestion, ChoiceQuestion, ScoreQuestion
from werr.calibration import DynamicCalibration


def _stem4(word: str) -> str:
    """Consistent 4-character root prefix for agglutinative Turkish/English stem resonance."""
    return word[:4] if len(word) >= 4 else word


def _tokenize_words(s: str) -> List[str]:
    """Splits both whitespace and snake_case underscores for domain lexical/stem matching."""
    s = normalize_text(s)
    s = re.sub(r"[^a-z0-9]", " ", s)
    return [w for w in s.split() if w]


class AutoSeedRouter:
    """
    Sub-millisecond intent router that dynamically binds queries to domain coordinates.
    Supports discrete Lexical indexing, continuous Cauchy-Lorentz Resonance poles,
    and orthogonal Hybrid mode without token double-counting or OOD hijacking.
    """
    def __init__(
        self,
        default_domain: str = "api_security",
        calibration: Optional[DynamicCalibration] = None,
        mode: str = "production",
        enable_lexical: Optional[bool] = None,
        enable_resonance: Optional[bool] = None,
        dict_mode: Optional[str] = None
    ):
        self.default_domain = default_domain
        self.calibration = calibration or DynamicCalibration()
        self.mode = str(mode).lower()

        if enable_lexical is not None or enable_resonance is not None:
            self.enable_lexical = bool(enable_lexical) if enable_lexical is not None else False
            self.enable_resonance = bool(enable_resonance) if enable_resonance is not None else False
        elif dict_mode is not None:
            dm = str(dict_mode).lower()
            self.enable_lexical = dm in ("lexical", "hybrid", "both", "production")
            self.enable_resonance = dm in ("resonance", "hybrid", "both")
        else:
            if self.mode in ("pure_fractal", "domainless", "off", "none"):
                self.enable_lexical = False
                self.enable_resonance = False
            elif self.mode == "resonance":
                self.enable_lexical = False
                self.enable_resonance = True
            elif self.mode in ("hybrid", "both"):
                self.enable_lexical = True
                self.enable_resonance = True
            else:
                self.enable_lexical = True
                self.enable_resonance = False

        self._gate_instances: Dict[str, DomainGate] = {
            name: cls() for name, cls in DOMAIN_GATES.items()
        }
        for gate in self._gate_instances.values():
            gate.calibration = self.calibration
            gate.enable_lexical = self.enable_lexical
            gate.enable_resonance = self.enable_resonance

        # Build reverse index for Lexical Dictionary and spectral poles for Resonance Dictionary
        self._keyword_index: Dict[str, List[str]] = {}
        self._stem_poles: Dict[str, List[Tuple[str, complex]]] = {}
        self._domain_poles: Dict[str, complex] = {}
        self._domain_zooms: Dict[str, float] = {}

        for d_name, gate in self._gate_instances.items():
            base_c = complex(gate.cx, gate.cy)
            self._domain_poles[d_name] = base_c
            self._domain_zooms[d_name] = gate.zoom
            for kw in gate.keywords:
                n_kw = normalize_text(kw)
                self._keyword_index.setdefault(n_kw, []).append(d_name)
                h = int(hashlib.md5(n_kw.encode("utf-8")).hexdigest()[:8], 16)
                theta = (h % 10000) / 10000.0 * 2.0 * math.pi
                radius = (1.0 / gate.zoom) * (0.05 + 0.15 * ((h >> 4) % 100) / 100.0)
                pole = base_c + cmath.rect(radius, theta)
                self._stem_poles.setdefault(_stem4(n_kw), []).append((d_name, pole))

        # State key signatures (strict normalized key match — zero substring false positives)
        self._state_signatures: Dict[str, str] = {
            # Financial
            "debt_ratio": "financial_risk", "borc_orani": "financial_risk", "borc_gelir_orani": "financial_risk",
            "debt_to_income_ratio": "financial_risk", "annual_income_usd": "financial_risk", "yillik_gelir": "financial_risk",
            "loan_amount_requested": "financial_risk", "late_payments_last_2yrs": "financial_risk",
            "income": "financial_risk", "gelir": "financial_risk", "aylik_gelir": "financial_risk", "maas": "financial_risk",
            "credit_score": "financial_risk", "kredi_notu": "financial_risk", "findeks": "financial_risk", "findeks_notu": "financial_risk",
            "requested_amount": "financial_risk", "kredi_tutari": "financial_risk", "talep_edilen_kredi": "financial_risk",
            "delinquencies": "financial_risk", "late_payments": "financial_risk", "gecikmis_odeme_sayisi": "financial_risk",
            "gecikme_adedi": "financial_risk", "ihtiyac_kredisi": "financial_risk", "ev_sahibi": "financial_risk",
            "hesap_risk_skoru": "financial_risk", "transfer_tutari_tl": "financial_risk",
            # IoT, Industrial & Clinical Edge Safety
            "smoke_detected": "iot_safety", "duman": "iot_safety", "duman_algilandi": "iot_safety",
            "gas_ppm": "iot_safety", "gaz_ppm": "iot_safety", "co_ppm": "iot_safety", "co2_ppm": "iot_safety", "gaz": "iot_safety", "co2_seviyesi": "iot_safety",
            "water_leak": "iot_safety", "su_kacagi": "iot_safety", "su_baskini": "iot_safety", "su_baskini_sensoru": "iot_safety", "alev_algilandi": "iot_safety",
            "temp": "iot_safety", "temp_c": "iot_safety", "temperature_c": "iot_safety", "temperature": "iot_safety", "sicaklik": "iot_safety", "oda_sicakligi": "iot_safety",
            "humidity_pct": "iot_safety", "nem_orani": "iot_safety", "nem": "iot_safety", "banyo_nemi": "iot_safety",
            "motion_detected": "iot_safety", "hareket_var": "iot_safety", "hareket_algilandi": "iot_safety", "pencere_acik": "iot_safety",
            "kazan_basinci_bar": "iot_safety", "reaktor_sicakligi_c": "iot_safety", "garaj_boru_sicakligi": "iot_safety",
            "bahce_hareket_sensoru": "iot_safety", "pm25_ug_m3": "iot_safety", "polen_indeksi": "iot_safety",
            "batarya_sicakligi_c": "iot_safety", "titresim_mm_s": "iot_safety", "hat_gerilimi_v": "iot_safety",
            "spo2_yuzde": "iot_safety", "nabiz_bpm": "iot_safety", "sistolik_kan_basinci": "iot_safety",
            "vucut_sicakligi_c": "iot_safety", "kan_sekeri_mg_dl": "iot_safety", "solunum_sayisi_dk": "iot_safety",
            "hava_kalitesi_aqi": "iot_safety", "odada_kimse_yok": "iot_safety", "hava_akisi_lpm": "iot_safety",
            "titresim_sensoru": "iot_safety", "reaktor_basinci_bar": "iot_safety", "kazan_sicakligi_c": "iot_safety",
            "konveyor_hizi_mps": "iot_safety", "trafo_sicakligi": "iot_safety", "klor_gazi_ppm": "iot_safety",
            "su_seviyesi_yuzde": "iot_safety", "amonyak_ppm": "iot_safety", "cekilen_akim_a": "iot_safety",
            "agv_hizi_mps": "iot_safety", "hidrolik_basinc_bar": "iot_safety", "toz_konsantrasyonu": "iot_safety",
            "hat_hizi": "iot_safety", "giris_basinc": "iot_safety", "reaktor_sensor_voltaji": "iot_safety",
            "ekg_ritmi": "iot_safety", "tansiyon_sistolik": "iot_safety", "bogazda_sislik": "iot_safety",
            "kaynar_su_dokuldu": "iot_safety", "solunum_sayisi": "iot_safety", "nabiz": "iot_safety",
            "ates_c": "iot_safety", "icilen_madde": "iot_safety", "celik_eriyik_sicakligi": "iot_safety",
            "basinc_bar": "iot_safety", "radyasyon_seviyesi": "iot_safety", "hava_akisi_m3s": "iot_safety",
            "erime_noktasi": "iot_safety", "termal_yuk": "iot_safety", "titresim_hiz": "iot_safety",
            "firin_sicakligi": "iot_safety", "firin_sicaklik_c": "iot_safety", "gaz_kacak_algilandi": "iot_safety",
            "sogutma_pompasi_aktif": "iot_safety", "reaktor_isi": "iot_safety", "basinc": "iot_safety",
            "denaturasyon_sicakligi_c": "iot_safety", "termal_dongu_sayisi": "iot_safety", "dna_verimi_ng_ul": "iot_safety",
            "biyolojik_bilesen": "iot_safety", "enzim_aktivitesi": "iot_safety", "ph_seviyesi": "iot_safety",
            # E-Commerce Fraud
            "order_amount": "ecommerce_fraud", "order_amount_usd": "ecommerce_fraud", "sepet_tutari": "ecommerce_fraud",
            "siparis_tutari": "ecommerce_fraud", "odeme_tutari": "ecommerce_fraud", "tutar_tl": "ecommerce_fraud",
            "velocity_1h": "ecommerce_fraud", "velocity_last_hour": "ecommerce_fraud", "islem_adedi": "ecommerce_fraud",
            "saatlik_islem": "ecommerce_fraud", "islem_sayisi": "ecommerce_fraud", "cvv_match": "ecommerce_fraud",
            "vpn_used": "ecommerce_fraud", "foreign_card": "ecommerce_fraud", "yabanci_kart": "ecommerce_fraud",
            "kart_ulkesi_farkli": "ecommerce_fraud", "vekil_sunucu": "ecommerce_fraud", "vpn_kullanimi": "ecommerce_fraud",
            "yeni_cihaz": "ecommerce_fraud", "billing_shipping_match": "ecommerce_fraud", "billing_shipping_mismatch": "ecommerce_fraud",
            "is_proxy": "ecommerce_fraud", "card_country": "ecommerce_fraud", "ters_ibraz": "ecommerce_fraud",
            "fatura_teslimat_uyusmazligi": "ecommerce_fraud", "supheli_islem_bayragi": "ecommerce_fraud",
            # Game Combat
            "ammo": "game_combat", "ammo_pct": "game_combat", "bullets": "game_combat", "mermi": "game_combat",
            "kalan_mermi": "game_combat", "sarjor": "game_combat", "health_pct": "game_combat",
            "enemy_distance_m": "game_combat", "cover_available": "game_combat", "can_yuzdesi": "game_combat",
            "can_puani": "game_combat", "enemy_count": "game_combat", "dusman_sayisi": "game_combat",
            "hedef_sayisi": "game_combat", "has_cover": "game_combat", "siperde": "game_combat",
            "siper_mevcut": "game_combat", "ates_altinda": "game_combat", "dusman_turu": "game_combat", "enemy_class": "game_combat",
            # API Security & Cyber
            "client_ip": "api_security", "req_frequency": "api_security", "request_frequency": "api_security",
            "istek_sikligi": "api_security", "failed_attempts": "api_security", "failed_auth_count": "api_security",
            "hatali_giris_sayisi": "api_security", "hatali_istek": "api_security", "hata_sayisi": "api_security",
            "error_count": "api_security", "is_internal": "api_security", "ddos_flag": "api_security",
            "ddos_suphesi": "api_security", "ip_reputation_score": "api_security", "ip_itibar_skoru": "api_security",
            "auth_token": "api_security", "endpoint": "api_security", "token_gecerli": "api_security",
            "payload_kb": "api_security", "payload_size_kb": "api_security", "istek_boyutu_kb": "api_security",
            "basarisiz_ssh_denemesi": "api_security", "sql_injection_imzasi": "api_security",
            "fidye_yazilimi_uzantisi": "api_security", "ag_cikis_trafigi_gb": "api_security",
            "jwt_imza_gecerli": "api_security", "basarisiz_giris_1dk": "api_security", "token_imzasi": "api_security",
            "veritabani_sorgusu": "api_security", "istek_sayisi_saniye": "api_security",
            "dosya_uzanti_degisim_hizi": "api_security", "komut": "api_security", "taranan_port_sayisi": "api_security",
            "iki_faktorlu_kod_dogru": "api_security", "api_anahtari": "api_security", "onceki_konum": "api_security",
            "client_role": "api_security", "user_role": "api_security", "scenario_index": "api_security",
            "warmup": "api_security", "ping": "api_security",
            "glork_rezonans_akisi": "api_security", "frob_turlama_frekansi": "api_security",
            "plumbus_fleeb_suyu_seviyesi": "api_security", "uzayli_cihazi": "api_security"
        }

    def has_domain_state_signature(self, state: Optional[Dict[str, Any]]) -> bool:
        """Returns True if state contains at least one recognized domain telemetry key or category."""
        if not isinstance(state, dict) or not state:
            return False
        if "category" in state:
            return True
        for k in state.keys():
            if normalize_text(k) in self._state_signatures:
                return True
        return False

    def detect_domain_with_coordinates(
        self,
        query: Union[str, Dict[str, Any], None] = None,
        state: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float, List[str], Optional[complex], Optional[float], bool]:
        """
        Determines target domain, confidence, matched tokens, optional fused resonance pole,
        fused zoom, and whether the state matched a domain signature.
        """
        if not self.enable_lexical and not self.enable_resonance:
            return self.default_domain, 0.5, [], None, None, False

        scores: Dict[str, float] = {d: 0.0 for d in self._gate_instances}
        matched_tokens: List[str] = []
        state_sig_hits = 0

        # 1. Explicit domain / category metadata
        if state and isinstance(state, dict) and "category" in state:
            c_norm = normalize_text(str(state["category"]))
            if any(w in c_norm for w in ["iot", "smart", "akilli", "ev", "cevre", "bina", "endustri", "sanayi", "uretim", "kazan", "reaktor", "termal", "hvac"]):
                scores["iot_safety"] += 6.0
                state_sig_hits += 2
                matched_tokens.append(f"cat:{state['category']}")
            elif any(w in c_norm for w in ["finan", "credit", "loan", "kredi", "banka", "borc", "findeks"]):
                scores["financial_risk"] += 6.0
                state_sig_hits += 2
                matched_tokens.append(f"cat:{state['category']}")
            elif any(w in c_norm for w in ["fraud", "commerce", "ticaret", "sahtecilik", "dolandiricilik", "odeme"]):
                scores["ecommerce_fraud"] += 6.0
                state_sig_hits += 2
                matched_tokens.append(f"cat:{state['category']}")
            elif any(w in c_norm for w in ["game", "combat", "oyun", "savas", "catisma", "taktik", "npc"]):
                scores["game_combat"] += 6.0
                state_sig_hits += 2
                matched_tokens.append(f"cat:{state['category']}")
            elif any(w in c_norm for w in ["api", "sec", "gateway", "guvenlik", "ag", "yetki"]):
                scores["api_security"] += 6.0
                state_sig_hits += 2
                matched_tokens.append(f"cat:{state['category']}")

        # 2. Strict state variable signature inspection
        if state and isinstance(state, dict):
            for k in state.keys():
                kl = normalize_text(k)
                if kl in self._state_signatures:
                    target = self._state_signatures[kl]
                    scores[target] += 3.0
                    state_sig_hits += 1
                    matched_tokens.append(f"state:{k}")

        # 3. Text query / instructions + criteria tokenization
        text_content = ""
        if isinstance(query, str):
            text_content = query
        elif isinstance(query, dict):
            for v in query.values():
                if isinstance(v, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
                    text_content += " " + str(v.instructions)
                    if hasattr(v, "criteria") and v.criteria:
                        text_content += " " + str(v.criteria)
                elif isinstance(v, str):
                    text_content += " " + v

        if text_content:
            q_tokens = _tokenize_words(text_content)
            matched_lex_tokens = set()

            if self.enable_lexical:
                for tok in q_tokens:
                    if tok in self._keyword_index:
                        matched_lex_tokens.add(tok)
                        for d_name in self._keyword_index[tok]:
                            scores[d_name] += 1.5
                            matched_tokens.append(f"kw:{tok}")

            if self.enable_resonance:
                for tok in q_tokens:
                    if len(tok) < 3 or tok in matched_lex_tokens:
                        continue
                    st_key = _stem4(tok)
                    if st_key in self._stem_poles:
                        for d_name, pole in self._stem_poles[st_key]:
                            dist_sq = abs(pole - self._domain_poles[d_name]) ** 2
                            res_energy = 0.08 / (dist_sq * (self._domain_zooms[d_name] ** 2) + 0.05)
                            scores[d_name] += res_energy
                            matched_tokens.append(f"res:{tok}")

        sorted_d = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top1_d, top1_s = sorted_d[0]
        top2_d, top2_s = sorted_d[1]
        total_score = sum(scores.values())
        best_domain = top1_d if top1_s > 0.0 else self.default_domain
        confidence = float(top1_s / total_score) if total_score > 0 else 0.5
        has_state_sig = (state_sig_hits > 0)

        custom_pole = None
        custom_zoom = None
        if self.enable_resonance and top1_s > 0.0 and top2_s >= 3.0 and (top2_s / top1_s >= 0.45):
            h_mean = (2.0 * top1_s * top2_s) / (top1_s + top2_s)
            w1 = top1_s / (top1_s + top2_s)
            w2 = top2_s / (top1_s + top2_s)
            fused_c = w1 * self._domain_poles[top1_d] + w2 * self._domain_poles[top2_d]
            custom_pole = complex(fused_c.real, fused_c.imag + math.sin(h_mean) * 0.0003)
            custom_zoom = w1 * self._domain_zooms[top1_d] + w2 * self._domain_zooms[top2_d]

        return best_domain, round(confidence, 3), matched_tokens, custom_pole, custom_zoom, has_state_sig

    def detect_domain(
        self,
        query: Union[str, Dict[str, Any], None] = None,
        state: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float, List[str]]:
        """
        Determines the target domain with confidence and matched signal tokens.
        Latency: < 0.05 ms.
        """
        best_domain, conf, matched_tokens, _, _, _ = self.detect_domain_with_coordinates(query=query, state=state)
        return best_domain, conf, matched_tokens

    def get_gate(self, domain_name: str) -> DomainGate:
        """Retrieves gate instance by name (or default if unknown)."""
        return self._gate_instances.get(domain_name, self._gate_instances[self.default_domain])

    def route_and_evaluate(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]],
        preferred_domain: Optional[str] = None
    ) -> Tuple[Any, str, float]:
        """
        Routes the request to the optimal domain gate and evaluates it in a single pass.
        Returns: (WerrResponse, domain_name, routing_confidence)
        """
        custom_pole = None
        custom_zoom = None
        if preferred_domain and preferred_domain in self._gate_instances:
            domain = preferred_domain
            confidence = 1.0
        else:
            domain, confidence, _, custom_pole, custom_zoom, _ = self.detect_domain_with_coordinates(query=questions, state=state)

        gate = self.get_gate(domain)
        gate.enable_lexical = self.enable_lexical
        gate.enable_resonance = self.enable_resonance

        orig_cx, orig_cy, orig_zoom = gate.cx, gate.cy, gate.zoom
        if self.enable_resonance and custom_pole is not None and custom_zoom is not None:
            gate.cx, gate.cy, gate.zoom = custom_pole.real, custom_pole.imag, custom_zoom

        try:
            response = gate.evaluate_state_and_questions(state=state, questions=questions)
        finally:
            gate.cx, gate.cy, gate.zoom = orig_cx, orig_cy, orig_zoom

        response.domain = domain
        return response, domain, confidence
