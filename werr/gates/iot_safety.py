"""
werr: IoT Life Safety & Industrial Hazard Domain Gate
Calibrated for sub-millisecond environmental hazard triage (smoke, fire, gas leaks, extreme temperature).
"""
from typing import Dict, Any, Tuple
import math
import hashlib
import numpy as np

from werr.gates.base import DomainGate, normalize_text, safe_float


class IoTSafetyGate(DomainGate):
    name = "iot_safety"
    # Calibrated Boundary Coordinates for Critical Hazard Transition
    cx = -0.745
    cy = 0.112
    zoom = 85.0
    default_threshold = 0.50

    keywords = [
        "iot", "sensor", "temperature", "temp", "smoke", "gas", "co2", "fire", "leak",
        "hazard", "evacuate", "alarm", "thermostat", "hvac", "smart_home", "furnace", "air_conditioner",
        "sicaklik", "duman", "gaz", "yangin", "kacak", "tehlike", "alarm", "tahliye", "sensor",
        "akilli_ev", "klima", "termostat", "havalandirma", "nem", "duman_dedektoru", "hareket",
        "pencere", "kapi", "su_baskini", "donma_riski", "freeze", "boiler", "ventilation", "occupancy", "presence", "motion", "flood"
    ]

    def project_state(self, state: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        values = []
        net_risk = 0.0

        temp = safe_float(state.get("temp", state.get("temperature", state.get("temp_c", state.get("sicaklik", state.get("oda_sicakligi", state.get("kazan_sicakligi_c", state.get("trafo_sicakligi", 22.0))))))), default=22.0)
        smoke = state.get("smoke_detected", state.get("smoke", state.get("duman", state.get("duman_algilandi", state.get("duman_sensoru", False)))))
        toxic_gas_ppm = safe_float(state.get("gas_ppm", state.get("co_ppm", state.get("gaz_ppm", state.get("gaz", state.get("amonyak_ppm", state.get("klor_gazi_ppm", 0.0)))))))
        co2_ppm = safe_float(state.get("co2_ppm", state.get("co2_seviyesi", 0.0)))
        water_leak = state.get("water_leak", state.get("leak", state.get("su_kacagi", state.get("su_baskini", state.get("su_baskini_sensoru", False)))))
        flame = state.get("flame_detected", state.get("fire", state.get("alev", state.get("yangin", state.get("yangin_alarmi", state.get("alev_algilandi", False))))))
        motion = state.get("motion_detected", state.get("presence", state.get("motion", state.get("hareket_var", state.get("hareket_algilandi", state.get("varlik", False))))))
        window_open = state.get("window_open", state.get("pencere_acik", False))
        room = normalize_text(str(state.get("room", state.get("zone", state.get("oda", state.get("alan", ""))))))

        # 1. Temperature Anomaly
        if temp > 65.0 or temp < -15.0:
            net_risk += 3.0  # Acute fire or catastrophic freezing emergency
            values.append(1.0)
        elif temp > 45.0:
            net_risk += 1.5  # Severe overheat
            values.append(0.6)
        elif temp > 35.0:
            net_risk += 0.5
            values.append(0.2)
        elif temp < 2.0:
            net_risk += 1.0  # Frost danger
            values.append(0.4)
        else:
            net_risk -= 0.5
            values.append(-0.5)

        # 2. Smoke & Flame Detection (Immediate life safety priority)
        if bool(smoke) or bool(flame):
            net_risk += 3.5
            values.append(1.0)
        else:
            values.append(-0.8)

        # 3. Toxic Gas (CO, natural gas, ammonia, chlorine) vs Indoor CO2
        if toxic_gas_ppm > 50.0:
            net_risk += 3.2
            values.append(1.0)
        elif toxic_gas_ppm > 14.0:
            net_risk += 2.2
            values.append(0.8)
        elif co2_ppm > 2000.0:
            net_risk += 2.5
            values.append(0.9)
        elif co2_ppm > 1200.0:
            net_risk += 1.0
            values.append(0.4)
        else:
            values.append(-0.8)

        # 4. Water Leak / Structural
        if bool(water_leak):
            net_risk += 2.0
            values.append(0.8)
        else:
            values.append(-0.5)

        # 5. Industrial & Clinical Acute Hazard Flags
        if any(bool(state.get(k, False)) for k in [
            "sensor_hatti_koptu", "sensor_hattı_koptu", "hortum_patladi", "kavitasyon",
            "guvenlik_bariyeri_kesildi", "sikisma_tehlikesi", "nobet_geciriyor", "kanama", "stridor"
        ]):
            net_risk += 3.0
        if "solunum_sayisi" in state and safe_float(state.get("solunum_sayisi"), default=14.0) <= 2.0:
            net_risk += 3.2
        if safe_float(state.get("apne_suresi_sn", 0.0)) >= 20.0:
            net_risk += 3.0

        # False wind alarm guard (locked door + unbroken glass in high wind)
        if state.get("cam_kirilmasi") is False and state.get("kapi_kilitli") is True:
            net_risk -= 2.0

        # 6. Zone Criticality Modulations
        if any(z in room for z in ['server', 'sunucu']) and temp > 32.0:
            net_risk += 1.2
        elif any(z in room for z in ['greenhouse', 'sera']) and temp < 10.0:
            net_risk += 0.8

        while len(values) < 4:
            values.append(0.0)

        return np.array(values[:4], dtype=np.float64), float(net_risk)
