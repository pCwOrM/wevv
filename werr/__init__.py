"""
werr: Zero-Memory Fractal System-One Decision Engine (Waves & Errors)
Open-source machine-native intuitive decision framework for software.
"""

from werr.datatypes import (
    NoulQuestion,
    ChoiceQuestion,
    ScoreQuestion,
    NoulAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    WerrResponse,
    WevvResponse,
)
from werr.engine import WerrEngine, WevvEngine
from werr.calibration import DynamicCalibration
from werr.presets import (
    create_security_guard,
    create_smart_router,
    create_risk_evaluator,
)
from werr.router import AutoSeedRouter
from werr.gates import (
    DomainGate,
    DOMAIN_GATES,
    APISecurityGate,
    FinancialRiskGate,
    IoTSafetyGate,
    EcommerceFraudGate,
    GameCombatGate,
)

from werr.adapters import JevWireAdapter

__version__ = "0.5.1"
__all__ = [
    "WerrEngine",
    "WevvEngine",
    "JevWireAdapter",
    "DynamicCalibration",
    "AutoSeedRouter",
    "DomainGate",
    "DOMAIN_GATES",
    "APISecurityGate",
    "FinancialRiskGate",
    "IoTSafetyGate",
    "EcommerceFraudGate",
    "GameCombatGate",
    "NoulQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "NoulAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "WerrResponse",
    "WevvResponse",
    "create_security_guard",
    "create_smart_router",
    "create_risk_evaluator",
]
