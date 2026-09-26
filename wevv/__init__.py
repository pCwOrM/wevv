"""
wevv (Legacy Compatibility Shim):
Redirects legacy 'import wevv' directly to the primary 'werr' package.
"""

from werr import *
from werr.engine import WerrEngine as WevvEngine

__version__ = "0.5.1"

