"""atlime - LIME and ArcTanh-LIME implemented from scratch (Python standard library only).

Public API
----------
LimeTabular, LimeConfig            baseline LIME (continuous tabular mode)
ArcTanhLime, ArcTanhConfig         epsilon-walk + tanh^-1 regressor (proposed)
DeltaAnchorEnsemble, DeltaConfig   +/- delta anchors blended into one surrogate
epsilon_walk                       the walk to the nearest local max / min
Rng                                deterministic PRNG shared with the web port
"""

from .arctanh import ArcTanhConfig, ArcTanhLime, ArcTanhResult
from .ensemble import DeltaAnchorEnsemble, DeltaConfig, EnsembleResult, anchor_points
from .lime import LimeConfig, LimeResult, LimeTabular, training_bounds, training_scale
from .rng import Rng
from .surrogates import BlendedSurrogate, LinearSurrogate, TanhSurrogate
from .walk import WalkResult, epsilon_walk

__all__ = [
    "ArcTanhConfig", "ArcTanhLime", "ArcTanhResult",
    "DeltaAnchorEnsemble", "DeltaConfig", "EnsembleResult", "anchor_points",
    "LimeConfig", "LimeResult", "LimeTabular", "training_bounds", "training_scale",
    "Rng", "BlendedSurrogate", "LinearSurrogate", "TanhSurrogate",
    "WalkResult", "epsilon_walk",
]
