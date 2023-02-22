from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class Positions:
    NOSE            : Optional[List[float]] = None
    LEFT_EYE        : Optional[List[float]] = None
    RIGHT_EYE       : Optional[List[float]] = None
    LEFT_EAR        : Optional[List[float]] = None
    RIGHT_EAR       : Optional[List[float]] = None
    LEFT_SHOULDER   : Optional[List[float]] = None
    RIGHT_SHOULDER  : Optional[List[float]] = None
    LEFT_ELBOW      : Optional[List[float]] = None
    RIGHT_ELBOW     : Optional[List[float]] = None
    LEFT_WRIST      : Optional[List[float]] = None
    RIGHT_WRIST     : Optional[List[float]] = None
    LEFT_HIP        : Optional[List[float]] = None
    RIGHT_HIP       : Optional[List[float]] = None
    LEFT_KNEE       : Optional[List[float]] = None
    RIGHT_KNEE      : Optional[List[float]] = None
    LEFT_ANKLE      : Optional[List[float]] = None
    RIGHT_ANKLE     : Optional[List[float]] = None
    LEFT_FOOT_INDEX : Optional[List[float]] = None
    RIGHT_FOOT_INDEX: Optional[List[float]] = None

    def serialize(self):
        return asdict(self)
    