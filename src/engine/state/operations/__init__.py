"""World operations — character and world scoped."""

from .character import CharacterOps
from .faction import FactionOps
from .world import WorldOps


class WorldOperations(CharacterOps, FactionOps, WorldOps):
    """Full operation surface. Shared __init__ / _log live on _OpsBase."""
