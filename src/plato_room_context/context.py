"""Contextual room awareness."""
import time
from dataclasses import dataclass, field
from enum import Enum

class RoomState(Enum):
    ACTIVE = "active"
    QUIET = "quiet"
    EMPTY = "empty"
    STORM = "storm"

@dataclass
class ContextSignal:
    source: str
    signal_type: str
    data: dict
    timestamp: float = field(default_factory=time.time)

class RoomContext:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self._inhabitants: set[str] = set()
        self._signals: list[ContextSignal] = []
        self._state = RoomState.EMPTY
        self._activity_count = 0
        self._last_activity = 0.0

    def enter(self, agent: str) -> str:
        self._inhabitants.add(agent)
        self._activity_count += 1
        self._last_activity = time.time()
        self._update_state()
        return f"{agent} entered {self.room_id}"

    def leave(self, agent: str) -> str:
        self._inhabitants.discard(agent)
        self._activity_count += 1
        self._last_activity = time.time()
        self._update_state()
        return f"{agent} left {self.room_id}"

    def signal(self, source: str, signal_type: str, data: dict = None) -> ContextSignal:
        sig = ContextSignal(source=source, signal_type=signal_type, data=data or {})
        self._signals.append(sig)
        self._activity_count += 1
        self._last_activity = time.time()
        if len(self._signals) > 100:
            self._signals = self._signals[-100:]
        self._update_state()
        return sig

    def _update_state(self):
        if not self._inhabitants: self._state = RoomState.EMPTY
        elif self._activity_count > 20 and len(self._inhabitants) > 3: self._state = RoomState.STORM
        elif time.time() - self._last_activity > 300: self._state = RoomState.QUIET
        else: self._state = RoomState.ACTIVE

    @property
    def inhabitants(self) -> list[str]: return list(self._inhabitants)
    @property
    def state(self) -> RoomState: return self._state
    @property
    def stats(self) -> dict:
        return {"room": self.room_id, "inhabitants": len(self._inhabitants),
                "signals": len(self._signals), "state": self._state.value,
                "activity": self._activity_count}
