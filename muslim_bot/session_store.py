from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str


class SessionStore:
    def __init__(self, max_turns: int = 12) -> None:
        self.max_turns = max_turns
        self._turns: dict[str, deque[Turn]] = {}

    def new_session_id(self) -> str:
        session_id = uuid4().hex
        self._turns[session_id] = deque(maxlen=self.max_turns)
        return session_id

    def ensure_session(self, session_id: str | None) -> str:
        if session_id and session_id in self._turns:
            return session_id
        if session_id:
            self._turns[session_id] = deque(maxlen=self.max_turns)
            return session_id
        return self.new_session_id()

    def append(self, session_id: str, user: str, assistant: str) -> None:
        self._turns.setdefault(session_id, deque(maxlen=self.max_turns)).append(
            Turn(user=user, assistant=assistant)
        )

    def history_prompt(self, session_id: str, latest_message: str) -> str:
        turns = self._turns.get(session_id)
        if not turns:
            return latest_message

        lines = [
            "Riwayat percakapan singkat untuk konteks. Jangan mengutip riwayat ini sebagai sumber agama.",
        ]
        for turn in turns:
            lines.append(f"Pengguna: {turn.user}")
            lines.append(f"Asisten: {turn.assistant}")
        lines.append(f"Pertanyaan terbaru: {latest_message}")
        return "\n".join(lines)
