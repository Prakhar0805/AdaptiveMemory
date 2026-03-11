from collections import deque
from typing import List, Dict, Optional, Any


class WorkingMemory:
    """Stores recent conversation turns using a fixed-size deque."""

    def __init__(self, maxlen: int = 10):
        self.memory = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.total_added = 0

    def add_turn(self, speaker: str, text: str, dia_id: str,
                 timestamp: Optional[str] = None, session: Optional[str] = None) -> None:
        self.memory.append({
            'speaker': speaker,
            'text': text,
            'dia_id': dia_id,
            'timestamp': timestamp or '',
            'session': session or '',
            'added_at': self.total_added
        })
        self.total_added += 1

    def get_recent(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        n = n or len(self.memory)
        return list(self.memory)[-n:]

    def get_formatted(self, n: Optional[int] = None, include_timestamps: bool = True) -> List[str]:
        turns = self.get_recent(n)
        formatted = []
        for turn in turns:
            if include_timestamps and turn['timestamp']:
                formatted.append(f"({turn['timestamp']}) {turn['speaker']}: {turn['text']}")
            else:
                formatted.append(f"{turn['speaker']}: {turn['text']}")
        return formatted

    def get_dia_ids(self, n: Optional[int] = None) -> List[str]:
        return [turn['dia_id'] for turn in self.get_recent(n)]

    def clear(self) -> None:
        self.memory.clear()

    def is_empty(self) -> bool:
        return len(self.memory) == 0

    def is_full(self) -> bool:
        return len(self.memory) == self.maxlen

    def get_stats(self) -> Dict[str, Any]:
        current = len(self.memory)
        return {
            'current_size': current,
            'max_size': self.maxlen,
            'total_added': self.total_added,
            'is_full': self.is_full(),
            'utilization': current / self.maxlen * 100 if self.maxlen > 0 else 0,
            'oldest_dia_id': self.memory[0]['dia_id'] if current > 0 else None,
            'newest_dia_id': self.memory[-1]['dia_id'] if current > 0 else None
        }

    def __len__(self) -> int:
        return len(self.memory)

    def __repr__(self) -> str:
        return f"WorkingMemory(size={len(self.memory)}/{self.maxlen}, total_added={self.total_added})"