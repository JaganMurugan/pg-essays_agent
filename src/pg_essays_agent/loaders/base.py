from collections.abc import Iterator
from typing import Protocol

from pg_essays_agent.models import Essay

class EssayLoader(Protocol):
    """Interface that every essay loader must follow"""

    def load(self) -> Iterator[Essay]:
        """Load essays and yield them one at a time"""
        ...