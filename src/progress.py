"""Progress event emitter for the GC Agent Regression Tester.

Provides thread-safe event distribution to multiple subscribers using queue.Queue.
Late subscribers (e.g. SSE after redirect) receive a replay of recent events.
"""

import queue
import threading
from collections import deque
from typing import List, Optional

from .models import ProgressEvent

# Keep enough history for SSE clients that connect after /run redirects to /results
_EVENT_HISTORY_SIZE = 500


class ProgressEmitter:
    """Publishes progress events to subscribers via thread-safe queues.

    Subscribers receive events through individual queue.Queue instances,
    enabling both SSE (web) and console consumers to receive updates independently.
    New subscribers are replayed the recent event history so they do not miss
    early suite/scenario events.
    """

    def __init__(self) -> None:
        """Initialize with empty subscriber list and event history."""
        self._subscribers: List[queue.Queue] = []
        self._history: deque[ProgressEvent] = deque(maxlen=_EVENT_HISTORY_SIZE)
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        """Return a new queue that will receive progress events.

        Replays buffered events emitted before this subscription.

        Returns:
            A queue.Queue instance that will receive all future ProgressEvent objects.
        """
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
            for event in self._history:
                q.put_nowait(event)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        """Remove a subscriber so it no longer receives events.

        Args:
            q: The queue previously returned by subscribe().
        """
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass  # Already removed or never subscribed

    def emit(self, event: ProgressEvent) -> None:
        """Publish a progress event to all subscribers and print to console.

        Args:
            event: The ProgressEvent to distribute.
        """
        line = f"[{event.event_type.value}] {event.message}"
        if event.detail:
            line += f" — {event.detail}"
        print(line)

        with self._lock:
            self._history.append(event)
            for subscriber in self._subscribers:
                subscriber.put_nowait(event)

    def clear_history(self) -> None:
        """Clear buffered events (e.g. before starting a new test run)."""
        with self._lock:
            self._history.clear()
