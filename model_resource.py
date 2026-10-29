"""Share a lazily loaded model without caching or replaying UI messages."""
from threading import Lock


class ModelResource:
    def __init__(self):
        self._model = None
        self._lock = Lock()

    def get(self, factory, on_load=None):
        with self._lock:
            if self._model is None:
                if on_load is not None:
                    on_load("loading")
                # Keep failures uncached so the next run can retry safely.
                self._model = factory()
            return self._model
