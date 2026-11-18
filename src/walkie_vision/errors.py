"""The package's exception classes.

A bottom-layer module: it imports nothing from walkie_vision, so every other
module may raise or catch these.
"""


class StartupError(RuntimeError):
    """A startup failure whose message says what to do; main() logs it without a traceback."""
