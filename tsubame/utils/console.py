import functools
from typing import Any

from rich.console import Console


@functools.cache
def get_console(*args: Any, **kwargs: Any) -> Console:
    return Console(*args, **kwargs)
