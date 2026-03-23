import dataclasses
from pathlib import Path
from typing import Any, Self

import tyro

from tsubame.utils.console import get_console


@dataclasses.dataclass
class InstantiatableConfig:
    _type: type

    def instantiate(self, **kwargs: Any) -> Any:
        return self._type(self, **kwargs)

    def print(self) -> None:
        console = get_console()
        console.rule(self.__class__.__name__)
        console.print(self)
        console.rule()

    def save(self, path: Path) -> None:
        string = tyro.extras.to_yaml(self)
        with open(path, "w") as fp:
            fp.write(string)

    @classmethod
    def load(cls, path: Path) -> Self:
        with open(path) as fp:
            string = fp.read()
        config = tyro.extras.from_yaml(cls, string)
        return config
