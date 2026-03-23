import dataclasses
from pathlib import Path
from typing import override

import loguru
import tyro

from tsubame.utils import distributed, git
from tsubame.utils.config import InstantiatableConfig
from tsubame.utils.console import get_console


@dataclasses.dataclass
class GitConfig:
    branch: str = dataclasses.field(default_factory=git.get_branch)
    commit_id: str = dataclasses.field(default_factory=git.get_commit_id)
    remote_url: str = dataclasses.field(default_factory=git.get_remote_url)


@dataclasses.dataclass
class RandomConfig:
    seed: int = 0


@dataclasses.dataclass
class BackendConfig:
    benchmark: bool = True
    deterministic: bool = False


@dataclasses.dataclass
class DistributedConfig:
    backend: distributed.Backend = distributed.Backend.NCCL
    init_method: distributed.InitMethod = distributed.InitMethod.ENV
    master_addr: str = dataclasses.field(default_factory=distributed.get_master_address)
    master_port: int = dataclasses.field(default_factory=distributed.get_master_port)
    global_size: int = dataclasses.field(default_factory=distributed.get_global_size)
    global_rank: int = dataclasses.field(default_factory=distributed.get_global_rank)
    local_size: int = dataclasses.field(default_factory=distributed.get_local_size)
    local_rank: int = dataclasses.field(default_factory=distributed.get_local_rank)


@dataclasses.dataclass
class ExperimentConfig:
    output_dir: Path = Path("outputs")
    project_name: str = "project"
    experiment_name: str = "experiment"

    def get_output_dir(self) -> Path:
        return self.output_dir

    def get_project_dir(self) -> Path:
        return self.get_output_dir() / self.project_name

    def get_experiment_dir(self) -> Path:
        return self.get_project_dir() / self.experiment_name


@dataclasses.dataclass
class TrainerConfig(InstantiatableConfig):
    _type: tyro.conf.Suppress[type] = dataclasses.field(default_factory=lambda: Trainer)
    git: tyro.conf.Fixed[GitConfig] = dataclasses.field(default_factory=GitConfig)
    random: RandomConfig = dataclasses.field(default_factory=RandomConfig)
    backend: BackendConfig = dataclasses.field(default_factory=BackendConfig)
    distributed: DistributedConfig = dataclasses.field(default_factory=DistributedConfig)
    experiment: ExperimentConfig = dataclasses.field(default_factory=ExperimentConfig)

    @override
    def save(self, basename: str = "config.yaml") -> None:
        experiment_dir = self.experiment.get_experiment_dir()
        experiment_dir.mkdir(parents=True, exist_ok=True)
        super().save(experiment_dir / basename)


@dataclasses.dataclass
class Trainer:
    config: TrainerConfig

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def run(self) -> None:
        console = get_console()
        console.rule(self.__class__.__name__)
        loguru.logger.info(
            f"This `{self.__class__.__name__}` class is a placeholder. "
            f"Please implement your own training logic here."
        )
        console.rule()
