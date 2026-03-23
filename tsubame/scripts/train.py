import random

import loguru
import numpy as np
import torch
import tyro
from rich.logging import RichHandler

from tsubame.engines import Trainer, TrainerConfig
from tsubame.utils import distributed
from tsubame.utils.console import get_console


def main(config: TrainerConfig) -> None:
    # ================================================================
    # Setup the logger.

    console = get_console()

    loguru.logger.remove()
    loguru.logger.add(
        sink=RichHandler(
            console=console,
            show_level=False,
            show_path=False,
            show_time=False,
            rich_tracebacks=True,
        ),
        colorize=False,
        diagnose=True,
        backtrace=True,
        format=distributed.get_format(),
    )

    # ================================================================
    # Print the config.

    if distributed.is_master_rank():
        config.print()
        config.save()

    # ================================================================
    # Initialize the process group for distributed training.

    distributed.init_process_group(
        backend=config.distributed.backend,
        init_method=config.distributed.init_method,
        master_addr=config.distributed.master_addr,
        master_port=config.distributed.master_port,
        global_size=config.distributed.global_size,
        global_rank=config.distributed.global_rank,
        device_id=config.distributed.local_rank,
    )

    loguru.logger.success("🔥 Distributed process group has been initialized!")

    # ================================================================
    # Ensure that each process exclusively works on a single GPU.

    torch.cuda.set_device(config.distributed.local_rank)

    # ================================================================
    # Set the backend options.

    torch.backends.cudnn.benchmark = config.backend.benchmark
    torch.backends.cudnn.deterministic = config.backend.deterministic
    torch.use_deterministic_algorithms(config.backend.deterministic, warn_only=True)

    # ================================================================
    # Fix the random seed for reproducibility.

    seed = config.random.seed + config.distributed.global_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ================================================================
    # Start training.

    trainer: Trainer = config.instantiate()

    torch.distributed.barrier()

    loguru.logger.success("🚀 Training started!")

    trainer.run()

    torch.distributed.barrier()

    loguru.logger.success("✨ Training finished!")

    torch.distributed.destroy_process_group()


def entrypoint() -> None:
    main(tyro.cli(TrainerConfig))
