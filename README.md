# Distributed Training Template for TSUBAME 4.0

This repository provides a template for multi-node distributed training on [TSUBAME 4.0](https://www.t4.cii.isct.ac.jp/docs/handbook.en). 
It does not include any actual training logic — please use it as a starting point for your own project.

## Installation

1. Install [uv](https://github.com/astral-sh/uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install this project

```bash
uv sync --frozen
```

## Training

### CLI

[tsubame-train](./tsubame/scripts/train.py) is an entry point for distributed training.
Currently, only the general settings, which are required regardless of your choice of data or model, can be configured via the CLI powered by [tyro](https://github.com/brentyi/tyro).
Please refer to [tyro](https://github.com/brentyi/tyro) for details about the configuration system.

```bash
module load openmpi
uv run tsubame-train -h
```

```console
usage: tsubame-train [-h] [OPTIONS]

╭─ options ─────────────────────────────────────────────────────────────────────────────╮
│ -h, --help                                                                            │
│       show this help message and exit                                                 │
╰───────────────────────────────────────────────────────────────────────────────────────╯
╭─ git options ─────────────────────────────────────────────────────────────────────────╮
│ --git.branch {fixed}                                                                  │
│       (fixed to: main)                                                                │
│ --git.commit-id {fixed}                                                               │
│       (fixed to: 68f58480161d6e62f88be049c312232e25edf1d6)                            │
│ --git.remote-url {fixed}                                                              │
│       (fixed to: git@github.com:skmhrk1209/TSUBAME-Distributed-Training-Template.git) │
╰───────────────────────────────────────────────────────────────────────────────────────╯
╭─ random options ──────────────────────────────────────────────────────────────────────╮
│ --random.seed INT                                                                     │
│       (default: 0)                                                                    │
╰───────────────────────────────────────────────────────────────────────────────────────╯
╭─ backend options ─────────────────────────────────────────────────────────────────────╮
│ --backend.benchmark, --backend.no-benchmark                                           │
│       (default: True)                                                                 │
│ --backend.deterministic, --backend.no-deterministic                                   │
│       (default: False)                                                                │
╰───────────────────────────────────────────────────────────────────────────────────────╯
╭─ distributed options ─────────────────────────────────────────────────────────────────╮
│ --distributed.backend {GLOO,NCCL,XCCL,UCC,MPI}                                        │
│       (default: NCCL)                                                                 │
│ --distributed.init-method {ENV,TCP,FILE}                                              │
│       (default: ENV)                                                                  │
│ --distributed.master-addr STR                                                         │
│       (default: 10.30.1.43)                                                           │
│ --distributed.master-port INT                                                         │
│       (default: 56521)                                                                │
│ --distributed.global-size INT                                                         │
│       (default: 1)                                                                    │
│ --distributed.global-rank INT                                                         │
│       (default: 0)                                                                    │
│ --distributed.local-size INT                                                          │
│       (default: 1)                                                                    │
│ --distributed.local-rank INT                                                          │
│       (default: 0)                                                                    │
╰───────────────────────────────────────────────────────────────────────────────────────╯
╭─ experiment options ──────────────────────────────────────────────────────────────────╮
│ --experiment.output-dir PATH                                                          │
│       (default: outputs)                                                              │
│ --experiment.project-name STR                                                         │
│       (default: project)                                                              │
│ --experiment.experiment-name STR                                                      │
│       (default: experiment)                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────╯
```

### [Interactive Jobs](https://www.t4.cii.isct.ac.jp/docs/handbook.en/jobs/#interactive)

For interactive jobs, you launch one process per GPU on each node via OpenMPI's `mpiexec` command from the master node.

1. Log in to the master node using the `qrsh` command.

```bash
qrsh -g ${GROUP_NAME} -l ${NODE_TYPE}=${NUM_NODES} -l h_rt=${WALLTIME} 
```

2. Launch a process for each GPU on each node, yielding a process group consisting of `$((NUM_GPUS * NUM_NODES))` processes.

```bash
module load openmpi
mpiexec -npernode ${NUM_GPUS} -n $((NUM_GPUS * NUM_NODES)) uv run tsubame-train
```

### [Batch Jobs](https://www.t4.cii.isct.ac.jp/docs/handbook.en/jobs/#submit)

Distributed training via a batch job is essentially the same as with an interactive job, but requires writing a batch script for each job, which can be tedious.
Therefore, I've created a utility, [tsubame-launch](./tsubame/scripts/launch.py), that handles everything from automatically generating batch scripts to submitting jobs in one go. 
[tsubame-launch](./tsubame/scripts/launch.py) can be configured via [Hydra](https://hydra.cc/docs/intro), making it easy to launch multiple batch jobs at once via Hydra's [Multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run).
You can configure [tsubame-launch](./tsubame/scripts/launch.py) by directly modifying [configs](./configs) or overriding them via CLI.
Please refer to [Hydra](https://hydra.cc/docs/intro) for details about the configuration system.


```bash
# Launch jobs with different parameters (e.g., `scheduler.params.seed`) at once via Hydra's Multi-run.
uv run tsubame-launch -m \
    scheduler.group_name=${GROUP_NAME} \
    scheduler.node_type=${NODE_TYPE} \
    scheduler.num_nodes=${NUM_NODES} \
    scheduler.num_gpus=${NUM_GPUS} \
    scheduler.walltime=${WALLTIME} \
    scheduler.params.seed=0,1 \
    experiment_name='seed_${scheduler.params.seed}'
```
