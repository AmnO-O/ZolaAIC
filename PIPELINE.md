# Pipeline

One implementation: [agent.py](src/lifemate/agent.py). Read [architecture](docs/architecture.md), [training protocol](docs/training.md), and [notebooks](notebooks/README.md).

Train metadata -> causal action examples -> QLoRA adapter -> frozen base/adapter live-tool comparison -> audio evaluation -> offline inference.

No old baseline variants remain in the working source. Historical code is available in Git history. The untrained model remains an experimental control, not a successful solution.
