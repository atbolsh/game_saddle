"""Training code for the self-training project.

``training/train.py`` is the source-agnostic QLoRA LIBRARY: the loop
(``run_training(sources, config)``), the ``TrainConfig`` knob dataclass, and
a generic CLI front-end (``python -m training.train``). Concrete runs are
short scripts that pick sources + config and call the loop -- copy
``run_first_iteration.py`` per run. Future training code -- concrete
DataSources (game traces, replay sets, planted errors), the early-warning
probe suite -- lands in this package too. The design docs (TRAINING_*.md)
live alongside; start with TRAINING_OVERVIEW.md.
"""
