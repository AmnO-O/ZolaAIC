# Historical comparison runners

These train-only scripts are not part of main inference. They retain direct Caller,
initial retrieval planning and adaptive BM25 controls for reproducible comparisons.
Run from the repository root using python experiments/<script>.py. All prompts
exclude gold labels; diagnostic evaluation happens after each episode.

The currently selected main architecture is app Planner/Caller. Frozen historical
Kaggle jobs pin earlier source commits and remain reproducible at those commits.
