# Output Data

Artifacts produced by training and evaluation runs. Generated content is
gitignored; only the directory layout and these READMEs are tracked.

| Subfolder | Contents | Created by |
|---|---|---|
| `trained_models/` | Saved SNN connection weights (`snn_weights.pt`) | `src/3.run_train_chbmit.py` |
| `train_logs/` | Per-epoch metrics CSV (`training_log.csv`) | `src/3.run_train_chbmit.py` |
| `results/` | Evaluation summary CSV and figures (confusion matrix, ROC, PR) | `src/4.run_eval_chbmit.py` |
