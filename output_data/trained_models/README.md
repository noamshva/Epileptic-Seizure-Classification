# trained_models/

SNN connection weights saved by `src/3.run_train_chbmit.py`:

- `snn_weights.pt` — `torch.save` of a dict mapping `"<src_layer>__<tgt_layer>"`
  to the connection's weight tensor. Loaded by `src/4.run_eval_chbmit.py`.

All generated content is gitignored; only this README is tracked.
