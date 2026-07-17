# Experimental code

This directory contains the PFLlib-based implementation used in the
experiments reported in the LANC 2026 paper:

"Impact of Client Participation on the Trade-Off Between Performance
and Computational Cost in Federated Learning."

The source code was derived from PFLlib, based on upstream commit:

0169ba7 - Merge pull request #257 from yucheng514/master

Local modifications relevant to the experiments include:

- FashionMNIST dataset generation;
- client sampling and server-side execution;
- execution and resource-monitoring scripts.

Generated datasets, trained model files, raw logs, checkpoints, and
temporary files are not included.

Original PFLlib project:
https://github.com/TsingZ0/PFLlib
