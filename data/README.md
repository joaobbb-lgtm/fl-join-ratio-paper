# Experimental data

This folder contains the consolidated FashionMNIST experimental results used to generate the tables and figures reported in the paper.

The original raw logs were produced from five repetitions per configuration under:

- algorithms: FedAvg, FedProx, SCAFFOLD
- join ratios: 0.10, 0.25, 0.50, 0.75, 1.00
- clients: 100
- global rounds: 50
- local epochs: 5

The raw experiment directory was maintained locally at:

`~/research/experiments/fashionmnist_gpu`

Due to size and verbosity, raw logs are not included in this repository. The consolidated CSV files provide the values used in the analysis notebook, tables, and figures.
