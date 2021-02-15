# NETS
Neural Embeddings in Team Sports and Application to Basketball Play Analysis

This is the code base for the paper "Neural Embeddings in Team Sports and Application to Basketball Play Analysis", currently under review for KDD 2021. It is based on pytorch with a single GPU, programmed and tested on Windows (check requirements.txt for further dependencies). To modify the hyper-parameters, modify the yaml files.

We provide the data for only one game, but the entire dataset can be found at https://github.com/sealneaward/nba-movement-data.

The entire pipeline can be executed by running main.py, which pretraines a model based on trajectory prediction (self-supervised), extracts weak labels for pick-and-roll and handoffs, and then finetunes the model based on the 3-way classification task pick-and-roll vs. handoff vs. other.
