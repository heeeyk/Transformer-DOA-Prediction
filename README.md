# Transformer-DOA-Prediction

[![](https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arXiv&logoColor=red)](https://arxiv.org/abs/2308.01929)
[![](https://img.shields.io/badge/Paper-IEEE-blue?style=plastic&logo=adobeacrobatreader&logoColor=blue)](https://ieeexplore.ieee.org/abstract/document/10218321)

This repository is the official implementation of paper **A Transformer-based Prediction Method for Depth of Anesthesia During Target-controlled Infusion of Propofol  and Remifentanil.**






## Usage

The main requirements are pytorch 1.4.0 with python 3.9.1.

The [`mainer`](mainer) sets up a container with a main function for this project. Run ['main_featurefusion'](mainer/main_featurefusion.py) to begin training or testing.
The [`loader`](loader) deposit some programs to load drug and BIS record (which can access in [VitalDB](https://vitaldb.net/)). 


