# Transformer-DOA-Prediction
**A Transformer-based Prediction Method for Depth of Anesthesia During Target-controlled Infusion of Propofol  and Remifentanil.**
[![](https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=red)](https://arxiv.org/abs/2308.01929)
[![](https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=blue)](https://ieeexplore.ieee.org/abstract/document/10218321)

https://ieeexplore.ieee.org/abstract/document/10218321

## Usage

The main requirements are pytorch 1.4.0 with python 3.9.1.

The [`mainer`](mainer) sets up a container with a main function for this project. Run ['main_featurefusion'](mainer/main_featurefusion.py) to begin training or testing.
The [`loader`](loader) deposit some programs to load drug and BIS record (which can access in [VitalDB](https://vitaldb.net/)). 


