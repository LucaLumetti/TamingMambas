# Mamba 3D Medical Image Segmentation
This repository is a collection of mamba-based and U-shaped models tailored for the segmentation of medical images. It provides a collections of state-of-the-art models we used to compare with for different related publications.


## Installation 

Requirements: `Ubuntu 20.04`, `CUDA 11.8`

1. Create a virtual environment: `conda create -n tamingmambas python=3.10 -y` and `conda activate tamingmambas `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.0.1: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
3. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d>=1.2.0` and `pip install mamba-ssm --no-cache-dir`
4. Download code: `git clone https://github.com/LucaLumetti/TamingMambas.git`
5. `cd TamingMambas/tamingmambas ` and run `pip install -e .`


Sanity check: Enter the Python command-line interface and run:

```python
import torch
import mamba_ssm
```

If you face problems with Mamba or causal-conv1d, try to install them manually.

## Cite Us
If you find this project useful for your research or development, please consider citing it:

[Accurate 3D Medical Image Segmentation with Mambas](https://iris.unimore.it/handle/11380/1367190).
```
@inproceedings{lumetti2025accurate,
  title={Accurate 3D Medical Image Segmentation with Mambas},
  author={Lumetti, Luca and Pipoli, Vittorio and Marchesini, Kevin and Ficarra, Elisa and Grana, Costantino and Bolelli, Federico and others},
  booktitle={Proceedings of 2025 IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2025}
}
```

[Taming Mambas for Voxel Level 3D Medical Image Segmentation](https://arxiv.org/abs/2410.15496).
```
@article{lumetti2024taming,
  title={Taming Mambas for Voxel Level 3D Medical Image Segmentation},
  author={Lumetti, Luca and Pipoli, Vittorio and Marchesini, Kevin and Ficarra, Elisa and Grana, Costantino and Bolelli, Federico},
  journal={arXiv preprint arXiv:2410.15496},
  year={2024}
}
```



## Acknowledgements

We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. We also thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba) and [U-Mamba](https://github.com/bowang-lab/U-Mamba) for making their valuable code publicly available.

