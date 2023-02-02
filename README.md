# ATAC: Adversarially Trained Actor Critic

This repository contains the code to reproduce the experimental results of ATAC algorithm in the paper <em>Adversarially Trained Actor Critic for Offline Reinforcement Learning </em>by Ching-An Cheng*, Tengyang Xie*, Nan Jiang, and Alekh Agarwal (https://arxiv.org/abs/2202.02446). 

***Please see also https://github.com/microsoft/lightATAC for a lightweight reimplementation of ATAC, which gives a 1.5-2X speed up compared with the original code here.

### Setup 

#### Clone the repository and create a conda environment.
```
git clone https://github.com/microsoft/ATAC.git
conda create -n atac python=3.8
cd atac
```
#### Prerequisite: Install Mujoco
(Optional) Install free mujoco210 for mujoco_py and mujoco211 for dm_control.
```
bash install_mujoco.sh
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia" >> ~/.bashrc
source ~/.bashrc
```
#### Install ATAC
```
conda activate atac
pip install -e .[mujoco210]
# or below, if the original paid mujoco is used.
pip install -e .[mujoco200]
```
#### Run ATAC
```
python scripts/main.py
```

### Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
