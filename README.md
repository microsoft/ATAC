# ATAC
### <em>Adversarially Trained Actor Critic for Offline Reinforcement Learning </em>by Ching-An Cheng*, Tengyang Xie*, Nan Jiang, Alekh Agarwal
<br>

### Clone the repository and create a conda environment.
```
git clone https://github.com/microsoft/ATAC.git
conda create -n atac python=3.8
cd atac
```
### Prerequisite: Install Mujoco
(Optional) Install free mujoco210 for mujoco_py and mujoco211 for dm_control.
```
bash install_mujoco.sh
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chinganc/.mujoco/mujoco210/bin:/usr/lib/nvidia" >> ~/.bashrc
source ~/.bashrc
```
### Install ATAC
```
conda activate atac
pip install -e .[mujoco210]
# or below, if the original paid mujoco is used.
pip install -e .[mujoco200]
```
### Run ATAC
```
python scripts/main.py
```
=======
# Project

> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
>>>>>>> 6ba8f72040b4b20692941421140b07a1267159e4
