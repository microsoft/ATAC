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
