sudo apt-get install libglew-dev patchelf
wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco-2.1.1-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
rm mujoco-2.1.1-linux-x86_64.tar.gz  mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv mujoco210 mujoco-2.1.1 ~/.mujoco