# Installation

## Tested environment
- Python 3.8.5
- NumPy 1.20.3
- OpenAI Gym 0.17.3
- stable-baselines3 1.3.0
- mpi4py 3.0.3
- MuJoCo 2.1.0
- mujoco-py 2.1.2.14
- (Ubuntu 20.04 & mpich) or (Windows 10/11 & Microsoft MPI 10.0.12498.5)

## Ubuntu 20.04
```bash
sudo apt install -y \
    libffi-dev libssl-dev zlib1g-dev liblzma-dev tk-dev \
    libbz2-dev libreadline-dev libsqlite3-dev libopencv-dev \
    libosmesa6-dev patchelf\
    build-essential git mpich

# Install mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -zxvf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco/mujoco210
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"' >> ~/.bashrc

# Install pyenv
git clone https://github.com/pyenv/pyenv.git ~/.pyenv

cd ~/.pyenv
git checkout v2.0.3 # (optional) Switch pyenv version
cd ..

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'export MUJOCO_PY_FORCE_CPU=1' >> ~/.bashrc
source ~/.bashrc

# Activate the environment
pyenv install 3.8.5
pyenv local 3.8.5

# Install dependencies
pip install mpi4py==3.0.3 \
    numpy==1.20.3 \
    gym==0.17.3 \
    box2d==2.3.10 \
    stable-baselines3==1.3.0 \
    Levenshtein==0.16.0 \
    opencv-python==4.5.5.64 \
    mujoco-py==2.1.2.14 \
    notebook

# Clone
git clone https://github.com/r-koike/eagent
```

- To enable training, follow these steps:
  - Download 2 files: code_****_18_1.pkl available in the [release note](https://github.com/r-koike/eagent)
  - Move them into `eagent/data/code_****_18_1.pkl`
  - These files required when `"max_num_limbs": 18`. If these files do not exist, they are automatically created, but this process can be very time-consuming.
- To check trained data, access the [release note](https://github.com/r-koike/eagent)

## Windows 10/11
This instruction uses PowerShell. Anaconda and other environments other than pyenv are also fine.

### Install python
```powershell
# Install pyenv
git clone https://github.com/pyenv-win/pyenv-win.git "$HOME/.pyenv"

# Add following environment variables:
# PYENV: %USERPROFILE%\.pyenv\pyenv-win
# PATH: %USERPROFILE%\.pyenv\pyenv-win\bin
# PATH: %USERPROFILE%\.pyenv\pyenv-win\shims

# Activate the environment
pyenv install 3.8.5
pyenv local 3.8.5

# Install dependencies
pip install mpi4py==3.0.3 `
    numpy==1.20.3 `
    gym==0.17.3 `
    box2d==2.3.10 `
    stable-baselines3==1.3.0 `
    Levenshtein==0.16.0 `
    opencv-python==4.5.5.64 `
    notebook

# Clone
git clone https://github.com/r-koike/eagent
```

### Install MuJoCo & mujoco-py
1. Download [MuJoCo 2.1.0](https://github.com/deepmind/mujoco/releases/tag/2.1.0)
2. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`
3. Add `%USERPROFILE%\.mujoco\mujoco210\bin` to PATH
4. To install mujoco-py, follow these steps:
```powershell
git clone -b v2.1.2.14 https://github.com/openai/mujoco-py
cd mujoco-py

# Replace isinstance(addr, (int, np.int32, np.int64)) -> hasattr(addr, '__int__')
# For more information, see https://github.com/openai/mujoco-py/issues/504#issuecomment-621183589
$data = Get-Content mujoco_py/generated/wrappers.pxi | ForEach-Object { $_ -creplace "isinstance\(addr, \(int, np.int32, np.int64\)\)", "hasattr(addr, '__int__')" }
$data | Out-File mujoco_py/generated/wrappers.pxi -Encoding utf8

# Insert the following line into mujoco_py/builder.py
# os.add_dll_directory("C:/Users/[your_username]/.mujoco/mujoco210/bin")
# For more information, see https://github.com/openai/mujoco-py/issues/638#issuecomment-969019281

# It is recommended to install for CPU.
# If you already have CUDA installed, add following environment variables:
# MUJOCO_PY_FORCE_CPU: 1

pip install -e .
```

### Install othor tools
- To enable training, install [Microsoft MPI (10.0.12498.5)](https://www.microsoft.com/en-us/download/details.aspx?id=57467)
  - To confirm that the installed MPI is being used, execute the following command: `gcm mpiexec | fl`
- To enable training, follow these steps:
  - Download 2 files: code_****_18_1.pkl available in the [release note](https://github.com/r-koike/eagent)
  - Move them into `eagent/data/code_****_18_1.pkl`
  - These files required when `"max_num_limbs": 18`. If these files do not exist, they are automatically created, but this process can be very time-consuming.
- To enable video recording, install [FFmpeg](https://ffmpeg.org/) and add bin directory to PATH
- To check trained data, access the [release note](https://github.com/r-koike/eagent)
