# Install nvcc
conda install nvidia::cuda-nvcc

# Download required pip packages
pip install mamba-ssm --no-build-isolation
pip install -r requirements.txt
pip install -r lichessbot/requirements.txt

# Set up python path for script importing
export PYTHONPATH=$(pwd)
