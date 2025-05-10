# Download required pip packages
pip install -r requirements.txt
pip install -r lichessbot/requirements.txt

# Download rosbag
sudo apt install python3-rosbag

# Set up python path for script importing
export PYTHONPATH=$(pwd)
