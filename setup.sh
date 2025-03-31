# Download required pip packages
pip install -r requirements.txt
pip install -r lichess-bot/requirements.txt

# Install stockfish
OS=$(uname -s) # Get the OS name
if [ "$OS" = "Linux" ]; then
    sudo apt install stockfish
elif [ "$OS" = "Darwin" ]; then
    brew stockfish
else
    echo "This is an unknown or unsupported system: $OS"
fi

# Download lichess pgn chess game databases
# You can go into download.sh to change which size datasets you want
bash datasets/lichess/download.sh
echo "Parsing chess database. This may take a while..."
python -m datasets.lichess.generate