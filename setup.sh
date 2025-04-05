# Download required pip packages
pip install -r requirements.txt
pip install -r lichessbot/requirements.txt

# # Install stockfish
# OS=$(uname -s) # Get the OS name
# if [ "$OS" = "Linux" ]; then
#     sudo apt install stockfish
# elif [ "$OS" = "Darwin" ]; then
#     brew install stockfish
# else
#     echo "This is an unknown or unsupported system: $OS"
# fi

# Install p7zip
# Install stockfish
OS=$(uname -s) # Get the OS name
if [ "$OS" = "Linux" ]; then
    sudo apt install p7zip-full
elif [ "$OS" = "Darwin" ]; then
    brew install p7zip
else
    echo "This is an unknown or unsupported system: $OS"
fi

# Download lichess pgn chess game databases
# You can go into download.sh to change which size datasets you want
bash datasets/lichess/download.sh
bash datasets/aegis/download.sh