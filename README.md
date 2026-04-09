# Athena

My attempt at a deep learning chess bot.

## Quickstart

```bash
uv sync
source .venv/bin/activate
cd src/athena/datasets
hf download joshuakgao/chessbenchmate --repo-type dataset --local-dir . --max-workers 1
cd ../../..
python src/athena/train.py
```

## Demo on Lichess

Create a copy of `config.yml.default` and name it `config.yml`. Then change your lichess token. Then in "lichessbot/homemade.py", adjust the Athena model config to fit your model parameters and update the model path.

```bash
# Download pretrained Athena weights
mkdir src/athena/checkpoints
wget https://huggingface.co/joshuakgao/athena-chess/resolve/main/athena.pt
mv athena.pt src/athena/checkpoints

# Host model on Lichess
python lichessbot/lichess-bot.py
```
