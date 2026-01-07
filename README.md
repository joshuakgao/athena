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

Create a copy of `config.yml.default` and name it `config.yml`. Then change your lichess token.
