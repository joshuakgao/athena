from tqdm import tqdm

from architectures.resnet import AthenaResnet
from datasets.chessbench.dataset import ChessbenchDataset

if __name__ == "__main__":
    athena = AthenaResnet(
        input_channels=24, width=256, num_blocks=19, K=128, M=16, device="cpu"
    )
    train_dataset = ChessbenchDataset("datasets/chessbench/data_mate", mode="train")
    for fen, move, win_prob, mate in tqdm(train_dataset, total=len(train_dataset)):
        try:
            tensor = athena.encode_win_prob(win_prob, mate, K=128, M=16)
        except Exception as e:
            print(fen, move, win_prob, mate)
            print(e)
