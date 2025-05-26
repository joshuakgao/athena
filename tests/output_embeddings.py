from embeddings import encode_win_prob
from datasets.chessbench.dataset import ChessbenchDataset
from tqdm import tqdm


if __name__ == "__main__":
    # tensor = encode_win_prob(0, -99, K=11, M=3)
    # print(tensor)
    # print(tensor.argmax())

    train_dataset = ChessbenchDataset("datasets/chessbench/data_mate", mode="train")
    for fen, move, win_prob, mate in tqdm(train_dataset, total=len(train_dataset)):
        try:
            tensor = encode_win_prob(win_prob, mate, K=128, M=16)
        except Exception as e:
            print(fen, move, win_prob, mate)
            print(e)
