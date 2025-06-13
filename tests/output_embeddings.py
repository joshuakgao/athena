from tqdm import tqdm

from architectures.resnet import AthenaResnet
from datasets.chessbench.dataset import ChessbenchDataset
import traceback

if __name__ == "__main__":
    athena = AthenaResnet(
        input_channels=24, width=256, num_blocks=19, K=64, M=16, device="cpu"
    )
    train_dataset = ChessbenchDataset("datasets/chessbench/data_mate", mode="train")
    for fen, move, win_prob, mate in tqdm(train_dataset, total=len(train_dataset)):
        try:
            tensor = athena.encode_win_prob(win_prob, mate)
            prob, m = athena.decode_win_prob_bins(tensor)
            if isinstance(m, int):
                mate_clamped = max(-athena.M, min(athena.M, mate))
            else:
                mate_clamped = m
            assert m == mate_clamped, f"Mate mismatch: {m} != {mate_clamped}"
            assert (
                abs(prob - win_prob) < 0.01
            ), f"Win prob mismatch: {prob} != {win_prob}"
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing {fen} {move} {win_prob} {mate}: {e}")
            continue
