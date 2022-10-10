import pickle
from catenets.datasets import load as catenets_load


def load(dataset_name: str, train_ratio: float = 1.0):
    if "tcga" in dataset_name:
        tcga_dataset = pickle.load(
            open("src/iterpretability/datasets/tcga/" + str(dataset_name) + ".p", "rb")
        )
        X_raw = tcga_dataset["rnaseq"]
    elif "news" in dataset_name:
        news_dataset = pickle.load(
            open("src/iterpretability/datasets/news/" + str(dataset_name) + ".p", "rb")
        )
        X_raw = news_dataset
    elif "twins" in dataset_name:
        # Total features  = 39
        X_raw, _, _, _, _, _ = catenets_load(dataset_name, train_ratio=1.0)
    elif "acic" in dataset_name:
        # Total features  = 55
        X_raw, _, _, _, _, _, _, _ = catenets_load("acic2016")
    else:
        print("Unknown dataset " + str(dataset_name))

    if train_ratio == 1.0:
        return X_raw
    else:
        X_raw_train = X_raw[: int(train_ratio * X_raw.shape[0])]
        X_raw_test = X_raw[int(train_ratio * X_raw.shape[0]) :]
        return X_raw_train, X_raw_test
