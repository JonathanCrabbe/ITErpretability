import numpy as np
import pandas as pd
import pickle
from scipy.stats import entropy

from sklearn.decomposition import PCA


def normalize_data(X):
    X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    return X_normalized


def process_news(max_num_features):
    try:
        news_dataset = pickle.load(open("news_full_dataset.p", "rb"))
    except:
        raise FileNotFoundError(
            "Full News dataset needs to be downloaded from: https://drive.google.com/file/d/1Blw38cm-bU4iQhfu2F7q3IJcY6VvLzyy/view?usp=sharing"
        )

    pca = PCA(max_num_features, svd_solver="randomized")
    news_dataset = pca.fit_transform(news_dataset["data"])

    # Normalize data to [0, 1]
    news_dataset = normalize_data(news_dataset)

    pickle.dump(news_dataset, open("news_" + str(max_num_features) + ".p", "wb"))


if __name__ == "__main__":
    process_news(max_num_features=100)
