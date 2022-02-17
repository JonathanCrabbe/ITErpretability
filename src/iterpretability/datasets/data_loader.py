import pickle


def load(dataset_name):
    if 'tcga' in dataset_name:
        tcga_dataset = pickle.load(open('src/iterpretability/datasets/tcga/' + str(dataset_name) + '.p', 'rb'))
        X_raw = tcga_dataset['rnaseq']
    else:
        print('Unknown dataset ' + str(dataset_name))

    return X_raw
