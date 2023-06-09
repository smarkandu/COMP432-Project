import numpy as np
import sklearn
import sklearn.preprocessing
import sklearn.neural_network
import torch


def get_data(no_of_bands):
    """
    Get input data and labels from NPZ file

    This is an original function

    :param no_of_bands: Number of bands for input
    :return: input data X, labels y
    """
    data_retrieval = 0

    if data_retrieval == 0:
        # !gdown 11Psi1ZZ2WJ-oHCG1UZcyjGQ1N1iFllDf
        path_for_npz = "data.npz"

    else:
        colab_notebook_root = "/content/drive/MyDrive/Colab Notebooks"
        path_for_npz = colab_notebook_root + "/COMP 432/Project/npz/data.npz"
        if data_retrieval == 1:
            path_for_npz = "G:/My Drive/Colab Notebooks/COMP 432/Project/npz/data.npz"
        else:
            from google.colab import drive
            drive.mount("/content/drive")

    with np.load(path_for_npz) as data:
        X = data['X']
        if no_of_bands == 1:
            X = X[:, :, :, 2]  # Get one channel
            X = X.reshape(-1, 1, 120, 120)
        else:
            X = X.reshape(-1, 3, 120, 120)

        # Normalize input from [0,255] to [0,1]
        X = X / 255.0

        y = data['y']
        # Encode labels to integers
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        return X, y


torch.manual_seed(0)  # Ensure model weights initialized with same random numbers

# Your code here. Aim for 8-11 lines.
drop_out_value = 0


# Split train and test data (aim for 1 line)
def get_and_split_data(train_size, no_of_bands=1):
    """
    This first gets the data and then splits it into training, test and validation sets
    It uses the train size parameter as a percentage for the training size and then for the remaining percentage,
    it divided that in half for the test and validation set.

    This is an original function.

    :param train_size: Percentage of data set to be split for training.  Remaining is split in half for test/val sets
    :param no_of_bands: Number of bands for input
    :return: Training set, validation set, test set, training labels, validation labels, test labels
    """
    X, y = get_data(no_of_bands)
    X_trn, X_tst, y_trn, y_tst = sklearn.model_selection.train_test_split(X, y, train_size=train_size, random_state=0)
    X_val, X_tst, y_val, y_tst = sklearn.model_selection.train_test_split(X_tst, y_tst, train_size=0.5, random_state=0)
    X_trn = torch.from_numpy(X_trn)
    X_val = torch.from_numpy(X_val)
    X_tst = torch.from_numpy(X_tst)
    y_trn = torch.from_numpy(y_trn)
    y_val = torch.from_numpy(y_val)
    y_tst = torch.from_numpy(y_tst)
    return X_trn, X_val, X_tst, y_trn, y_val, y_tst
