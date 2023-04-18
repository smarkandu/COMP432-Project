import argparse
import numpy as np
import sklearn
import sklearn.preprocessing
import sklearn.neural_network
import torch
import copy
import torch.nn.functional as F
from orion.client import report_objective  # Orion


def get_data():
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
        X = X[:, :, :, 2]  # Get one channel
        X = X.reshape(-1, 1, 120, 120)
        X = X / 255.0  # Normalize data

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

plain_cnn_model = torch.nn.Sequential(  # 120 X 120 X 1
    torch.nn.Conv2d(1, 8, kernel_size=5, padding=2),  # 120 X 120 X ?
    # torch.nn.Dropout(drop_out_value),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 60 X 60 X ?

    torch.nn.Conv2d(8, 16, kernel_size=5, padding=2),  # 60 X 60 X ?
    # torch.nn.Dropout(drop_out_value),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 30 X 30 X ?

    torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),  # 30 X 30 X ?
    # torch.nn.Dropout(drop_out_value),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 15 X 15 X ?

    torch.nn.Conv2d(32, 64, kernel_size=5, padding=2),  # 15 X 15 X ?
    # torch.nn.Dropout(drop_out_value),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 7 X 7 X ?

    torch.nn.Conv2d(64, 128, kernel_size=5, padding=2),  # 15 X 15 X ?
    # torch.nn.Dropout(drop_out_value),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 7 X 7 X ?

    torch.nn.Flatten(1, -1),
    torch.nn.Linear(1152, 15)
)


# Split train and test data (aim for 1 line)
def get_and_split_data(train_size):
    X, y = get_data()
    X_trn, X_tst, y_trn, y_tst = sklearn.model_selection.train_test_split(X, y, train_size=train_size, random_state=0)
    X_val, X_tst, y_val, y_tst = sklearn.model_selection.train_test_split(X_tst, y_tst, train_size=0.5, random_state=0)
    X_trn = torch.from_numpy(X_trn)
    X_val = torch.from_numpy(X_val)
    X_tst = torch.from_numpy(X_tst)
    y_trn = torch.from_numpy(y_trn)
    y_val = torch.from_numpy(y_val)
    y_tst = torch.from_numpy(y_tst)
    return X_trn, X_val, X_tst, y_trn, y_val, y_tst


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


def train(X_trn, y_trn, model, loss, args, optimizer):
    losses = []
    accuracies = []
    model.train()

    for i in range(0, len(X_trn), args.batchsize):
        images = X_trn[i:i + args.batchsize].to(device)
        labels = y_trn[i:i + args.batchsize].to(device)
        # print('images.shape',images.shape)
        outputs = model(images)
        l = loss(outputs, labels)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(l.item())

        # Append Accuracy
        predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        accuracies.append(sum(predictions == labels).item() / args.batchsize)
    return torch.tensor(losses).mean(), torch.tensor(accuracies).mean()


def test(X_tst, y_tst, model, loss, args):
    losses = []
    accuracies = []
    err = 0
    model.eval()
    for i in range(0, len(X_tst), args.batchsize):
        images = X_tst[i:i + args.batchsize].to(device)
        labels = y_tst[i:i + args.batchsize].to(device)
        outputs = model(images)
        l = loss(outputs, labels)
        losses.append(l.item())

        predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        accuracies.append(sum(predictions == labels).item() / args.batchsize)
        err = err + (predictions != labels).sum().item()
    return torch.tensor(losses).mean(), err / len(X_tst), torch.tensor(accuracies).mean()


def plot_epochs(X_trn, y_trn, X_tst, y_tst, model, loss, args, plot=False):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    for epoch in range(args.epochs):
        train_loss, train_accuracy = train(X_trn, y_trn, model, loss, args, optimizer)
        test_loss, test_err, test_accuracy = test(X_tst, y_tst, model, loss, args)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print(
            'Epoch: {}  train_loss={:.4f}, test_loss={:.4f}, test_err={:.2f}%, train_accuracy={:.2f}%, test_accuracy={:.2f}%' \
            .format(epoch + 1, train_loss, test_loss, test_err * 100, train_accuracy * 100, test_accuracy * 100))

    if plot == True:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, '-s', label='train')
        plt.plot(test_losses, '-s', label='test')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, '-s', label='train')
        plt.plot(test_accuracies, '-s', label='test')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()

    return train_accuracies[-1], test_accuracies[-1]


def orion_train():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 1.0)')
    parser.add_argument('--neurons', type=int, default=100,
                        help='number of neurons (default: 100)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='If True it prints the test error (default: False)')
    parser.add_argument('--weightdecay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument("-f", required=False)
    args = parser.parse_args()
    print(args)

    # Select data
    X_trn, X_val, X_tst, y_trn, y_val, y_tst = get_and_split_data(0.70)
    model = copy.deepcopy(plain_cnn_model).to(device)

    # Your code for defining loss, optimizer, and training loop here. Aim for 10-12 lines.
    loss = torch.nn.CrossEntropyLoss()
    train_score, val_score = plot_epochs(X_trn, y_trn, X_val, y_val, model, loss, args)
    valid_error = (100 * (1 - val_score)).item()
    print("Valid Error (\%): " + str(valid_error))

    report_objective(valid_error)

    # if args.eval:
    #   test_error = 100*(1 - model.score(X_tst, y_tst))
    #   print("Test Error (\%): " + str(test_error))


if __name__ == '__main__':
    orion_train()