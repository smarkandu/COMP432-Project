import argparse
import torch
import copy
from orion.client import report_objective  # Orion
from models.plain_cnn import plain_cnn_model
from preprocessing.data_preprocessing import get_and_split_data
from train_test import plot_epochs

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


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
    parser.add_argument('--plot', type=bool, default=False,
                        help='If True it plots the metric')
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
