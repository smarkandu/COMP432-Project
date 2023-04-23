import argparse
import torch
import copy
from orion.client import report_objective  # Orion
from models.plain_cnn import plain_cnn_model
from models.Lenet_5 import lenet5_model
from models.alex_net import alex_net_model
from preprocessing.data_preprocessing import get_and_split_data
from train_test import plot_epochs

torch.manual_seed(0)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


def orion_train():
    """
        Used to initiate the Orion hyperparameter search

        Partially taken from the hyperparameter tuning lab (Orion).

        But heavily modified such that:
        - It uses pytorch instead
        - linked to the necessary functions via import statements (test/training loops, graphing, etc)
        - Added / removed arguments per what I needed
        - Runs for the models of this project (determined by the new "model" parameter)

    :return: Nothing is returned (all print statements)
    """
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 1.0)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='If True it prints the test error (default: False)')
    parser.add_argument('--weightdecay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--debug', type=bool, default=False,
                        help='If True it prints the debug traces (default: False)')
    parser.add_argument('--model', type=int, default=-1,
                        help='Enter # of model to test (default: -1)')
    # parser.add_argument('--momentum', type=float, default=0,
    #                     help='Enter value of momentum (default: 0)')
    args = parser.parse_args()
    print(args)

    # Select data

    if args.model == 0:
        print("plain_cnn selected")
        X_trn, X_val, X_tst, y_trn, y_val, y_tst = get_and_split_data(0.70)
        model = copy.deepcopy(plain_cnn_model).to(device)
    elif args.model == 1:
        print("lenet-5 selected")
        X_trn, X_val, X_tst, y_trn, y_val, y_tst = get_and_split_data(0.70)
        model = copy.deepcopy(lenet5_model).to(device)
    elif args.model == 2:
        print("AlexNet selected")
        X_trn, X_val, X_tst, y_trn, y_val, y_tst = get_and_split_data(0.70, 3)
        model = copy.deepcopy(alex_net_model).to(device)
    else:
        raise Exception("Error: Model type not recognized!")

    # Your code for defining loss, optimizer, and training loop here. Aim for 10-12 lines.
    loss = torch.nn.CrossEntropyLoss()

    # Call plot_epochs
    _, _, _, test_accuracies = plot_epochs(X_trn, y_trn, X_val, y_val, model, loss, args)

    val_score = test_accuracies[-1]
    valid_error = (100 * (1 - val_score)).item()
    print("Valid Error (\%): " + str(valid_error))

    report_objective(valid_error)

    # if args.eval:
    #   test_error = 100*(1 - model.score(X_tst, y_tst))
    #   print("Test Error (\%): " + str(test_error))


if __name__ == '__main__':
    orion_train()
