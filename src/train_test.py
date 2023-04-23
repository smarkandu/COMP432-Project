import torch
import torch.nn.functional as F


def train(X_trn, y_trn, model, loss, args, optimizer):
    """
    Training loop for model

    Partially taken from the "Debug" Week 5 tutorial.

    But had the following modifications made:
    - Doesn't use a dataloader
    - Accepts any data set (was hardcoded before to MNIST)
    - Added mean training accuracy as a return value
    - Arguments parameter for any "customization" to the test loop

    :param X_trn: Training data
    :param y_trn: Training labels
    :param model: Model to be trained
    :param loss: loss object
    :param args: arguments for the hyperparameters, etc.
    :param optimizer: optimizer object to be used in training loop
    :return: mean training losses, mean training accuracies
    """

    losses = []
    accuracies = []
    model.train()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

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
    """
    Test loop for model

    Partially taken from the "Debug" Week 5 tutorial.

    But had the following modifications made:
    - Doesn't use a dataloader
    - Accepts any data set (was hardcoded before to MNIST)
    - Added mean test accuracy as a return value
    - Arguments parameter for any "customization" to the test loop

    :param X_tst: Test data (new)
    :param y_tst: Test labels (new)
    :param model: Model to use for testing
    :param loss: loss object
    :param args: arguments for the hyperparameters, etc
    :return: mean test loss, mean test error, mean test_accuracy
    """
    losses = []
    accuracies = []
    err = 0
    model.eval()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    for i in range(0, len(X_tst), args.batchsize):
        images = X_tst[i:i + args.batchsize].to(device)
        labels = y_tst[i:i + args.batchsize].to(device)
        outputs = model(images)
        l = loss(outputs, labels)
        losses.append(l.item())

        predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        accuracies.append(sum(predictions == labels).item() / args.batchsize)
        err = err + (predictions != labels).sum().item()

    # Prepare Output
    test_loss = torch.tensor(losses).mean()
    test_err = err / len(X_tst)
    test_accuracy = torch.tensor(accuracies).mean()
    return test_loss, test_err, test_accuracy


def plot_epochs(X_trn, y_trn, X_tst, y_tst, model, loss, args):
    """
    Trains model then runs model on test data.

    Partially taken from the "Debug" Week 5 tutorial.

    Modified such that:
     - it can be called via Orion
     - Plotting can be enabled/disabled
     - It now can plot the accuracy graph as well (not just the loss graph)
     - It sets the optimizer per the values from the "args" parameter (necessary for Orion)

    :param X_trn: Training data
    :param y_trn: Training labels
    :param X_tst: Test data
    :param y_tst: Test labels
    :param model: model to train
    :param loss: loss function
    :param args: arguments for the hyperparameters, whether to print graph, etc.
    :return: train losses, test losses, train accuracies, test accuracies
    """

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # if args.model == 2: optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weightdecay,
    # momentum=args.momentum) else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    for epoch in range(args.epochs):
        train_loss, train_accuracy = train(X_trn, y_trn, model, loss, args, optimizer)
        test_loss, test_err, test_accuracy = test(X_tst, y_tst, model, loss, args)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        if args.debug:
            print(
                'Epoch: {}  train_loss={:.4f}, test_loss={:.4f}, test_err={:.2f}%, train_accuracy={:.2f}%, '
                'test_accuracy={:.2f}%' \
                    .format(epoch + 1, train_loss, test_loss, test_err * 100, train_accuracy * 100, test_accuracy * 100))

    # Test on entire training set
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    X_tst = X_tst.to(device)
    y_tst = y_tst.to(device)
    print("Loss on entire test set: %.4f" % (loss(model(X_tst), y_tst)))

    return train_losses, test_losses, train_accuracies, test_accuracies
