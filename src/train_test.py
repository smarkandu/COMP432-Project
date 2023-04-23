import torch
import torch.nn.functional as F


def train(X_trn, y_trn, model, loss, args, optimizer):
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
    print("Loss on entire test set: %.4f" % (epoch, loss(model(X_tst), y_tst)))

    return train_losses, test_losses, train_accuracies, test_accuracies
