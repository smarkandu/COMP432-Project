import torch
import torch.nn.functional as F


def train(train_loader, model, loss, args, optimizer):
    losses = []
    accuracies = []
    model.train()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    for images, labels in train_loader:
        images = images.to("cuda:0")
        labels = labels.to("cuda:0")        # print('images.shape',images.shape)

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


def test(test_loader, model, loss, args):
    losses = []
    accuracies = []
    err = 0
    model.eval()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    for images, labels in test_loader:
        # Added
        images = images.to("cuda:0")
        labels = labels.to("cuda:0")
        outputs = model(images)
        l = loss(outputs, labels)
        losses.append(l.item())

        predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        accuracies.append(sum(predictions == labels).item() / args.batchsize)
        err = err + (predictions != labels).sum().item()

    # Prepare Output
    test_loss = torch.tensor(losses).mean()
    test_err = err / len(images)
    test_accuracy = torch.tensor(accuracies).mean()
    return test_loss, test_err, test_accuracy


def run_epochs(train_loader, test_loader, model, loss, args):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    for epoch in range(args.epochs):
        train_loss, train_accuracy = train(train_loader, model, loss, args, optimizer)
        test_loss, test_err, test_accuracy = test(test_loader, model, loss, args)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        if args.debug:
            print(
                'Epoch: {}  train_loss={:.4f}, test_loss={:.4f}, test_err={:.2f}%, train_accuracy={:.2f}%, '
                'test_accuracy={:.2f}%' \
                    .format(epoch + 1, train_loss, test_loss, test_err * 100, train_accuracy * 100, test_accuracy * 100))

    return train_losses, test_losses, train_accuracies, test_accuracies


# face_dataset = NPZ_DataSet(path='')
# from torch.utils.data import random_split
# generator2 = torch.Generator().manual_seed(42)
# train_data, val_data, test_data = random_split(face_dataset, [0.3, 0.3, 0.4], generator=generator2)
#
# print(train_data)
# from argparse import Namespace
# args = Namespace(batchsize=64, epochs=30, lr=0.001, weightdecay=0, eval=False,plot=True,debug=True)
#
# # Data loaders
# train_loader = torch.utils.data.DataLoader(dataset = train_data,
#                                              batch_size = args.batchsize,
#                                              shuffle = True)
#
# test_loader = torch.utils.data.DataLoader(dataset = test_data,
#                                       batch_size = args.batchsize,
#                                       shuffle = False)
#
#
# for images, labels in train_loader:
#     print(labels)
#
# # print(train_loader)
#
#
# train_losses, test_losses, train_accuracies, test_accuracies = run_epochs(train_loader, test_loader, model, loss, args)
# plot_graphs(train_losses, test_losses, train_accuracies, test_accuracies)
