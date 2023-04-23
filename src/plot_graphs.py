import matplotlib.pyplot as plt


def plot_graphs(train_losses, test_losses, train_accuracies, test_accuracies):
    """
    Creates two graphs: a loss vs epoch graph and an accuracy vs epoch graph
    Two curves for each graph: the training curve and test curve

    This was taken from the "Debug" lab:
    https://colab.research.google.com/drive/1_CaBwPTjCJPrH5ZZ6uA86v6FzFKu8anP?usp=share_link

    :param train_losses: list of losses for training
    :param test_losses: list of losses for testing
    :param train_accuracies: list of accuracies for training
    :param test_accuracies: list of accuracies for testing
    :return: None (just shows the graph)
    """
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
