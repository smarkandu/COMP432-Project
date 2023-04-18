import matplotlib.pyplot as plt


def plot_graphs(train_losses, test_losses, train_accuracies, test_accuracies):
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
