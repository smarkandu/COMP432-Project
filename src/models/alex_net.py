import torch

alex_net_model = torch.nn.Sequential(
    # Note: Conv2d modified from the usual first layer of AlexNet
    # where kernel_size=11 and stride=4 in order to obtain an appropriate feature map size of 55X55X96
    torch.nn.Conv2d(3, 96, kernel_size=12, stride=2),
    torch.nn.BatchNorm2d(96),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=3, stride=2),

    torch.nn.Conv2d(96, 256, kernel_size=5, padding=2),
    torch.nn.BatchNorm2d(256),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=3, stride=2),

    torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(384),
    torch.nn.ReLU(),
    torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(384),
    torch.nn.ReLU(),
    torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(256),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=3, stride=2),
    torch.nn.Dropout(0.5),

    torch.nn.Flatten(1, -1),
    torch.nn.Linear(9216, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Linear(4096, 15)
)
