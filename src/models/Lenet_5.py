import torch

lenet5_model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 6, kernel_size=12, stride=4),
    torch.nn.Tanh(),
    torch.nn.AvgPool2d(kernel_size=2, stride=2),

    torch.nn.Conv2d(8, 16, kernel_size=5),
    torch.nn.Tanh(),
    torch.nn.AvgPool2d(kernel_size=2, stride=2),

    torch.nn.Conv2d(16, 120, kernel_size=5),

    torch.nn.Linear(120, 84),
    torch.nn.Tanh(),
    torch.nn.Linear(84, 15)
)
