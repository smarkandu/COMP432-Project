import torch

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
