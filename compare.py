import time

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
    [transforms.ToTensor()
        #,transforms.RandomHorizontalFlip(0.5)
     ]
)

num_workers = 0

batch_size = 4

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

cuda = torch.cuda.is_available()

if not cuda:
    raise AssertionError("Cannot find CUDA environment to compare!")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def validate(test_loader, model, environment):
    start_time = time.time()
    with torch.no_grad():
        match = total = 0
        for data in tqdm.tqdm(test_loader):
            images, labels = data
            images = images.to(environment)
            labels = labels.to(environment)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            match += (predicted == labels).sum().item()
        accuracy = match / total
    time_consumed = time.time() - start_time
    return accuracy, time_consumed

def run_training(environment='cpu', seed=42, max_iteration=800,
                 target_acc=0.85, validate_every=4000, model_closure=None):
    if model_closure is None:
        model = Net()  # just uses example net from tutorial.
    else:
        model = model_closure()
    torch.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True if environment != 'cpu' else False, pin_memory_device=environment if environment != 'cpu' else "")
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True if environment != 'cpu' else False, pin_memory_device=environment if environment != 'cpu' else "")
    st = time.time()
    _i = 0
    accuracy = 0
    validation_time_consumed = 0
    with torch.autocast(environment):
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=model.parameters(), lr=0.001)
        model.to(environment)
        pbar = tqdm.tqdm(total=4000 * len(train_loader))
        for epoch in range(4000):
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(environment)
                labels = labels.to(environment)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar.update()
                _i += 1
                if _i % validate_every == 0:
                    # validate accuracy after epoch
                    accuracy, validation_time = validate(test_loader, model, environment)
                    validation_time_consumed += validation_time
                    elapsed = time.time() - st - validation_time_consumed
                    if accuracy >= target_acc:
                        print(
                            f"Finished test for environment {environment} with accuracy {accuracy} in {elapsed}s (avg speed : {_i / elapsed}/s)")
                        return
                if _i >= max_iteration:
                    if accuracy == 0:
                        accuracy, validation_time = validate(test_loader, model, environment)
                        validation_time_consumed += validation_time
                    elapsed = time.time() - st - validation_time_consumed
                    print(
                        f"Finished with final accuracy {accuracy} in {elapsed}s in environment {environment} (avg speed : {_i / elapsed}/s)")
                    return
        if accuracy == 0:
            accuracy, validation_time = validate(test_loader, model, environment)
            validation_time_consumed += validation_time
            elapsed = time.time() - st - validation_time_consumed
            print(
                f"Finished test for environment {environment} with final accuracy {accuracy} in {elapsed}s (avg speed : {_i / elapsed}/s)")
            return
    elapsed = time.time() - st - validation_time_consumed
    print(f"Finished with final accuracy {accuracy} in {elapsed}s in environment {environment} (avg speed : {_i / elapsed}/s)")


for env in ('cuda', 'cpu'):
    run_training(env)
    # maybe use model_closure=torchvision.models.efficientnet_b0
