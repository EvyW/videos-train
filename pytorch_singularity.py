import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

def evy(a,b):
    a = 3
    b = 5
    print(a+b)


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break




# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")




classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
















import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy
import torch

def data_to_tensor_pair(data, device):
    x = torch.tensor([x for x, y in data], device=device)
    y = torch.tensor([y for x, y in data], device=device)
    return x, y

def evaluate_model(model, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        return compute_accuracy(y_pred, y)

def compute_accuracy(predictions, expected):
    correct = 0
    total = 0
    for y_pred, y in zip(predictions, expected):
        correct += round(y_pred.item()) == round(y.item())
        total += 1
    return correct / total

def construct_model(hidden_units, num_layers):
    layers = []
    prev_layer_size = 2
    for layer_no in range(num_layers):
        layers.extend([
            torch.nn.Linear(prev_layer_size, hidden_units),
            torch.nn.Tanh()
        ])
        prev_layer_size = hidden_units
    layers.extend([
        torch.nn.Linear(prev_layer_size, 1),
        torch.nn.Sigmoid()
    ])
    return torch.nn.Sequential(*layers)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1.0)
    parser.add_argument('--output')
    args = parser.parse_args()

    if torch.cuda.is_available():
        print('CUDA is available -- using GPU')
        device = torch.device('cuda')
    else:
        print('CUDA is NOT available -- using CPU')
        device = torch.device('cpu')

    # Define our toy training set for the XOR function.
    training_data = data_to_tensor_pair([
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0])
    ], device)

    # Define our model. Use default initialization.
    model = construct_model(hidden_units=10, num_layers=2)
    model.to(device)

    loss_values = []
    accuracy_values = []
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    for iter_no in range(args.iterations):
        print('iteration #{}'.format(iter_no + 1))
        # Perform a parameter update.
        model.train()
        optimizer.zero_grad()
        x, y = training_data
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss_value = loss.item()
        print('  loss: {}'.format(loss_value))
        loss_values.append(loss_value)
        loss.backward()
        optimizer.step()
        # Evaluate the model.
        accuracy = evaluate_model(model, x, y)
        print('  accuracy: {:.2%}'.format(accuracy))
        accuracy_values.append(accuracy)

    if args.output is not None:
        print('saving model to {}'.format(args.output))
        torch.save(model.state_dict(), args.output)

    # Plot loss and accuracy.
    fig, ax = plt.subplots()
    ax.set_title('Loss and Accuracy vs. Iterations')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.set_xlim(left=1, right=len(loss_values))
    ax.set_ylim(bottom=0.0, auto=None)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_array = numpy.arange(1, len(loss_values) + 1)
    loss_y_array = numpy.array(loss_values)
    left_plot = ax.plot(x_array, loss_y_array, '-', label='Loss')
    right_ax = ax.twinx()
    right_ax.set_ylabel('Accuracy')
    right_ax.set_ylim(bottom=0.0, top=1.0)
    accuracy_y_array = numpy.array(accuracy_values)
    right_plot = right_ax.plot(x_array, accuracy_y_array, '--', label='Accuracy')
    lines = left_plot + right_plot
    ax.legend(lines, [line.get_label() for line in lines])
    plt.show()



if __name__ == '__main__':
    main()




