import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, io
import os
import random

class DemoDataset(Dataset):
    """
    Dataset class collecting random frames from the dataset.
    """

    def __init__(self, root_dir: str, samples: int = 10, transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.labels = [0, 1]*samples
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        frame1_name = "robot" if random.random() > 0.5 else "mandarin"
        folder_name = os.path.join(self.root_dir, frame1_name)        
        file_name = random.choice(os.listdir(folder_name))

        frame1 = io.read_image(os.path.join(folder_name, file_name)) / 255

        is_change = bool(label)
        frame2_name = "robot" if (frame1_name == "robot" and not is_change) or (frame1_name == "mandarin" and is_change) else "mandarin"
        folder_name = os.path.join(self.root_dir, frame2_name)        
        file_name = random.choice(os.listdir(folder_name))

        frame2 = io.read_image(os.path.join(folder_name, file_name)) / 255
        if self.transform:
            frame2 = self.transform(frame2)
            frame1 = self.transform(frame1)
            frames = torch.Tensor(2, *frame1.shape)
            frames[0] = frame1
            frames[1] = frame2
        else:
            frames = [frame1, frame2]
        label = torch.tensor([label, not label], dtype=float)
        return frames, label


class CNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16*16*16, 50)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = nn.functional.relu(x)

        return x

class Net(nn.Module):

    def __init__(self, cnn, rnn) -> None:
        super().__init__()
        self.cnn = cnn
        self.rnn = rnn
        self.classifier = nn.Linear(20, 2)
    
    def forward(self, x):
        in_shape = x.shape[:2]
        x = torch.flatten(x, 0, 1)
        x = self.cnn(x)
        x = torch.reshape(x, (*in_shape, *x.shape[1:]))
        x, _ = self.rnn(x)
        x = self.classifier(x[:, -1, :])
        x = nn.functional.softmax(x, dim=-1)

        return x


class DemoModel():

    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Normalize([0.4, 0.4, 0.4], [0.2, 0.2, 0.2])
        ])
        
        # Init CNN-RNN network
        cnn = CNN()
        rnn = nn.LSTM(50, 20, 1, batch_first=True)
        self.net = Net(cnn, rnn)

        # Define loss and optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def train(self, data_path="./cnn-rnn-demo-frames/train"):
        # Init dataset loader
        demo_data = DemoDataset(data_path, samples=20, transform=self.transform)
        demo_loader = DataLoader(demo_data, batch_size=40)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(device)

        hist = []

        # Train the model
        epochs = 100
        for epoch in range(epochs):
            running_loss = 0.
            for i, data in enumerate(demo_loader, 0):
                
                # Send data to GPU
                inputs, labels = data
                inputs.to(device)
                labels.to(device)

                # Zero grad
                self.net.zero_grad()

                # Perfrom forward and backward pass
                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                # Print loss
                running_loss += loss
                print(f"Epoch {epoch+1}/{epochs}: [{i+1}/{len(demo_loader)}] loss: {running_loss/(i+1)}", end="\r")

            hist.append(running_loss)
            print()
        
        return hist
        

    def test(self, data_path="./cnn-rnn-demo-frames/test"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Init dataset loader
        test_data = DemoDataset(data_path, samples=20, transform=self.transform)
        test_loader = DataLoader(test_data, batch_size=40)

        inputs, labels = next(iter(test_loader))
        inputs.to(device)
        labels.to(device)

        with torch.no_grad():
            outs = self.net(inputs)
            loss = self.loss_function(outs, labels)
            preds = torch.argmax(outs, 1)
            labels = torch.argmax(labels, 1)
        
        return outs, loss, labels, preds

if __name__ == "__main__":
    model = DemoModel()
    model.train()
    output, loss, labels, preds = model.test()

    print(f"Test output: {output}\nTest loss: {loss}\n Test Accuracy {(labels == preds).sum()}")




        

