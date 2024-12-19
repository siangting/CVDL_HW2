import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import time

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.Grayscale(num_output_channels=3),
         transforms.ToTensor()])

    batch_size = 100

    trainSet = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size,
                                              shuffle=True, num_workers=2, pin_memory=True)

    testSet = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=batch_size,
                                             shuffle=False, num_workers=2, pin_memory=True)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    model = torch.load("./models/vggModel.pth").cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    trainLoss = []
    valLoss = []
    trainAcc = []
    valAcc = []
    startTime = time.time()
    current = datetime.now().strftime("%Y%m%d-%H%M%S")
    epochs = 30
    pbar = tqdm(range(epochs), desc="Epoch")
    bestAcc = 0.0
    for epoch in pbar:
        # print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        runningLoss = 0.0
        correctTrain = 0
        totalTrain = 0
        for inputs, labels in tqdm(trainLoader, unit="images", unit_scale=trainLoader.batch_size, leave=False,
                                   desc="Train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            _, predicted = outputs.max(1)
            totalTrain += labels.size(0)
            correctTrain += predicted.eq(labels).sum().item()

        trainLoss.append(runningLoss / len(trainLoader))
        trainAcc.append(100 * correctTrain / totalTrain)

        model.eval()
        runningTestLoss = 0.0
        correctTest = 0
        totalTest = 0
        with torch.no_grad():
            for inputs, labels in tqdm(testLoader, unit="images", unit_scale=testLoader.batch_size, leave=False,
                                       desc="Test"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                runningTestLoss += loss.item()
                _, predicted = outputs.max(1)
                totalTest += labels.size(0)
                correctTest += predicted.eq(labels).sum().item()

        valLoss.append(runningTestLoss / len(testLoader))
        valAcc.append(100 * correctTest / totalTest)
        # print(f"Epoch{epoch}, Train Loss: {trainLoss[-1]}, Acc: {trainAcc[-1]}.\nValid Loss: {valLoss[-1]}, Acc: {valAcc[-1]}")
        pbar.set_postfix({
            "TrainLoss": trainLoss[-1],
            "TrainAcc": trainAcc[-1],
            "ValidLoss": valLoss[-1],
            "ValidAcc": valAcc[-1]
        })
        # lr_scheduler.step()
        if valAcc[-1] > bestAcc:
            torch.save(model, f"./models/trained_model_{current}")

    endTime = time.time() - startTime
    print(f'\n The Training Took {endTime} seconds')

    f = plt.figure(1)
    plt.plot(range(1, epochs + 1), trainLoss, color='blue', label='TrainLoss')
    plt.plot(range(1, epochs + 1), valLoss, color='orange', label='valLoss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()

    g = plt.figure(2)
    plt.plot(range(1, epochs + 1), trainAcc, color='blue', label='TrainAcc')
    plt.plot(range(1, epochs + 1), valAcc, color='orange', label='valAcc')
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.title("Accuracy")
    plt.legend()
    plt.show()