import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

device="cpu"
curr = torch.accelerator.current_accelerator(check_available=True)
match curr:
    case torch.device():
        device = curr.type 
    case None:
        device = "cpu"
print(f"Using {device} device")

wz_classes = ['z','w']

class WZDataset(Dataset):
    def __init__(self,set_number:int):
        xsim = np.load('wz_data_'+str(set_number)+'.npy')
        xlab = np.load('wz_labels_'+str(set_number)+'.npy')
        self.points = torch.from_numpy(xsim).float()
        self.labels = torch.from_numpy(xlab)
    def __len__(self):
        return self.labels.size()[0]
    def __getitem__(self, index):
        return self.points[index],self.labels[index]
        


class WZClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
                nn.Linear(2,100),
                nn.ReLU(),
                nn.Linear(100,100),
                nn.ReLU(),
                nn.Linear(100,100),
                nn.ReLU(),
                nn.Linear(100,2),
                )

    def forward(self,x):
        return self.stack(x)


def train_loop(the_model,train_data_loader,print_every=100):
    crit = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(the_model.parameters(), lr=0.01, momentum=0.9)
    sched = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)
    the_model.train()
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (data,labels) in enumerate(train_data_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data.to(device)
            labels = labels.to(device)


            # forward + backward + optimize
            outputs = the_model(inputs)
            loss = crit(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_every == print_every-1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/print_every:.3f}')
                running_loss = 0.0

        sched.step()
    print('Finished Training')



def test_loop(the_model,test_data_loader):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in wz_classes}
    total_pred = {classname: 0 for classname in wz_classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data
            inputs = inputs.to(device)

            outputs = the_model(inputs)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[wz_classes[label]] += 1
                total_pred[wz_classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
