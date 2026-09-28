import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

device="cpu"
curr = torch.accelerator.current_accelerator(check_available=True)
match curr:
    case torch.device():
        device = curr.type 
    case None:
        device = "cpu"
print(f"Using {device} device")

class AstroLabeledDataset(Dataset):
    def __init__(self):
        xsim = np.load('astro_data.npy')
        xlab = np.load('astro_labels.npy')
        gr = np.load('astro_rad.npy')
        self.points = torch.from_numpy(xsim).float()
        self.labels = torch.from_numpy(xlab).float()
        self.guiding = torch.from_numpy(gr).float()
    def __len__(self):
        return self.labels.size()[0]
    def __getitem__(self, index):
        return self.points[index],self.labels[index],self.guiding[index]

class AstroTestDataset(Dataset):
    def __init__(self):
        xsim = np.load('astro_test_data.npy')
        # xlab = np.load('astro_labels.npy')
        gr = np.load('astro_test_rad.npy')
        self.points = torch.from_numpy(xsim).float()
        # self.labels = torch.from_numpy(xlab).float()
        self.guiding = torch.from_numpy(gr).float()
    def __len__(self):
        return self.guiding.size()[0]
    def __getitem__(self, index):
        return self.points[index],self.guiding[index]



k_dim = 14
m_dim = 110

class AgeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
                nn.Linear(m_dim,64),
                nn.ReLU(),
                nn.Linear(64,32),
                nn.ReLU(),
                nn.Linear(32,16),
                nn.ReLU(),
                nn.Linear(16,1),
                )

    def forward(self,x):
        return self.stack(x)



def train_loop(the_model,init_dataset,internal_val_percent=0.1,epochs=10,epoch_print_every=20):
    crit = torch.nn.MSELoss()
    optimizer = optim.Adam(the_model.parameters(),lr=0.01,weight_decay=.0001)
    sched = lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.8,patience=100)

    train_set,val_set = torch.utils.data.random_split(init_dataset,[1.0-internal_val_percent,internal_val_percent])
    train_data_loader = DataLoader(train_set,batch_size=400,shuffle=True)
    val_data_loader = DataLoader(val_set,batch_size=200,shuffle=True)
    print(f"Holding back {len(val_data_loader)} batches for running validation")

    train_losses = []
    val_losses = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        the_model.train()
        epoch_print = epoch%epoch_print_every == 0

        running_loss = 0.0
        for i, (data,labels,_) in enumerate(train_data_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data.to(device)
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = the_model(inputs)
            loss = crit(outputs,labels)

            loss.backward()
            optimizer.step()


            # print statistics
            running_loss += loss.item()
        if epoch_print:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/epoch_print_every:.3f}')
            train_losses.append(running_loss/epoch_print_every)
            running_loss = 0.0

        the_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for (data,labels) in val_data_loader:
                val_loss += crit(the_model(data.to(device)),labels.unsqueeze(1).to(device)).item()
            val_loss /= len(val_data_loader)
            if epoch_print:
                print(f"\t Epoch validation loss: {val_loss:.3f}")
                print(f"\t Epoch LR: {sched.get_last_lr()[0]:.3f}")
                val_losses.append(val_loss)
            sched.step(val_loss)
    print('Finished Training')
    return train_losses,val_losses
