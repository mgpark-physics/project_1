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

wz_classes = ['z','w'] # Massive gauge bosons?

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





# ============================================================


import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split


def plot_classifier(model, dataset, set_number):
    """Show the data and the decision regions learned by the NN."""

    x = dataset.points.numpy()
    y = dataset.labels.numpy()

    # Build a grid of hypothetical (x1, x2) events
    x1 = np.linspace(x[:, 0].min(), x[:, 0].max(), 250)
    x2 = np.linspace(x[:, 1].min(), x[:, 1].max(), 250)
    xx, yy = np.meshgrid(x1, x2)

    grid = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()],
        dtype=torch.float32,
    ).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(grid).argmax(dim=1).cpu().numpy()

    pred = pred.reshape(xx.shape)

    # Plot learned classification regions
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, pred,
                levels=[-0.5, 0.5, 1.5],
                alpha=0.2)

    # Plot only a subset so the figure stays readable
    rng = np.random.default_rng(1)
    idx = rng.choice(len(x), min(5000, len(x)), replace=False)

    for label, name in [(0, "Z"), (1, "W")]:
        mask = y[idx] == label
        ax.scatter(x[idx][mask, 0],
                   x[idx][mask, 1],
                   s=5, alpha=0.3,
                   label=name)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"wz_set{set_number}_classifier.pdf")
    plt.close(fig)



if __name__ == "__main__":

    torch.manual_seed(1)

    for set_number in [1, 2]:

        print(f"\n========== DATASET {set_number} ==========")

        dataset = WZDataset(set_number)

        # CrossEntropyLoss expects integer class labels
        dataset.labels = dataset.labels.long()

        print("data shape :", dataset.points.shape)
        print("label shape:", dataset.labels.shape)

        # 70% training, 15% validation, 15% testing
        train_set, val_set, test_set = random_split(
            dataset,
            [0.70, 0.15, 0.15],
            generator=torch.Generator().manual_seed(1),
        )

        train_loader = DataLoader(
            train_set,
            batch_size=400,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=400,
        )

        test_loader = DataLoader(
            test_set,
            batch_size=400,
        )

        # The lecturer's 2 -> 100 -> 100 -> 100 -> 2 MLP
        model = WZClassifier().to(device)

        print("\nTraining:")
        train_loop(model, train_loader)

        print("\nValidation:")
        test_loop(model, val_loader)

        print("\nTest:")
        test_loop(model, test_loader)

        plot_classifier(model, dataset, set_number)



# ============================================================
# Diagnostic: average (m) and difference (d)
# ============================================================

def plot_m_d(dataset, set_number):

    x = dataset.points.numpy()
    y = dataset.labels.numpy()

    # Common latent-information direction and difference direction
    m = 0.5 * (x[:, 0] + x[:, 1])
    d = 0.5 * (x[:, 0] - x[:, 1])

    # Print simple numerical summaries
    print(f"\n--- Dataset {set_number}: m and d ---")
    for label, name in [(0, "Z"), (1, "W")]:
        mask = (y == label)
        print(
            f"{name}: "
            f"<m> = {m[mask].mean():6.3f},  std(m) = {m[mask].std():6.3f};  "
            f"<d> = {d[mask].mean():6.3f},  std(d) = {d[mask].std():6.3f}"
        )

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    # Use common binning for Z and W
    m_bins = np.linspace(m.min(), m.max(), 80)
    d_bins = np.linspace(d.min(), d.max(), 80)

    for label, name in [(0, "Z"), (1, "W")]:

        mask = (y == label)

        ax[0].hist(
            m[mask],
            bins=m_bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=name,
        )

        ax[1].hist(
            d[mask],
            bins=d_bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=name,
        )

    ax[0].set_xlabel(r"$m=(x_1+x_2)/2$")
    ax[1].set_xlabel(r"$d=(x_1-x_2)/2$")

    ax[0].set_ylabel("Normalized events")

    ax[0].legend()
    ax[1].legend()

    fig.tight_layout()

    fig.savefig(f"wz_set{set_number}_m_d.pdf")
    fig.savefig(f"wz_set{set_number}_m_d.png", dpi=200)

    plt.close(fig)


# Run the diagnostic for both datasets
for set_number in [1, 2]:
    dataset = WZDataset(set_number)
    plot_m_d(dataset, set_number)