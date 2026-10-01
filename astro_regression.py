import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt


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


# ============================================================
# Histogram plotting of true labels vs guiding radius, for training and validation sets.

SEED = 442
VAL_FRACTION = 0.10

# Appendix A uses 0 < r < 14.
# Use 1-unit-wide bins in guiding radius.
R_BINS = np.arange(0.0, 15.0, 1.0)


def make_train_val_split(dataset, val_fraction=VAL_FRACTION, seed=SEED):
    generator = torch.Generator().manual_seed(seed)

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [1.0 - val_fraction, val_fraction],
        generator=generator,
    )

    return train_set, val_set


def get_labels_and_radius(subset):
    """
    Extract true labels y and guiding radii r from a torch Subset.
    """
    indices = torch.as_tensor(subset.indices, dtype=torch.long)
    dataset = subset.dataset

    y = dataset.labels[indices].detach().cpu().numpy().reshape(-1)
    r = dataset.guiding[indices].detach().cpu().numpy().reshape(-1)

    return y, r


def binned_mean(y, r, bins):
    """
    Compute mean y in bins of guiding radius r.
    """
    counts, _ = np.histogram(r, bins=bins)
    y_sum, _ = np.histogram(r, bins=bins, weights=y)

    means = np.full(len(counts), np.nan)

    nonempty = counts > 0
    means[nonempty] = y_sum[nonempty] / counts[nonempty]

    centers = 0.5 * (bins[:-1] + bins[1:])

    return centers, means, counts


def plot_true_y_vs_radius(train_set, val_set,
                          filename="astro_true_y_vs_radius.pdf"):

    y_train, r_train = get_labels_and_radius(train_set)
    y_val, r_val = get_labels_and_radius(val_set)

    centers, mean_train, _ = binned_mean(
        y_train, r_train, R_BINS
    )

    _, mean_val, _ = binned_mean(
        y_val, r_val, R_BINS
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    ax.plot(
        centers,
        mean_train,
        marker="s",
        linestyle="none",
        markerfacecolor="none",
        label=f"Training ($N={len(train_set)}$)",
    )

    ax.plot(
        centers,
        mean_val,
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        label=f"Validation ($N={len(val_set)}$)",
    )

    ax.set_xlim(0, 14)

    ax.set_xlabel(r"Guiding radius $r$")
    ax.set_ylabel(r"Mean true label $y$")

    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(filename)

    plt.close(fig)




# ============================================================
# (c) Training
# ============================================================

EPOCHS = 150
TRAIN_BATCH_SIZE = 400
VAL_BATCH_SIZE = 200


def train_age_model(the_model, train_set, val_set,
                    epochs=EPOCHS, print_every=20):

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        the_model.parameters(),
        lr=0.01,
        weight_decay=0.0001,
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.8,
        patience=100,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        # -------------------------
        # Training
        # -------------------------
        the_model.train()

        train_loss_sum = 0.0
        n_train = 0

        for data, labels, _ in train_loader:

            inputs = data.to(device)
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = the_model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            train_loss_sum += loss.item() * batch_size
            n_train += batch_size

        train_loss = train_loss_sum / n_train

        # -------------------------
        # Validation
        # -------------------------
        the_model.eval()

        val_loss_sum = 0.0
        n_val = 0

        with torch.no_grad():

            for data, labels, _ in val_loader:

                inputs = data.to(device)
                labels = labels.unsqueeze(1).to(device)

                outputs = the_model(inputs)
                loss = criterion(outputs, labels)

                batch_size = labels.size(0)

                val_loss_sum += loss.item() * batch_size
                n_val += batch_size

        val_loss = val_loss_sum / n_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (
            epoch == 0
            or (epoch + 1) % print_every == 0
            or epoch == epochs - 1
        ):
            print(
                f"Epoch {epoch + 1:4d}: "
                f"train MSE = {train_loss:.4f}, "
                f"val MSE = {val_loss:.4f}, "
                f"lr = {optimizer.param_groups[0]['lr']:.5f}"
            )

    print("Finished Training")

    return train_losses, val_losses



# ============================================================
# Prediction on a labeled subset
# ============================================================

def predict_labeled_subset(the_model, subset):

    loader = DataLoader(
        subset,
        batch_size=400,
        shuffle=False,
    )

    y_true = []
    y_pred = []
    radii = []

    the_model.eval()

    with torch.no_grad():

        for data, labels, guiding in loader:

            inputs = data.to(device)

            outputs = the_model(inputs)

            y_true.append(
                labels.detach().cpu().numpy().reshape(-1)
            )

            y_pred.append(
                outputs.detach().cpu().numpy().reshape(-1)
            )

            radii.append(
                guiding.detach().cpu().numpy().reshape(-1)
            )

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    radii = np.concatenate(radii)

    return y_true, y_pred, radii


# ============================================================
# (d) Validation diagnostics
# ============================================================

def print_validation_summary(y_true, y_pred):

    residual = y_pred - y_true

    mse = np.mean(residual**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residual))

    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    slope, intercept = np.polyfit(
        y_true,
        y_pred,
        1,
    )

    print()
    print("Validation summary")
    print("------------------")
    print(f"MSE:              {mse:.4f}")
    print(f"RMSE:             {rmse:.4f}")
    print(f"MAE:              {mae:.4f}")
    print(f"Correlation:      {correlation:.4f}")
    print(f"Best-fit slope:   {slope:.4f}")
    print(f"Best-fit intercept: {intercept:.4f}")

    print()
    print("Distribution")
    print("------------")
    print(
        f"true y: mean = {np.mean(y_true):.3f}, "
        f"std = {np.std(y_true):.3f}"
    )
    print(
        f"pred y: mean = {np.mean(y_pred):.3f}, "
        f"std = {np.std(y_pred):.3f}"
    )


def plot_validation_true_vs_pred(
    y_true,
    y_pred,
    filename="astro_validation_true_vs_pred.pdf",
):

    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    ax.scatter(
        y_true,
        y_pred,
        s=8,
        alpha=0.3,
    )

    low = min(np.min(y_true), np.min(y_pred))
    high = max(np.max(y_true), np.max(y_pred))

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=1.2,
        label=r"$\hat y = y$",
    )

    ax.set_xlabel(r"True label $y$")
    ax.set_ylabel(r"Estimated label $\hat{y}$")

    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# ============================================================
# (e) Estimated y-hat versus guiding radius
# ============================================================

def plot_estimated_y_vs_radius(
    the_model,
    train_set,
    val_set,
    filename="astro_estimated_y_vs_radius.pdf",
):

    _, yhat_train, r_train = predict_labeled_subset(
        the_model,
        train_set,
    )

    _, yhat_val, r_val = predict_labeled_subset(
        the_model,
        val_set,
    )

    centers, mean_train, _ = binned_mean(
        yhat_train,
        r_train,
        R_BINS,
    )

    _, mean_val, _ = binned_mean(
        yhat_val,
        r_val,
        R_BINS,
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    ax.plot(
        centers,
        mean_train,
        marker="s",
        linestyle="none",
        markerfacecolor="none",
        label=f"Training ($N={len(train_set)}$)",
    )

    ax.plot(
        centers,
        mean_val,
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        label=f"Validation ($N={len(val_set)}$)",
    )

    ax.set_xlim(0, 14)

    ax.set_xlabel(r"Guiding radius $r$")
    ax.set_ylabel(r"Mean estimated label $\hat{y}$")

    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":

    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ========================================================
    # Dataset and fixed train/validation split
    # ========================================================

    labeled_dataset = AstroLabeledDataset()

    train_set, val_set = make_train_val_split(
        labeled_dataset
    )

    print(f"Total labeled events: {len(labeled_dataset)}")
    print(f"Training events:      {len(train_set)}")
    print(f"Validation events:    {len(val_set)}")

    # ========================================================
    # (b) True y versus guiding radius
    # ========================================================

    plot_true_y_vs_radius(
        train_set,
        val_set,
    )

    # ========================================================
    # (c) Train the regression model
    # ========================================================

    model = AgeClassifier().to(device)

    train_losses, val_losses = train_age_model(
        model,
        train_set,
        val_set,
    )

    torch.save(
        model.state_dict(),
        "astro_age_model.pt",
    )

    # ========================================================
    # (d) Validation
    # ========================================================

    y_val, yhat_val, r_val = predict_labeled_subset(
        model,
        val_set,
    )

    print_validation_summary(
        y_val,
        yhat_val,
    )

    plot_validation_true_vs_pred(
        y_val,
        yhat_val,
    )

    # ========================================================
    # (e) Estimated y-hat versus guiding radius
    # ========================================================

    plot_estimated_y_vs_radius(
        model,
        train_set,
        val_set,
    )

    print()
    print("Saved:")
    print("  astro_true_y_vs_radius.pdf")
    print("  astro_validation_true_vs_pred.pdf")
    print("  astro_estimated_y_vs_radius.pdf")
    print("  astro_age_model.pt")