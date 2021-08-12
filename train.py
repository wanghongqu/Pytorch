import torch
from glob import glob
import os
import time
import shutil
from tensorboardX import SummaryWriter
from torch.cuda.amp import *
from model.ResNet import *
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms as T
from torchmetrics import Accuracy


def transforms():
    return T.Compose([
        # T.Pad(3),
        # T.RandomResizedCrop(size=32),
        # T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        # T.RandomAutocontrast(),
        # T.RandomGrayscale(),
        # T.RandomRotation(35),
        # T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.AutoAugment(),
        T.ToTensor(),
        T.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        T.RandomErasing()
    ])


def test_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])


# log
writer = SummaryWriter("log")

# prepare data
batch_size = 512
train_data = CIFAR100("./", train=True, transform=transforms(), download=True)

test_data = CIFAR100("./", train=False, transform=test_transforms(), download=True)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=30)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=30)

# prepare model
model = ResNet()
model.apply(weight_init)
if (len(glob("*.pth")) != 0):
    model.load_state_dict(torch.load(glob("*.pth")[0]))
    print("load old weights ", glob("*.pth")[0])

optim = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

# train
scaler = GradScaler()

# basic param
best_acc = 0
epoch = 100
weight_decay = 0.001

# check_validation
accuracy = Accuracy()


@torch.no_grad()
def check_validation():
    acc = []
    loss = []
    model.train(False)

    for batch in test_loader:
        x_test, y_test = batch
        if (torch.cuda.is_available()):
            x_test = x_test.cuda().detach()
            y_test = y_test.cuda().detach()
        y_pred = model(x_test).detach()
        loss.append(loss_func(y_pred, y_test).detach().cpu().numpy())
        pred_idx = torch.argmax(y_pred.softmax(dim=-1), dim=-1)
        acc.append((torch.sum(pred_idx == y_test).detach().cpu() / y_test.shape[0]).numpy())
    reg_loss = 0
    for name, w in model.named_parameters():
        l2_reg = torch.norm(w, p=2)
        reg_loss = reg_loss + l2_reg

    reg_loss = weight_decay * reg_loss
    print("epoch:", i, "test acc:", np.mean(acc), "\tloss:", (np.mean(loss) + reg_loss).item())
    return np.mean(acc), np.mean(loss) + reg_loss, reg_loss


if torch.cuda.is_available():
    model = model.cuda()
for i in range(epoch):
    train_loss = []
    train_acc = []
    train_time = []
    for j, batch in enumerate(train_loader):
        l1_loss = 0
        start = time.time()
        model.train();
        model.zero_grad()
        x_train, y_train = batch
        if (torch.cuda.is_available()):
            x_train = x_train.cuda().detach()
            y_train = y_train.cuda().detach()
        with autocast():
            reg_loss = 0
            for name, w in model.named_parameters():
                l2_reg = torch.norm(w, p=2)
                reg_loss = reg_loss + l2_reg

            reg_loss = weight_decay * reg_loss

            y_pred = model(x_train)
            loss_val = loss_func(y_pred, y_train) + reg_loss

        scaler.scale(loss_val).backward()

        scaler.step(optim)
        scaler.update()

        acc_val = accuracy(y_pred.softmax(dim=-1).detach().cpu(), y_train.detach().cpu())
        train_acc.append(acc_val)
        train_loss.append(loss_val.item())
        end = time.time()
        train_time.append(end - start)

    test_acc, test_loss, reg_loss = check_validation()
    train_acc_, train_loss_, train_time_ = np.mean(train_acc), np.mean(train_loss), np.mean(train_time)
    if test_acc > best_acc:
        os.system("rm -rf *.pth")
        best_acc = test_acc
        torch.save(model.state_dict(), "{:.2f}_{:.2f}.pth".format(test_acc, test_loss))
    writer.add_scalars("acc", {'train': train_acc_, 'test': test_acc}, i)
    writer.add_scalars("loss", {'train': train_loss_, 'test': test_loss, "regg loss": reg_loss}, i)
    writer.add_scalar("train cost time", train_time_, i)
    writer.flush()
