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


# def transforms():
#     return T.Compose([T.AutoAugment(), T.ToTensor()])


def transforms():
    return T.Compose([
        T.Pad(5),
        T.RandomResizedCrop(size=32),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomAutocontrast(),
        T.RandomGrayscale(),
        T.RandomRotation(30),
        T.ToTensor(),
        T.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        T.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=True)
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
    model = model.load_state_dict(torch.load(glob("*.pth")[0]))
    print("load old weights ", glob("*.pth")[0])

optim = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

# train
scaler = GradScaler()

# basic param
best_acc = 0
epoch = 100

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
    print("test acc:", np.mean(acc), "\tloss:", np.mean(loss))
    return np.mean(acc)


if torch.cuda.is_available():
    model = model.cuda()
for i in range(epoch):
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
            y_pred = model(x_train)
            loss_val = loss_func(y_pred, y_train)
        scaler.scale(loss_val).backward()

        scaler.step(optim)
        scaler.update()

        acc_val = accuracy(y_pred.softmax(dim=-1).detach().cpu(), y_train.detach().cpu())
        end = time.time()
        print("epoch:{} ,[{}/{}],acc:{:.3f} total loss:{:.2f},cost time:{:.2f}".format(i, j, len(
            train_loader), acc_val, loss_val, end - start))
        writer.add_scalar("train acc:", acc_val, i * len(train_loader) + j)
        writer.add_scalar("train loss:", loss_val, i * len(train_loader) + j)
        # writer.add_scalar("regg loss:", loss_val, i * len(train_loader) + j)
        writer.flush()
    cur_acc = check_validation()
    if cur_acc > best_acc:
        os.system("rm -rf *.pth")
        best_acc = cur_acc
        torch.save(model.state_dict(), str(cur_acc) + ".pth")
    writer.add_scalar("test acc:", cur_acc, i)
