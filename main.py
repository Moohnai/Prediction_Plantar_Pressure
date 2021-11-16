import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from random import shuffle
from utils.pytorchtools import EarlyStopping
from data.utils import DataReader
from data.preprocess import DataPreprocess
from model.model import Net
from vit import ViT


# *******softmax_cross_entropy_with_logits & 3 deconvolution layer**********

# load data
P_R, P_L, H_R, H_L = DataReader(path='data')
# preprocess data, make all data to same size
data, label = DataPreprocess([H_L, H_R])


# COP=np.load('COP.npy' , allow_pickle=True)
# LABEL=np.load('Label.npy' , allow_pickle=True)


# train
i=np.arange(data.__len__())
np.random.shuffle(i)
train_idx = i[:int(0.8*len(i))]
data_train = [data[idx] for idx in train_idx]
label_train = [label[idx] for idx in train_idx]

# test
test_idx = i[int(0.8*len(i)):int(0.95*len(i))]
data_test = [data[idx] for idx in test_idx]
label_test = [label[idx] for idx in test_idx]

# validation
valid_idx = i[int(0.95*len(i)):]
data_valid = [data[idx] for idx in valid_idx]
label_valid = [label[idx] for idx in valid_idx]



#
## LSTM+ConvTranspose
# model

hidden_dim=144
label_size=47*32
input_size=47*32
# model = Net(hidden_dim, input_size)
model = ViT(
    img_size=224,
    patch_size=16,
    in_ch=3,
    num_classes=1000,
    use_mlp=True,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    drop_rate=0.0,
)
model.load_state_dict('vit_16_224_imagenet1000.pth')
loss_function1 = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

early_stopping = EarlyStopping(patience=5, verbose=True)


# train
running_loss = 0
print_per_epoch = 5
mini_batch_size = 100
targets = []
labels = []
# Threshold = 10
# reg_t = 1
alpha=0e-6

for epoch in range(500):
    scheduler.step()
    ind_list = [i for i in range(len(data_train))]
    shuffle(ind_list)
    data_train = [data_train[idx] for idx in ind_list]
    label_train = [label_train[idx] for idx in ind_list]
    for i in range(len(data_train)):
        model.zero_grad()
        Net_out=model(torch.FloatTensor(data_train[i]).view((data_train[i].shape[2]), 1 , 47*32))
        targets.append(Net_out)
        labels.append(label_train[i])


        if(i%mini_batch_size== mini_batch_size-1) or (i == len(data_train)-1):

            targets = torch.cat(targets).view(-1, 47*32)
            labels = np.array(labels).reshape(-1, 47*32)

            loss =alpha* torch.sum(torch.sum(- (torch.FloatTensor(labels)) * F.log_softmax(targets, -1), -1))+loss_function1((targets), torch.FloatTensor(labels))

            targets = []
            labels = []
            loss.mean().backward()
            optimizer.step()
            # print(Loss)
#
            running_loss += loss.detach().cpu().numpy()
    if epoch % print_per_epoch == print_per_epoch-1:  # print every 2000 mini-batches
        print('[%d, %5d] loss: %.5f' %
              (epoch + 1, i + 1, running_loss / (500*print_per_epoch)))
        running_loss = 0.0
        # Validation
        labels_2 = []
        targets_2 = []
        with torch.no_grad():
            running_loss_Valid = 0
            for i in range(len(data_valid)):
                model.zero_grad()
                Net_out = model(torch.FloatTensor(data_valid[i]).view((data_valid[i].shape[2]), 1, 47 * 32))

                targets_2.append(Net_out)
                labels_2.append(label_valid[i])
            targets_2 = torch.cat(targets_2).view(-1, 47 * 32)
            labels_2= np.array(labels_2).reshape(-1, 47 * 32)
            loss = loss_function1((targets_2), torch.FloatTensor(labels_2))
            running_loss_Valid += loss.detach().cpu().numpy()
            print('Val_loss: %.5f' % (running_loss_Valid/len(data_valid)))
        #### Check early stopping
        early_stopping(running_loss_Valid, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break



torch.save(model.state_dict(), 'Model_softmax_cross_entropy_with_logits.pt')
#
# model.load_state_dict(torch.load('Model_softmax_cross_entropy_with_logits.pt'))

#test
labels_2 = []
targets_2 = []
with torch.no_grad():
    running_loss_test=0
    for i in range(len(data_test)):
        model.zero_grad()
        Net_out = model(torch.FloatTensor(data_test[i]).view((data_test[i].shape[2]), 1, 47 * 32))

        targets_2.append(Net_out)
        labels_2.append(label_test[i])
    targets_2 = torch.cat(targets_2).view(-1, 47 * 32)
    labels_2 = np.array(labels_2).reshape(-1, 47 * 32)
    loss = loss_function1((targets_2), torch.FloatTensor(labels_2))
    running_loss_test += loss.detach().cpu().numpy()
    print('test_loss: %.5f' % (running_loss_test/len(data_test)))





from random import *
i=randint(1, labels_2.__len__())
print(i)

# print(torch.LongTensor(targets_2[i]))
# print(labels_2[i])


### Test test

targets_2=targets_2.view(-1,47,32)
labels_2=np.array(labels_2).reshape(-1, 47 , 32)
# torch.threshold(targets_2 , 0.01, 0 )
plt.figure(17)
plt.imshow(labels_2[i])
plt.figure(18)
plt.imshow(targets_2[i].cpu())


### Train test
plt.figure(19)
plt.imshow(label_train[i])
plt.figure(20)
plt.imshow(model(torch.FloatTensor(data_train[i]).view((data_train[i].shape[2]), 1 , 47*32)).view(47,32).detach().cpu().numpy())
plt.show()
