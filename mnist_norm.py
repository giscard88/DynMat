import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
import numpy as np
import pylab

#train_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=True, download=True,
#                   transform=transforms.Compose([
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.1307,), (0.3081,))
#                   ])),
#   batch_size=60000, shuffle=False)

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('/home/jung/pytorch/data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       ])),
    batch_size=60000, shuffle=False)

for batch_idx, (data, target) in enumerate(train_loader):
    print batch_idx

seed=int(sys.argv[1])
np.random.seed(seed)
arr = np.arange(10)
if seed==0:
    pass
else:
    np.random.shuffle(arr)

sel_labels=arr[:3]
     

index01=[]
label_tr01=[]
label_tr2=[]

index2=[]   


for xi, xin in enumerate(target):
    if xin==sel_labels[0]:
        index01.append(xi)
        label_tr01.append(0)
    elif xin==sel_labels[1]:
        index01.append(xi)
        label_tr01.append(1)
    elif xin==sel_labels[2]:
        index2.append(xi)
        label_tr2.append(2)


index01=torch.LongTensor(index01)
index2=torch.LongTensor(index2)
tr_target_01=torch.LongTensor(label_tr01)
tr_target_2=torch.LongTensor(label_tr2)

train_01=torch.index_select(data,0,index01)


train_2=torch.index_select(data,0,index2)




num_tr=len(train_01)
train_01=train_01.view(num_tr,784)

num_tr=len(train_2)
train_2=train_2.view(num_tr,784)

for xi, xin in enumerate(train_01):
    nr=torch.norm(xin,2)
    train_01[xi]=xin/nr
for xi, xin in enumerate(train_2):
    nr=torch.norm(xin,2)
    train_2[xi]=xin/nr



test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('/home/jung/pytorch/data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=10000, shuffle=False)


for batch_idx, (data, target) in enumerate(test_loader):
    print batch_idx
index01=[]
index2=[]
label_tst01=[]
label_tst2=[]

index2=[]     

for xi,xin in enumerate(target):
    if xin==sel_labels[0]:
        index01.append(xi)
        label_tst01.append(0)
    elif xin==sel_labels[1]:
        index01.append(xi)
        label_tst01.append(1)
    elif xin==sel_labels[2]:
        index2.append(xi)
        label_tst2.append(2)
index01=torch.LongTensor(index01)
index2=torch.LongTensor(index2)

tst_target_01=torch.LongTensor(label_tst01)
tst_target_2=torch.LongTensor(label_tst2)


test_01=torch.index_select(data,0,index01)

test_2=torch.index_select(data,0,index2)




num_tr=len(test_01)
test_01=test_01.view(num_tr,784)

num_tr=len(test_2)
test_2=test_2.view(num_tr,784)


#now the set the magnitude to be 1. 
for xi, xin in enumerate(test_01):
    nr=torch.norm(xin,2)
    test_01[xi]=xin/nr

for xi, xin in enumerate(test_2):
    nr=torch.norm(xin,2)
    test_2[xi]=xin/nr


print train_01.size()

torch.save(train_01,'train_01.pt')
torch.save(train_2,'train_2.pt')
torch.save(tr_target_01,'tr_target_01.pt')
torch.save(tr_target_2,'tr_target_2.pt')

torch.save(test_01,'test_01.pt')
torch.save(test_2,'test_2.pt')
torch.save(tst_target_01,'tst_target_01.pt')
torch.save(tst_target_2,'tst_target_2.pt')





#script




