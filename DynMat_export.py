import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import json
import sys



class First:
    W0=1.0
    wd=0.1
    def __init__(self,size):
        self.W=[]
        self.size=size
        self.n_neuron=0
        self.labels=[]
        self.label_n=0
        print 'successfully initialized'

    def add_neuron(self, X,label=None):
       
        X=np.array(X)
        self.W.append(X)
        self.n_neuron=self.n_neuron+1
        
        if label==None:
            pass
        else:
            if label in self.labels:
                pass
            else:
                self.label_n=self.label_n+1
            self.labels.append(label)

    def get_weight1(self):
        temp=np.array(self.W)
        self.weight1=torch.FloatTensor(temp)

    def get_weight2(self):

        conn=[]
        labels=np.array(self.labels)
        label_fix=max(labels)+1 # just for conveninence, let's assume that each stimulus has ordered labels. 
        for xin in xrange(label_fix):
            zeros=np.zeros(self.n_neuron)
            indices=np.where(labels==xin)[0]
            zeros[indices]=self.W0
            conn.append(zeros)
        conn=np.array(conn)
        self.weight2=torch.FloatTensor(conn)
            
        

    def forward(self, X):
        X=torch.transpose(X,0,1)
        self.get_weight1()
        WI=self.weight1
        self.get_weight2()
        WT=self.weight2
        self.H=torch.mm(WI,X)

   
        temp=(self.H-1)/self.wd
        temp=torch.pow(temp,2)
        temp=torch.exp(-temp)
        self.mH=self.H
 
        self.T=torch.mm(WT,temp)

    def forward_i(self, X):
        
        X=X.view(784,1)
        self.get_weight1()
        WI=self.weight1
        self.get_weight2()
        WT=self.weight2
        self.H=torch.mm(WI,X)


        temp=(self.H-1)/self.wd

        temp=torch.pow(temp,2)
        temp=torch.exp(-temp)
        self.mH=self.H
        self.T=torch.mm(WT,temp)

class classifier(torch.nn.Module):
    def __init__(self, D_in,H,D_out):
        super(classifier,self).__init__()
        self.fc1=nn.Linear(D_in,H)
        self.fc2=nn.Linear(H,D_out)
    def forward(self, x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        return x    


    
        
def train(inputs,y,rate):

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = torch.nn.MSELoss(size_average=False)
    y_pred=model(inputs)
    learning_rate = rate

    loss = loss_fn(y_pred, y)
    model.zero_grad()
    loss.backward()


    for param in model.parameters():
        param.data -= learning_rate * param.grad.data
    return loss.data[0]

def prepare_data(inputs, labels):
    Stage1.forward(inputs)
    #print 'H',Stage1.H
    x=Variable(torch.t(Stage1.H), requires_grad=False) ## this is changed from original model (model_b) from Stage1.mH
    y_app=torch.t(Stage1.T)
    tg_labels=[]
    for xin in labels:
        temp=np.zeros(Stage1.label_n)
        temp[xin]=1.0
        tg_labels.append(temp)
    tg_labels=np.array(tg_labels)
    tg_labels=torch.Tensor(tg_labels)

    y = Variable(tg_labels, requires_grad=False)
    x=x.cuda()
    y=y.cuda()
    return x,y,y_app # actual input data, labels, outputs from memory 



def test(inputs,tgts):  # test how well the trained classifier can perform 
    y_pred=model(inputs)
    count_01=0
    count_2=0
    all_01=0
    all_2=0
    count_all=0
    y_pred=y_pred.cpu()
    ans=y_pred.data.numpy().argmax(axis=1)
    #print len(ans), type(ans)
    for xi,xin in enumerate(ans):
        #print tgts[xi]
        if tgts[xi]==2:
            all_2=all_2+1
            if xin==tgts[xi]:
                count_2=count_2+1
                count_all=count_all+1
        else:
            all_01=all_01+1
            if xin==tgts[xi]:
                count_01=count_01+1
                count_all=count_all+1
    return y_pred, float(count_01)/float(all_01)#,float(count_2)/float(all_2), float(count_all)/float(len(ans)) 

def test_freeze(inputs,tgts):  # test how well the trained classifier can perform 
    inputs=inputs.narrow(1,0,temp_hidden)
    #print inputs.size()
    y_pred=model(inputs)
    count_01=0
    count_2=0
    all_01=0
    
    ans=y_pred.data.numpy().argmax(axis=1)
    #print len(ans), type(ans)
    for xi,xin in enumerate(ans):
        #print xin, tst_target_01[xi]
        if tgts[xi]==2:
            pass
        else:
            all_01=all_01+1
            if xin==tgts[xi]:
                count_01=count_01+1
    return y_pred, float(count_01)/float(all_01)   

def test_freeze_all(inputs,tgts,y_app):  # test how well the trained classifier can perform 
    inputs=inputs.narrow(1,0,temp_hidden)
    #print inputs.size()
    y_pred=model(inputs)
    y_pred=y_pred.cpu()
    y_pred=y_pred.data.numpy()
    (r,c)=y_pred.shape
    #print r,c

    y_pred_mod=np.zeros((r,c+1))
    #print 'check',r,c,y_pred_mod.shape
    y_pred_mod[:,0]=y_pred[:,0]
    y_pred_mod[:,1]=y_pred[:,1]
    y_pred_mod[:,2]=y_app[:,2]*1.0
    y_pred=y_pred_mod
    y_pred=torch.Tensor(y_pred)
    y_pred=Variable(y_pred,requires_grad=False)
    count_01=0
    count_2=0
    all_01=0
    all_2=0
    
    ans=y_pred.data.numpy().argmax(axis=1)
    #print len(ans), type(ans)
    for xi,xin in enumerate(ans):
        #print xin, tst_target_01[xi]
        if tgts[xi]==2:
            all_2=all_2+1
            if tgts[xi]==xin:
                count_2=count_2+1
        else:
            all_01=all_01+1
            if xin==tgts[xi]:
                count_01=count_01+1
    return y_pred, float(count_01)/float(all_01), float(count_2)/float(all_2)         

def test2(y_app,tgts): # test how well (cognitive memory: from earlier paper, episodic memory: we can call it that way) can perform. 
    count_01=0
    count_2=0
    all_01=0
    all_2=0
    count_all=0
    for xi,xin in enumerate(y_app):
        
        
        ag=np.argmax(xin)
        #print np.array(xin), ag
        #print xi, ag, tst_target_01[xi]
        if tgts[xi]==2:
            all_2=all_2+1
            if tgts[xi]==ag:
                count_2=count_2+1
                count_all=count_all+1
        else:
            all_01=all_01+1
            if tgts[xi]==ag:
                count_01=count_01+1
                count_all=count_all+1

    return y_app, float(count_01)/float(all_01),float(count_2)/float(all_2), float(count_all)/float(len(y_app))      

def test_f(inputs,tgts):  # test how well the trained classifier can perform 
    y_pred=model(inputs)
    count_01=0
    count_2=0
    all_01=0
    all_2=0
    count_all=0
    y_pred=y_pred.cpu()
    ans=y_pred.data.numpy().argmax(axis=1)
    #print len(ans), type(ans)
    for xi,xin in enumerate(ans):
        #print tgts[xi]
        if tgts[xi]==2:
            all_2=all_2+1
            if xin==tgts[xi]:
                count_2=count_2+1
                count_all=count_all+1
        else:
            all_01=all_01+1
            if xin==tgts[xi]:
                count_01=count_01+1
                count_all=count_all+1
    return y_pred, float(count_01)/float(all_01),float(count_2)/float(all_2), float(count_all)/float(len(ans))   

def shuffle(train_mix,tr_target_mix):
    (r,c)=train_mix.size()

    arr=np.arange(r)
    np.random.shuffle(arr)

    train_mix_t=torch.zeros(r,c)
    tr_target_t=torch.zeros(r)

    for xi,xin in enumerate(arr):
        train_mix_t[xi,:]=train_mix[xin,:]
        tr_target_t[xi]=tr_target_mix[xin]

    train_mix=train_mix_t
    tr_target_mix=tr_target_t
    tr_target_mix=tr_target_mix.int()
    train_mix_t=[]
    tr_target_t=[]


    return train_mix,tr_target_mix


train_01=torch.load('train_01.pt') #MNIST and CIFAR were read from different format. 
train_2=torch.load('train_2.pt')
test_01=torch.load('test_01.pt')
test_2=torch.load('test_2.pt')

tr_target_01=torch.load('tr_target_01.pt')
tr_target_2=torch.load('tr_target_2.pt')
tst_target_01=torch.load('tst_target_01.pt')
tst_target_2=torch.load('tst_target_2.pt')



#train_01=train_01v.data
#train_2=train_2v.data

#test_01=test_01v.data
#test_2=test_2v.data

#del train_01v, train_2v, test_01v, test_2v

Results={}




Threshold=float(sys.argv[1])
Results['Threshold']=Threshold

Stage1=First(784)
target=tr_target_01[0]
#print target

Stage1.add_neuron(train_01[0],target)




for xi, xin in enumerate(train_01):
    #print xi
    Stage1.forward_i(xin)
    labels=np.array(Stage1.labels)
    no=2
    target=tr_target_01[xi]
    index=np.where(labels==target)[0]
    if len(index)>0:
            
        index=torch.LongTensor(index)
        sel_H=torch.index_select(Stage1.H,0,index)
        
        indices=np.where(sel_H>=Threshold)[0]
    
        if len(indices)>0:
            pass
        else:
            print xi, target,'add'
            Stage1.add_neuron(xin,target)
    else:
        print xi, target, 'gen'
        Stage1.add_neuron(xin,target)
            
print Stage1.n_neuron

Results['Initinal_Stored']=Stage1.n_neuron

train_batch=[]
target_batch=[]
batch_size=100
size=len(train_01)

Results['batch_size_initial']=batch_size

repeat_n=size/batch_size


targets=[]
for xin in xrange(repeat_n):
    if xin==repeat_n-1:
        temp=train_01[batch_size*xin:-1]
        target=tr_target_01[batch_size*xin:-1]
    else:    
        temp=train_01[batch_size*xin:batch_size*(xin+1)]
        target=tr_target_01[batch_size*xin:batch_size*(xin+1)]
    x,y,y_app=prepare_data(temp,target)
    train_batch.append(x)
    target_batch.append(y)
    targets.append(target)


#print Stage1.weight2


Hid=200
model=classifier(Stage1.n_neuron,Hid,Stage1.label_n)
model=model.cuda()

ans1,p01_trn=test(x,tr_target_01)

Results['HiddenClassifier']=Hid

print 'train',p01_trn
#print ans1

x,y,y_app=prepare_data(test_01,tst_target_01)


ans1,p01_trn=test(x,tst_target_01)


print 'test',p01_trn


for lr in xrange(1000):
    cost=0
    for xi, xin in enumerate(train_batch):
        
        cost_t=train(train_batch[xi],target_batch[xi],1e-4)
        cost=cost+cost_t
        #print lr, xi, cost_t



    if lr%100==0:    
        print lr,cost
        Results['Cost1_'+str(lr)]=cost


x,y,y_app=prepare_data(train_01,tr_target_01)
ans1,p01_trn=test(x,tr_target_01)


print 'train',p01_trn
Results['TrainCorrect_initinal']=p01_trn

x,y,y_app=prepare_data(test_01,tst_target_01)


ans1,p01_trn=test(x,tst_target_01)


print 'test',p01_trn
Results['TestCorrect_initinal']=p01_trn

#print ans1






sel_num_add=-1


train_mix=torch.cat((train_01,train_2),0)
tr_target_mix=torch.cat((tr_target_01,tr_target_2),0)

train_mix,tr_target_mix=shuffle(train_mix,tr_target_mix)

test_mix=torch.cat((test_01,test_2),0)
tst_target_mix=torch.cat((tst_target_01,tst_target_2),0)

test_mix,tst_target_mix=shuffle(test_mix,tst_target_mix)
temp_hidden=Stage1.n_neuron # the size of first layer before adding new categories.. for testing the system.


ct=0
for xi, xin in enumerate(train_2[:sel_num_add]):
    Stage1.forward_i(xin)
    labels=np.array(Stage1.labels)
    target=tr_target_2[xi]
    index=np.where(labels==target)[0]
    flag=0
    if len(index)>0:
            
        index=torch.LongTensor(index)
        sel_H=torch.index_select(Stage1.H,0,index)
        
        indices=np.where(sel_H>=Threshold)[0]
    
        if len(indices)>0:
            pass
        else:
            print xi, target,'add'
            Stage1.add_neuron(xin,target)
            flag=1
    else:
        print xi, target, 'gen'
        Stage1.add_neuron(xin,target)
        flag=1
            

    if flag==1:
        x,y,y_app=prepare_data(test_mix,tst_target_mix)
        ans1,p01_trn,p2_trn=test_freeze_all(x,tst_target_mix,y_app)
        ans2,p01_app,p2_app,c_app=test2(y_app,tst_target_mix)
        print 'test of trn:',p01_trn,p2_trn
        print 'test of app',p01_app,p2_app
        Results['Test_trn'+str(ct)]=[p01_trn,p2_trn]
        Results['Test_app'+str(ct)]=[p01_app,p2_app,c_app]
        ct=ct+1
Results['num_3rd']=ct



print 're-train network'

Stage1.get_weight1()
train_batch=[]
target_batch=[]

size=len(train_mix)

repeat_n=size/batch_size


targets=[]
for xin in xrange(repeat_n):
    if xin==repeat_n-1:
        temp=train_mix[batch_size*xin:-1]
        target=tr_target_mix[batch_size*xin:-1]
    else:    
        temp=train_mix[batch_size*xin:batch_size*(xin+1)]
        target=tr_target_mix[batch_size*xin:batch_size*(xin+1)]
    x,y,y_app=prepare_data(temp,target)
    train_batch.append(x)
    target_batch.append(y)
    targets.append(target)


print Stage1.n_neuron,Hid,Stage1.label_n

Results['Final_Stored']=Stage1.n_neuron

model=classifier(Stage1.n_neuron,Hid,Stage1.label_n)
model=model.cuda()
x,y,y_app=prepare_data(train_mix,tr_target_mix)
for lr in xrange(3000):
    cost=0
    for xi, xin in enumerate(train_batch):
        
        cost_t=train(train_batch[xi],target_batch[xi],1e-4)
        cost=cost+cost_t
        #print lr, xi, cost_t

    if lr%100==0:    
        print lr,cost
        Results['Cost2_'+str(lr)]=cost

for lr in xrange(2000):
    cost=0
    for xi, xin in enumerate(train_batch):
        
        cost_t=train(train_batch[xi],target_batch[xi],1e-5)
        cost=cost+cost_t
        #print lr, xi, cost_t

    if lr%100==0:    
        print lr,cost
        Results['Cost3_'+str(lr)]=cost






x,y,y_app=prepare_data(test_mix,tst_target_mix)
ans1,p01_trn,p2_trn,c=test_f(x,tst_target_mix)
ans2,p01_app,p2_app,c_app=test2(y_app,tst_target_mix)
print 'test of trn:', p01_trn,p2_trn,c
print 'test of app:', p01_app,p2_app,c_app

Results['Final_Test_trn']=[p01_trn,p2_trn]
Results['Final_Test_app']=[p01_app,p2_app]


x,y,y_app=prepare_data(train_mix,tr_target_mix)
ans1,p01_trn,p2_trn,c_trn=test_f(x,tr_target_mix)
ans2,p01_app,p2_app,c_app=test2(y_app,tr_target_mix)
print 'train of trn:', p01_trn,p2_trn,c
print 'train of app:', p01_app,p2_app,c_app

Results['Final_Train_trn']=[p01_trn,p2_trn,c_trn]
Results['Final_Train_app']=[p01_app,p2_app,c_app]


fp=open('results'+str(Threshold)+'.json','w')
json.dump(Results,fp)
fp.close()


#print test_2

#print test2()

#script 
