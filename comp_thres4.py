import json
import numpy as np
import pylab

fn_list=['0.5', '0.55', '0.6', '0.65', '0.7','0.75','0.8']

xv=[]
for xin in fn_list:
	xv.append(float(xin))

STLM01=[]
STLM2=[]
synthetic01=[]
synthetic2=[]

LTLM01=[]
LTLM2=[]
for xi, xin in enumerate(fn_list):
    fp=open('results'+xin+'.json','r')
    temp=json.load(fp)
    fp.close()
    num=temp['num_3rd']
    cor_app_01=[]
    cor_app_2=[]
    cor_trn_01=[]
    cor_trn_2=[]
    for yin in xrange(num):
        app_01,app_2,total=temp['Test_app'+str(yin)]
        trn_01,trn_2=temp['Test_trn'+str(yin)]
        cor_app_01.append(app_01)
        cor_app_2.append(app_2)
        cor_trn_01.append(trn_01)
        cor_trn_2.append(trn_2)
    STLM01.append(1-cor_app_01[-1])
    STLM2.append(1-cor_app_2[-1])
    synthetic01.append(1-cor_trn_01[-1])
    synthetic2.append(1-cor_trn_2[-1])
    ans_01,ans_2=temp['Final_Test_trn']
    LTLM01.append(1-ans_01)
    LTLM2.append(1-ans_2)

#STLM01=np.mean(np.array(STLM01))
#STLM2=np.mean(np.array(STLM2))

#synthetic01=np.mean(np.array(synthetic01)) 
#synthetic2=np.mean(np.array(synthetic2))    
xv1=[]
xv2=[]
xv3=[]
xv4=[]



xv=np.array(xv)
xv1=xv-0.02
xv2=xv-0.01
xv3=xv+0.00
xv4=xv+0.01
opacity=0.5
bar_width=0.01

pylab.bar(xv1,STLM01, bar_width,alpha=opacity,color='r',label='STLM01')
pylab.bar(xv2,STLM2, bar_width,alpha=opacity,color='g',label='STLM2')
pylab.bar(xv3,LTLM01, bar_width,alpha=opacity,color='b',label='final01')
pylab.bar(xv4,LTLM2, bar_width,alpha=opacity,color='c',label='final2')
pylab.legend(loc=0)
pylab.savefig('MNIST-comp_thres4.eps')    
pylab.show()

    
