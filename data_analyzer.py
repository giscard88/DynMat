import json
import numpy as np
import pylab

fn_list=['0.5', '0.55', '0.6', '0.65', '0.7','0.75','0.8']

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
        cor_app_01.append(1-app_01)
        cor_app_2.append(1-app_2)
        cor_trn_01.append(1-trn_01)
        cor_trn_2.append(1-trn_2)
    xv=range(num)
    xv=np.array(xv)+1
    pylab.figure(xi+1)
    maxv2=np.amax(np.array(cor_app_2))
    maxv=np.amax(np.array(cor_app_01))
    minv=np.amin(np.array(cor_app_01))
    if maxv<0.1:
        pylab.semilogy(xv,cor_app_01,'-r',label='STLM01',linewidth=2)
        pylab.semilogy(xv,cor_app_2,'-b',label='STLM2',linewidth=2)
    else:
        pylab.plot(xv,cor_app_01,'-r',label='STLM01',linewidth=2)
        pylab.plot(xv,cor_app_2,'-b',label='STLM2',linewidth=2)
    pylab.ylim([minv*0.9, maxv2*1.1])
    pylab.xlim([0.8, xv[-1]*1.01])
    pylab.legend()
    pylab.title(xin)
    pylab.savefig('MNIST'+xin+'.eps')
pylab.show()

    
