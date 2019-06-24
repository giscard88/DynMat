import json
import numpy as np
import pylab

fn_list=['0.5', '0.55', '0.6', '0.65', '0.7','0.75','0.8']

example_num=[]
final_01=[]
final_2=[]
final_all=[]
initial_01=[]
xv=[]
for xi, xin in enumerate(fn_list):
    xv.append(float(xin))
    fp=open('results'+xin+'.json','r')
    temp=json.load(fp)
    fp.close()
    example_num.append(temp['Final_Stored']) 
    
    ans_01,ans_2=temp['Final_Test_trn']
    initial_01.append(1-temp['TestCorrect_initinal'])

    final_01.append(1-ans_01)
    final_2.append(1-ans_2)
    #final_all.append(ans_all)
    
pylab.figure(1)
pylab.plot(xv,initial_01,'-ro',label='initial')
pylab.plot(xv,final_01,'-go',label='LTLM01')
pylab.plot(xv,final_2,'-bo',label='LTLM2')
#pylab.yticks([0.0005, 0.001,0.0025,0.005,0.01],['0.05','0.1','0.25', '0.5', '1'])
#pylab.ylim([0.0004,0.011])
#pylab.ylabel('error rate (%)')
#pylab.plot(xv,fianl_all,label='final_all')
pylab.legend()

pylab.savefig('MNIST_comp_thres1.eps')
pylab.figure(2)

xv=np.array(xv)-0.0125
pylab.bar(xv,example_num,width=0.025)
pylab.ylabel('# stored examples')
pylab.xlabel('threshold')
pylab.savefig('MNIST_comp_thres2.eps')
pylab.show()


