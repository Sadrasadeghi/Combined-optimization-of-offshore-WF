# =============================================================================
# finding TI_max array in order to use in optmization
# =============================================================================
import pandas as pd
import numpy as np

wt =9
wd=np.arange(5,360,30)
ws=np.arange(4,26)
sigma1=np.zeros((22))
TI=np.zeros((22))
Iref=0.14
for i in range(len(ws)):
        
        sigma1[i]=Iref*(0.75*ws[i]+5.6)
        TI[i]=(sigma1[i]/ws[i]+0.1)/2
        
from py_wake.examples.data.hornsrev1 import V80 
from py_wake.examples.data.hornsrev1 import Hornsrev1Site # We work with the Horns Rev 1 site, which comes already set up with PyWake.
from py_wake.turbulence_models import GCLTurbulence
from scipy.optimize import minimize
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular
from py_wake.wind_farm_models import All2AllIterative
from py_wake.deficit_models import NOJDeficit, SelfSimilarityDeficit
from py_wake.superposition_models import LinearSum
from py_wake.superposition_models import SquaredSum
from py_wake.superposition_models import MaxSum
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular
from scipy.optimize import minimize
from py_wake.rotor_avg_models import RotorCenter, GridRotorAvg, EqGridRotorAvg, GQGridRotorAvg, CGIRotorAvg, PolarGridRotorAvg,polar_gauss_quadrature

def newSite(x,y,wt):
    xNew=np.array([7*80*i for i in range(3)])
    yNew=np.array([7*80*i for i in range(3)])
    x_newsite=np.array([xNew[0],xNew[0],xNew[0],xNew[1],xNew[1],xNew[1],xNew[2],xNew[2],xNew[2]])
    y_newsite=np.array([yNew[0],yNew[1],yNew[2],yNew[0],yNew[1],yNew[2],yNew[0],yNew[1],yNew[2]])
    return (x_newsite[:(wt)],y_newsite[:(wt)])

TI_average=np.ones((len(wd),len(ws)))
for l in range(len(wd)):
    for k in range(len(ws)):
        
        site = Hornsrev1Site()
        x, y = site.initial_position.T
        x_newsite, y_newsite=newSite(x,y,wt)
        windTurbines = V80()
        site.ds['TI']=TI[k]
    
        
        wf_model =  All2AllIterative(site,windTurbines,
                                      wake_deficitModel=NOJDeficit(),
                                       blockage_deficitModel=SelfSimilarityDeficit(),
                                      superpositionModel=LinearSum(),
                                      rotorAvgModel=CGIRotorAvg(21),
                                      turbulenceModel=GCLTurbulence())
        # run wind farm simulation
        sim_res = wf_model(
            x_newsite, y_newsite, # wind turbine positions
            h=None, # wind turbine heights (defaults to the heights defined in windTurbines)
            wd=wd[l], # Wind direction (defaults to site.default_wd (0,1,...,360 if not overriden))
            ws=ws[k], # Wind speed (defaults to site.default_ws (3,4,...,25m/s if not overriden))
            )
        TI_average[l][k]=np.mean(sim_res.TI_eff)
        

temp=np.zeros((12,3))

for l in range(12):
    for k in range(3):
        if k==0:
            temp[l][k]=np.mean(TI_average[l][0:8])
            
        if k==1:
            
            temp[l][k]=np.mean(TI_average[l][8:17])
        if k==2:
            
            temp[l][k]=np.mean(TI_average[l][17:25])


            
   

import matplotlib.pyplot as plt
x,y=np.meshgrid(ws,wd)
z=TI_average
z_min, z_max = 0.11, np.abs(z).max()

c = plt.imshow(z, cmap =plt.cm.jet, vmin = z_min, vmax = z_max,
                 extent =[x.min(), x.max(), y.min(), y.max()],
                    interpolation ='nearest', origin ='lower',aspect='auto')

plt.xticks(np.arange(x.min(), x.max()+1, 1))

plt.yticks(np.arange(y.min(), y.max()+1, 10))
plt.colorbar(c)
plt.xlabel('WS')
plt.ylabel('WD')
plt.title('TI_NMT_Modified', fontweight ="bold")
plt.tight_layout()
plt.show()

# wd=np.arange(5,360,10)
# plt.yticks(np.arange(y.min(), y.max()+1, 30))


out1=TI.reshape(-1)
out1=pd.DataFrame(out1)
out1.to_excel('7D,TI layout optmization,scatter,NMT Modified,weighted.xlsx')