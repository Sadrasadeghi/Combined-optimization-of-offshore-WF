import numpy as np
from py_wake.examples.data.hornsrev1 import V80 # The farm is comprised of 80 V80 turbines which
from py_wake.examples.data.hornsrev1 import Hornsrev1Site # We work with the Horns Rev 1 site, which comes already set up with PyWake.
from py_wake.turbulence_models import GCLTurbulence
from py_wake.wind_farm_models import All2AllIterative
from py_wake.deficit_models import NOJDeficit, SelfSimilarityDeficit
from py_wake.superposition_models import LinearSum
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular
from scipy.optimize import minimize
import pandas as pd
import time
import xarray as xr

#---------------------------------------------

TI_input=pd.read_excel('7D,TI layout optmization,scatter,NMT Modified,weighted.xlsx')
TI_input=TI_input.to_numpy()
TI_input=TI_input[:,1]
TI_max=TI_input.reshape((12,3))  

wt =9
wd=np.arange(0,360,30)
ws=np.arange(5,26,7)
    
site = Hornsrev1Site()
x_in, y_in = site.initial_position.T
xNew=np.array([7*80*i for i in range(3)])
yNew=np.array([7*80*i for i in range(3)])
x_new=np.array([xNew[0],xNew[0],xNew[0],xNew[1],xNew[1],xNew[1],xNew[2],xNew[2],xNew[2]])
y_new=np.array([yNew[0],yNew[1],yNew[2],yNew[0],yNew[1],yNew[2],yNew[0],yNew[1],yNew[2]])
x_newsite,y_newsite=x_new[:(wt)],y_new[:(wt)]
#-------------------------------------------------------



def wt_simulation(c,wt,wd,ws,l,k,TI_effMax,TI_NMT,x,y):
   
    site = Hornsrev1Site() 
    windTurbines = V80()
    site.ds['TI']=TI_NMT
    
    windTurbines.powerCtFunction = PowerCtFunctionList(
    key='operating',
    powerCtFunction_lst=[PowerCtTabular(ws=[0, 100], power=[0, 0], power_unit='w', ct=[0, 0]), # 0=No power and ct
                         V80().powerCtFunction], # 1=Normal operation
    default_value=1) 
     

#--------------------------
    
    power=0
    
    if (all(c <= 0.5)==True):
                power=0
    else:

        operating = np.ones(wt) # shape=(#wt,wd,ws)
        operating[c <= 0.5]=0
        
        wf_model =  All2AllIterative(site,windTurbines,
                                          wake_deficitModel=NOJDeficit(),
                                           blockage_deficitModel=SelfSimilarityDeficit(),
                                          superpositionModel=LinearSum(),
                                          
                                          turbulenceModel=GCLTurbulence())
    
        # run wind farm simulation
        sim_res = wf_model(
            x, y, # wind turbine positions
            h=None, # wind turbine heights (defaults to the heights defined in windTurbines)
            wd=l, # Wind direction (defaults to site.default_wd (0,1,...,360 if not overriden))
            ws=k, # Wind speed (defaults to site.default_ws (3,4,...,25m/s if not overriden))
            operating=operating)
        
    
        for i in range(len(x_newsite)):
            sim_res.Power[i] =sim_res.Power[i]-1e9*(max(0,sim_res.TI_eff[i]-TI_effMax))**2
        
        power=np.sum(sim_res.Power)
        
    return (float(-power))
 


# =============================================================================


def simulation(wt,wd,ws,TI_max,x,y):
    
    C_result=np.zeros((len(wd),len(ws),wt))
    power=np.zeros((len(wd),len(ws),1))
    status=np.zeros((len(wd),len(ws),1))
    
#-----------------------------------------------    
    sigma1=np.zeros((len(ws)))      # Defining NMT Modified Method
    TI_NMT=np.zeros((len(ws)))
    Iref=0.14
    
    for i in range(len(ws)): 
    
        sigma1[i]=Iref*(0.75*ws[i]+5.6)
        TI_NMT[i]=(sigma1[i]/ws[i]+0.1)/2

#---------------------------------------------------
    
    for l in range(len(wd)):
        for k in range(len(ws)):
            TI_effMax=TI_max[l][k]
            x0 = np.full((wt),0.5)
            bounds=np.full((wt,2),(0,1)).reshape(-1, 2)             
            res= minimize(wt_simulation, x0=x0, bounds=bounds,options={'maxiter': 1e6, 'disp': True},args=(wt,wd,ws,wd[l],ws[k],TI_effMax,TI_NMT[k],x,y,))
            C_result[l][k]=res.x
            power[l][k]=-res.fun
            status[l][k]=res.success
            
    C_result=C_result.reshape((12,3,9))  
    C=np.zeros((9,12,3))
    for l in range(12):
        for k in range(3):
            for i in range(9):
                C[i][l][k]=C_result[l][k][i]
            
                
    def AEP_Out(c,wt,wd,ws,TI_effMax,TI_NMT,x,y):
       
        windTurbines = V80()
        site.ds['ws'] = xr.DataArray(np.arange(5,26,7) , dims=['ws'])
        site.ds['TI'] = xr.DataArray(TI_NMT, dims=['ws'])
    
              
        windTurbines.powerCtFunction = PowerCtFunctionList(
        key='operating',
        powerCtFunction_lst=[PowerCtTabular(ws=[0, 100], power=[0, 0], power_unit='w', ct=[0, 0]), # 0=No power and ct
                              windTurbines.powerCtFunction], # 1=Normal operation
        default_value=1)
    
        operating = np.ones((wt,len(wd),len(ws))) # shape=(#wt,wd,ws)
        operating[c <= 0.5]=0
        
        wf_model =  All2AllIterative(site,windTurbines,
                                      wake_deficitModel=NOJDeficit(),
                                      blockage_deficitModel=SelfSimilarityDeficit(),
                                      superpositionModel=LinearSum(),
                                      turbulenceModel=GCLTurbulence())

        sim_res = wf_model(
            x, y, # wind turbine positions
            h=None, # wind turbine heights (defaults to the heights defined in windTurbines)
            wd=wd, # Wind direction (defaults to site.default_wd (0,1,...,360 if not overriden))
            ws=ws, # Wind speed (defaults to site.default_ws (3,4,...,25m/s if not overriden))
            operating=operating)
        
        for i in range(wt):
            for l in range(len(wd)):
                for k in range(len(ws)):
                      if sim_res.TI_eff[i][l][k]-TI_max[l][k] > 0 :
                          sim_res.Power[i][l][k]=sim_res.Power[i][l][k]-1e9*(sim_res.TI_eff[i][l][k]-TI_max[l][k])**2
        
        AEP = float(sim_res.aep().sum())
        
        return(AEP,operating)
     
    AEP,oper=AEP_Out(C, wt, wd, ws,TI_effMax, TI_NMT, x, y)
    return(AEP,oper)


# =============================================================================
#                            Layout optmization
# =============================================================================



def objective(x,y):
    

    
    windTurbines = V80()
    
    wf_model =  All2AllIterative(site,windTurbines,
                                 wake_deficitModel=NOJDeficit(),
                                 blockage_deficitModel=SelfSimilarityDeficit(),
                                  superpositionModel=LinearSum(),
                                  turbulenceModel=GCLTurbulence()
                                 )
    sim_res = wf_model(
            x, y, # wind turbine positions
            h=None, # wind turbine heights (defaults to the heights defined in windTurbines)
            wd=np.arange(5,360,30), # Wind direction (defaults to site.default_wd (0,1,...,360 if not overriden))
            ws=np.arange(4,26), # Wind speed (defaults to site.default_ws (3,4,...,25m/s if not overriden))
            )
    
    return(float(sim_res.aep().sum()))
# =============================================================================
# 
# =============================================================================

Emax=100000
runs=40
D=80
DistMin=5*D
iteration=0
result=[]
result_hat=[]
optimized_layout=[]
optimized_layout_x=[]
optimized_layout_y=[]
cpu=[]
result_iter=[]
optimized_layout_iter_x=[]
optimized_layout_iter_y=[]
improve_result=[]
Final_operating=[]

while iteration < runs :
    
    t0 = time.perf_counter()
    objective_old=objective(x_newsite,y_newsite) 
    x0,y0=x_newsite.copy(),y_newsite.copy()
    x_temp,y_temp=x_newsite.copy(),y_newsite.copy()
    Aep_hat = 0
    feasibilty=False
    Improve_flag=False
    stop=0
    
    while stop < Emax:
        
        while feasibilty==False:
            
            if Improve_flag==False:
                
                selected_wt=np.random.randint(low=0,high=9)
                gamma = np.random.random() * 2*D
                theta = np.random.random() * 2 * np.pi
                delta_x,delta_y =gamma * np.cos(theta) , gamma * np.sin(theta)
                x_Move = x0[selected_wt] + delta_x
                y_Move = y0[selected_wt] + delta_y

            else:
                
                gamma = np.random.random() * 2*D    
                delta_x,delta_y =gamma * np.cos(theta) , gamma * np.sin(theta)
                x_Move += delta_x
                y_Move += delta_y

                
            feasibilty=True
            if ((x_Move < 0) | (x_Move > 1120) | (y_Move < 0) | (y_Move > 1120)):
                feasibilty=False
                Improve_flag=False
                    
            if feasibilty==True:        
                for j in range(9):
                    if (selected_wt!=j):
                    
                        if (np.sqrt(abs((x_Move ** 2 - x0[j] ** 2) + (y_Move ** 2 - y0[j] ** 2))) < DistMin) :
                            feasibilty=False
                            Improve_flag=False
                            break
                        

    
    #--------------------------------------------------
        x_temp[selected_wt] = x_Move
        y_temp[selected_wt] = y_Move
        objective_new = objective (x_temp,y_temp)
        
        if objective_new > objective_old :
            
            improve_result.append (objective_new)
# =============================================================================
#             initial Layout + sector management optmization
# =============================================================================
            if Aep_hat == 0 :    
                
                x0[selected_wt] = x_temp[selected_wt].copy()  
                y0[selected_wt] = y_temp[selected_wt].copy()
                objective_old = objective_new
                Aep_hat,Operating = simulation(wt,wd,ws,TI_max,x0,y0)
                print(f'Aep_hat[0] = {Aep_hat} , Operating[0] = {Operating}')
# =============================================================================
#  Sector Management after 100 layout optmization
# =============================================================================
            elif len(improve_result) % 100 == 0 :
                
               
                sector_aep_temp,sector_oper_temp = simulation(wt,wd,ws,TI_max,x_temp,y_temp)
                    
                if sector_aep_temp > Aep_hat:
                        
                    x0[selected_wt] = x_temp[selected_wt].copy()
                    y0[selected_wt] = y_temp[selected_wt].copy()
                    objective_old = objective_new                        
                    Aep_hat = sector_aep_temp
                    Operating = sector_oper_temp 
                    print(f'Aep_hat{stop} = {Aep_hat} , Operating{stop} = {Operating}')

                    print(objective_old)
                    print(Aep_hat)
                    print(Operating)
                    print()
                        
            else:
                    
                    x0[selected_wt] = x_temp[selected_wt].copy()
                    y0[selected_wt] = y_temp[selected_wt].copy()
                    objective_old = objective_new
                    print(f'{stop} : {objective_old}')

                    
            Improve_flag=True
        
        else:
            x_temp[ selected_wt] = x0[selected_wt].copy()
            y_temp [selected_wt] = y0[selected_wt].copy()
            Improve_flag=False
            
        
        feasibilty=False    
        stop +=1

        
    result.append(objective_old)
    result_hat.append(Aep_hat)
    optimized_layout_x.append(x0)
    optimized_layout_y.append(y0)
    h=np.array(Operating).reshape(-1)
    Final_operating.append(h)
    cputime=round(time.perf_counter() - t0)
    cpu.append(cputime)
    iteration +=1

    out1=pd.DataFrame(result)
    out1.to_excel('Result7D.xlsx')                # Objective Function
    out1=pd.DataFrame(result_hat)
    out1.to_excel('Result_hat7D.xlsx')                # Objective Function
    out1=pd.DataFrame(optimized_layout_x)
    out1.to_excel('layout_x7D.xlsx')  
    out1=pd.DataFrame(optimized_layout_y)
    out1.to_excel('layout_y7D.xlsx') 
    out1=pd.DataFrame(Final_operating)
    out1.to_excel('Operating7D.xlsx')  
    out1=pd.DataFrame(cpu)
    out1.to_excel('cpu7D.xlsx')
    
    



