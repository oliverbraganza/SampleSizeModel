# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:21:39 2019

@author: Oliver Braganza
"""
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.power as getpower
import matplotlib as mpl

# make pdf text illustrator readable
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

alpha = 0.005	 

def getESS(b,d,IF):
    """ 
    Calculate ESS for
    b: baserate of true hypotheses (between 0 and 1),
    d: Effect size (Cohens d),
    IF: Income factor (# of sample pairs purchasable per publication)
    """
    SS = np.arange(4,1000,2)
    Power = np.zeros(len(SS))
    falsePR = np.zeros(len(SS))
    truePR = np.zeros(len(SS))
    totalPR = np.zeros(len(SS))
    Income = np.zeros(len(SS))
    Profit = np.zeros(len(SS))
    for i,s in enumerate(SS):
        ''' 1-sample t-test '''
        # analysis = getpower.TTestPower()
        # Power[i] = analysis.solve_power(effect_size=d, nobs=s, alpha=alpha, power=None, alternative='two-sided')
        ''' 2-sample t-test '''
        analysis = getpower.TTestIndPower()
        Power[i] = analysis.solve_power(effect_size=d, nobs1=s, ratio=1.0, alpha=alpha, power=None, alternative='two-sided')
        
        falsePR[i] = alpha * (1-b)
        truePR[i] = Power[i] * b
        totalPR[i] = falsePR[i] + truePR[i]
        Income[i] = totalPR[i] * IF
        Profit[i] = Income[i] - s
    ESSidx = np.argmax(Profit)
    ESS = SS[ESSidx]
    SSSidx = (np.abs(Power-0.8)).argmin()
    SSS = SS[SSSidx]
    TPR_ESS = totalPR[ESSidx]
    PPV_ESS =  truePR[ESSidx]/totalPR[ESSidx]
    PPV_SSS =  truePR[SSSidx]/totalPR[SSSidx]
    Power_ESS = Power[ESSidx]
    
    '''
    ESS = equilibrium sample size (sample size at which Profit is maximal)
    SSS = scientifically appropriate sample size (with power=80%)
    TPR_ESS = total positive rate at ESS (describes published literature)
    PPV_ESS = positive predictive value at ESS
    Power_ESS = power at ESS
    PPV_SSS, positive predictive value at SSS
    Income = vector of income for each tested sample size
    SS = vector of tested sample sizes
    Profit = vector of profit for each tested sample size
    '''
    return ESS, SSS, TPR_ESS, PPV_ESS, Power_ESS, PPV_SSS, Income, SS, Profit

def Fig1():
    
    f, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(1, 5, figsize=(8, 2))
    f.subplots_adjust(hspace=.3, wspace=.6, left=0.1, right=0.9)
    
    # panel A
    d = 0.5
    IF = 200
    ESS, SSS, TPR_ESS, PPV_ESS, P_ESS, PPV_SSS, Income, SS, Profit05 = getESS(0.5,d,IF)
    ax1.plot(SS, Income, c='g', label='b=0.5')
    ESS, SSS, TPR_ESS, PPV_ESS, P_ESS, PPV_SSS, Income, SS, Profit02 = getESS(0.2,d,IF)
    ax1.plot(SS, Income, c='b', label='b=0.2')
    ax1.plot(SS, SS, c='k')
    ax1.set_xlim([0,200])
    ax1.set_ylim([0,200])
    ax1.set_ylabel('Income, Cost (MU)')
    ax1.set_xlabel('s')
    ax1.legend()
    
    # panel B
    wB = np.linspace(0, 1, 11)
    wESS = np.empty((11))
    wSSS = np.empty((11))
    wP = np.empty((11))
    incidence = np.empty((11))
    wPPV = np.empty((11))
    d = 0.5
    IF = 200
    ax2.plot(SS, Profit05, c='g', label='b=0.5')
    ax2.plot(SS, Profit02, c='b', label='b=0.2')
    for i, b in enumerate(wB):
        ESS, SSS, TPR_ESS, PPV_ESS, P_ESS, PPV_SSS, Income, SS, Profit = getESS(b,d,IF)
        ax2.plot(SS, Profit, c='k', zorder=0, alpha=0.5)
        wESS[i] = ESS
        wP[i] = P_ESS
        incidence[i] = 1/ESS * TPR_ESS
        wSSS[i] = SSS
        wPPV[i] = PPV_ESS
    ax2.set_xlim([0,200])
    ax2.set_ylim([0,150])
    ax2.set_ylabel('Profit (MU)')
    ax2.set_xlabel('s')
    
    # panel C
    ax3.plot(wB, wESS, c='k')
    ax3.plot([0.2,0.2],[0,wESS[-1]], c='b', alpha=0.5)
    ax3.plot([0.5,0.5],[0,wESS[-1]], c='g', alpha=0.5)
    ax3.set_ylim([0,wESS[-1]])
    ax3.set_xlim([0,1])
    ax3.set_ylabel('ESS')
    ax3.set_xlabel('b')
    
    # panel D
    ax4.plot(wB, wP, c='k')
    ax4.set_xlim([0,1])
    ax4.set_ylabel('Power(ESS)')
    ax4.plot([0.2,0.2],[0,1], c='b', alpha=0.5)
    ax4.plot([0.5,0.5],[0,1], c='g', alpha=0.5)
    ax4.set_xlabel('b')
    ax4.set_ylim([0,1])
    
    # panel E
    x_bins = np.linspace(0,1,10) 
    ax5.hist(wP, bins=x_bins, weights=incidence, color='k')
    ax5.set_xlabel('P(ESS)')
    ax5.set_ylabel('# of studies')
    
    f.savefig('SSG1.pdf')

def Fig2():
    '''
    relation of ESS and Power to b for various values of d and IF     
    '''
    wD=[0.1,0.2,0.5,1,1.5]
    wB = np.linspace(0, 1, 20)
    wIF = [10,50,100,200,500,1000]
    wwwESS = np.empty((len(wD),len(wIF),len(wB)))
    wwwP = np.empty((len(wD),len(wIF),len(wB)))
    for d_idx,d in enumerate(wD):
        for b_idx,b in enumerate(wB):
            for IF_idx,IF in enumerate(wIF):
                wwwESS[d_idx,IF_idx,b_idx], SSS, TPR_ESS, PPV_ESS, wwwP[d_idx,IF_idx,b_idx], PPV_SSS, I, SS, Pr = getESS(b,d,IF)
     
    f, ax_array = plt.subplots(10, 6, figsize=(8, 12))
    f.subplots_adjust(hspace=.3, wspace=.3, left=0.1, right=0.9)           
    for d_idx,d in enumerate(wD):   
        for IF_idx,IF in enumerate(wIF):
            ax = ax_array[d_idx,IF_idx]
            ax.plot(wB, wwwESS[d_idx,IF_idx], c='k', label='ESS')
            ax.set_xlim([0,1])
            ax.set_ylim([0, np.max(wwwESS)])
            if d_idx == 0:
                ax.set_title('IF ' + str(IF))
                # ax.set_xlabel('b')
            if d_idx < len(wD)-1:
                ax.set_xticklabels([])
            if IF_idx>0:
                ax.set_yticklabels([])
            else:    
                ax.set_ylabel('ESS')
                   
    for d_idx,d in enumerate(wD):   
        for IF_idx,IF in enumerate(wIF):
           ax = ax_array[d_idx+5,IF_idx]
           ax.plot(wB, wwwP[d_idx,IF_idx], c='k', label='ESS')
           ax.set_xlim([0,1])
           ax.set_ylim([0,1])
           if d_idx == 0:
               ax.set_title('IF ' + str(IF))
           # ax.set_xlabel('b')
           if d_idx < len(wD)-1:
               ax.set_xticklabels([])
           if IF_idx>0:
               ax.set_yticklabels([])
           else:    
               ax.set_ylabel('Power(ESS)')

    f.savefig('SSG2.pdf')
    
def Fig3():
    '''
    Generates random input distributions for d, b and IF, 
    draws random combinations and computes the resulting ESS and Power
    plots histogram of resulting power and mean PPV
    all histograms normalized to sum to 1    
    '''
    
    niches = 1000
    inp_weights = np.full((niches),1/niches)
    wESS = np.empty((niches))
    wPPV = np.empty((niches))
    wP = np.empty((niches))
    incidence = np.empty((niches))
    wD = np.random.gamma(3.5, 0.2, size=niches)
    
    f, ax_array = plt.subplots(5, 5, figsize=(8, 7))
    f.subplots_adjust(hspace=.5, wspace=.6, left=0.1, right=0.9)
    
    dictionary_B = {}
    dictionary_IF = {}
    dictionary = {}
    wBs = ['B_uniform', 'B_bimodal', 'B_low', 'B_bimodal_low']
    wIFs = ['IF_uniform', 'IF_low', 'IF_medium', 'IF_high']
    
    dictionary_B['B_uniform'] = np.random.rand(niches)
    dictionary_B['B_low'] = np.random.beta(1.1,10, size=niches)
    dictionary_B['B_bimodal'] = np.concatenate((np.random.beta(1.1, 10, size=int(niches/2)),np.random.beta(10, 1.1, size=int(niches/2))))
    dictionary_B['B_bimodal_low'] = np.concatenate((np.random.beta(1.1, 10, size=int(niches*0.9)),np.random.beta(10, 1.1, size=int(niches*0.1))))
    
    dictionary_IF['IF_uniform'] = np.random.rand(niches) * 1000
    dictionary_IF['IF_low'] = np.random.beta(1.1, 10, size=niches) * 1000
    dictionary_IF['IF_medium'] = np.random.beta(10, 10, size=niches) * 1000
    dictionary_IF['IF_high'] = np.random.beta(10, 1.1, size=niches) * 1000

    ax = ax_array[0,0]
    x_bins_wD = np.linspace(0,2,20)
    ax.hist(wD, bins=x_bins_wD, weights=inp_weights, color='grey')
    ax.set_xlim([0,2])
    ax.set_xlabel('d')
    ax.set_yticklabels([])
    
    for wB_idx, wB in enumerate(wBs):
        ax = ax_array[0,wB_idx+1]
        x_bins_wB = np.linspace(0,1,20)
        ax.hist(dictionary_B[wB], bins=x_bins_wB, weights=inp_weights, color='grey')
        ax.set_xlim([0,1])
        ax.set_xlabel('b')
        ax.set_yticklabels([])
    
    for wIF_idx, wIF in enumerate(wIFs):    
        ax = ax_array[wIF_idx+1,0]
        x_bins_wIF = np.linspace(0,1000,20)
        ax.hist(dictionary_IF[wIF], bins=x_bins_wIF, weights=inp_weights, color='grey')
        ax.set_xlim([0,1000])
        ax.set_xlabel('IF')
        ax.set_yticklabels([])
    
    for wB_idx, wB in enumerate(wBs):
        dictionary[wB] = {}
        for wIF_idx, wIF in enumerate(wIFs):
            
            ax = ax_array[wIF_idx+1,wB_idx+1]
            for i, d in enumerate(wD):
                b = dictionary_B[wB][i]
                IF = dictionary_IF[wIF][i]
                ESS, SSS, TPR_ESS, PPV_ESS, P_ESS, PPV_SSS, Income, SS, Profit = getESS(b,d,IF)
                wESS[i] = ESS
                wP[i] = P_ESS
                incidence[i] = 1/ESS * TPR_ESS
                wPPV[i] = PPV_ESS
                dictionary[wB][wIF] = wP
            
            norm = incidence.sum()
            x_bins_wP = np.linspace(0,1,20)
            ax.hist(wP, bins=x_bins_wP, weights=incidence/norm, color='k')
            #ax.scatter(wP, incidence)
            ax.set_xlim([0,1])
            ax.set_xlabel('P')
            ax.text(0.7,1,"%3.2f"% np.average(wPPV, weights=incidence), transform=ax.transAxes)
            ax.set_yticklabels([])
      
    f.savefig('SSG3.pdf')
    