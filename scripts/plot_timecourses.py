"""Plot contact and hydrogen-bond series with five-nanosecond block means."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/figures';OUT.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(3,2,figsize=(11,10))
for j,kind in enumerate(['contacts','hydrogen_bonds']):
    a=np.loadtxt(ROOT/f'data/Fig_R12/R12_WT_relaxed_{kind}_0to50ns_10ps.csv',delimiter=',',skiprows=1)
    means=[]
    for k in range(10):
        mask=(a[:,0]>=k*5)&((a[:,0]<(k+1)*5) if k<9 else (a[:,0]<=50))
        means.append(a[mask,1:].mean(axis=0))
    means=np.array(means)
    for n in range(3):
        axes[j,0].plot(a[:,0],a[:,n+1],alpha=.35,lw=.4,label=f'Replica {n+1}')
        axes[j,1].plot(np.arange(10)*5+2.5,means[:,n],'.-',label=f'Replica {n+1}')
    for ax in axes[j]:ax.set(xlabel='Time (ns)',ylabel=kind.replace('_',' '));ax.legend()
    right=axes[j,1].twinx();right.plot(np.arange(10)*5+2.5,means[:,3],color='#BE5128',label='Sum')
    mean=means[:,3].mean();sd=means[:,3].std(ddof=1);right.axhspan(mean-sd,mean+sd,color='#BE5128',alpha=.10);right.set_ylabel('Summed block mean')
b=np.loadtxt(ROOT/'data/Fig_R12/R12b_WT_tensioned_contacts_cutoff0p6nm_0to50ns_raw_and_moving_mean.csv',delimiter=',',skiprows=1)
axes[2,0].plot(b[:,0],b[:,1],alpha=.25,lw=.3);axes[2,0].plot(b[:,0],b[:,2],lw=1);axes[2,0].set(xlabel='Time (ns)',ylabel='Tensioned-DNA contacts');axes[2,1].axis('off')
fig.tight_layout();fig.savefig(OUT/'contacts_hydrogen_bonds_timecourses.png',dpi=180);plt.close(fig)
print('Saved time-course plots.')
