"""Plot replica statistics, short-range energy fractions and PMF profiles."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'data'
OUT=BASE/'results/figures'
OUT.mkdir(exist_ok=True,parents=True)

def read(p,sep=','):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f,delimiter=sep))

def main():
    rows=read(DATA/'MainText_Fig6/Fig6_group_mean_and_between_replica_SD.csv')
    reps=read(DATA/'MainText_Fig6/Fig6_replica_means.csv')
    fig,axs=plt.subplots(1,3,figsize=(11,4))
    conditions=['WT relaxed','WT tensioned','11KA relaxed','11KA tensioned']
    for ax,metric,label in zip(axs,['contacts','hbnum','interaction_energy'],['Contacts (cutoff 0.6 nm)','Hydrogen bonds','SR interaction energy (kJ/mol)']):
        for j,c in enumerate(conditions):
            r=next(r for r in rows if r['condition']==c and r['metric']==metric)
            v=[float(r['mean']) for r in reps if r['condition']==c and r['metric']==metric]
            ax.errorbar(j,float(r['mean']),yerr=float(r['sample_sd']),fmt='_',ms=22,capsize=5,color='black')
            ax.scatter(np.repeat(j,len(v)),v,s=20)
        ax.set_xticks(range(4),conditions,rotation=45,ha='right');ax.set_ylabel(label);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(OUT/'Fig6b-d_group_statistics_verification.png',dpi=200);plt.close(fig)
    r=read(DATA/'Fig_R11/R11_verified_energy_fraction.csv')
    fig,ax=plt.subplots(figsize=(7,4));ax.bar([x['configuration'] for x in r],[float(x['IDR_percent']) for x in r]);ax.set_ylabel('IDR fraction (%)');ax.set_ylim(0,100);ax.tick_params(axis='x',labelrotation=20);fig.tight_layout();fig.savefig(OUT/'R11_energy_fraction_verification.png',dpi=200);plt.close(fig)
    for coord,panel,unit in [('x_complex','Fig6f','nm'),('theta','R13','degrees'),('idr_x','R14','nm'),('chainA_x','R15','nm')]:
        fig,ax=plt.subplots(figsize=(6,4))
        for branch,color in [('relaxed','#2869ac'),('tensioned','#ce4537')]:
            rows=read(DATA/'PMF_numeric_backup'/f'{panel}_{coord}_{branch}_PMF_100bins.tsv','\t')
            xkey=next(k for k in rows[0] if k.startswith('bin_center_'))
            ax.plot([float(r[xkey]) for r in rows],[float(r['pmf_shifted_kj_mol']) for r in rows],label=branch,color=color)
        ax.set(xlabel=f'{coord} ({unit})',ylabel='PMF - own minimum (kJ/mol)');ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/f'{panel}_{coord}_numeric_backup_verification.png',dpi=200);plt.close(fig)
    print('Numerical verification plots saved to',OUT)

if __name__=='__main__':main()
