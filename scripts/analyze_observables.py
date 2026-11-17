"""Calculate replica statistics, energy fractions and time-window summaries."""
from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; OUT=ROOT/'results/statistics'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True); checks=[]; means=[];groups=defaultdict(list)
    selections=read(DATA/'MainText_Fig6/Fig6_selection_intervals.csv')
    ref=read(DATA/'MainText_Fig6/Fig6_replica_means.csv')
    for sel in selections:
        rows=read(DATA/'MainText_Fig6'/sel['data_file'])
        values=[float(r['plotted_observable']) for r in rows if float(r['retained_after_initial20percent'])==1]
        avg=float(np.mean(values));key=(sel['condition'],sel['metric']);groups[key].append(avg)
        expected=next(r for r in ref if all(r[k]==sel[k] for k in ['condition','metric','replica_id']))
        np.testing.assert_allclose(avg,float(expected['mean']),atol=1e-9,rtol=1e-12)
        means.append(dict(condition=sel['condition'],metric=sel['metric'],replica=sel['replica_id'],frames=len(values),mean=avg))
    save('replica_means.csv',means)
    summaries=[dict(condition=c,metric=m,n=len(v),mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1))) for (c,m),v in groups.items()]
    for r in summaries:
        expected=next(x for x in read(DATA/'MainText_Fig6/Fig6_group_mean_and_between_replica_SD.csv') if x['condition']==r['condition'] and x['metric']==r['metric'])
        np.testing.assert_allclose([r['mean'],r['sample_sd']],[float(expected['mean']),float(expected['sample_sd'])],atol=1e-9,rtol=1e-12)
    save('group_summary.csv',summaries);checks.append({'analysis':'replica statistics','series':len(means)})
    energies=[]
    for r in read(DATA/'Fig_R11/R11_verified_energy_fraction.csv'):
        cfg=r['configuration'];a=read(DATA/f'Fig_R11/R11_{cfg}_energy_timeseries.csv')
        idr=np.mean([float(x['IDR_Coul_SR_kJ_mol'])+float(x['IDR_LJ_SR_kJ_mol']) for x in a])
        other=np.mean([float(x['nonIDR_Coul_SR_kJ_mol'])+float(x['nonIDR_LJ_SR_kJ_mol']) for x in a])
        frac=abs(idr)/(abs(idr)+abs(other));np.testing.assert_allclose(frac,float(r['IDR_fraction']),atol=1e-12)
        energies.append(dict(configuration=cfg,IDR_mean_kJ_mol=idr,nonIDR_mean_kJ_mol=other,IDR_fraction=frac))
    save('energy_fractions.csv',energies);checks.append({'analysis':'energy fractions','configurations':len(energies)})
    for kind in ['contacts','hydrogen_bonds']:
        a=np.loadtxt(DATA/f'Fig_R12/R12_WT_relaxed_{kind}_0to50ns_10ps.csv',delimiter=',',skiprows=1)
        np.testing.assert_array_equal(a[:,4],a[:,1:4].sum(axis=1));blocks=[]
        expected=read(DATA/f'Fig_R12/R12_WT_relaxed_{kind}_5ns_window_means.csv')
        for k in range(10):
            mask=(a[:,0]>=k*5)&((a[:,0]<(k+1)*5) if k<9 else (a[:,0]<=50))
            v=a[mask,1:].mean(axis=0)
            np.testing.assert_allclose(v,[float(expected[k][x]) for x in ['replicate_01_mean','replicate_02_mean','replicate_03_mean','sum_mean']],atol=1e-9)
            blocks.append(dict(center_ns=k*5+2.5,frames=int(mask.sum()),replicate_01=v[0],replicate_02=v[1],replicate_03=v[2],sum_mean=v[3]))
        save(f'{kind}_blocks.csv',blocks)
        checks.append({'analysis':kind,'frames':len(a),'blocks':10})
    b=np.loadtxt(DATA/'Fig_R12/R12b_WT_tensioned_contacts_cutoff0p6nm_0to50ns_raw_and_moving_mean.csv',delimiter=',',skiprows=1)
    smoothed=np.convolve(np.pad(b[:,1],(25,24),mode='edge'),np.ones(50)/50,mode='valid')
    np.testing.assert_allclose(smoothed,b[:,2],atol=1e-8)
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('Verified replica statistics, energy fractions, block means and moving averages.',flush=True)
if __name__=='__main__':main()
