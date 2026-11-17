"""Recalculate four PMF projections using stored offsets or a fresh WHAM solve."""
from pathlib import Path
from types import SimpleNamespace
import argparse,csv,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pmf_projection_common as pmf
def read(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--solve',action='store_true');opt=parser.parse_args()
    out=ROOT/'results/pmf';out.mkdir(parents=True,exist_ok=True);report=[]
    beta=1/(pmf.K_B_KJ_MOL_K*310);k_values=(500,500,1000,500)
    for branch in ['relaxed','tensioned']:
        weights=None;start=time.perf_counter()
        for coord in ['x_complex','theta','idr_x','chainA_x']:
            args=SimpleNamespace(root=ROOT/'inputs/pmf',coordinate=coord,pullx_name='production_pullx.xvg',tpr_name='production.tpr',trajectory_name='production.xtc',com_cache_name='pmf_projection_com.xvg',refresh_com_cache=False,gmx='gmx',axis_origin='initial',dna_reference_group=None,protein_complex_group=None,protein_b2_group=None,protein_a_group=None)
            windows=pmf.read_metadata(args.root,branch,coord,args.pullx_name,False)
            samples=pmf.read_samples(args,args.root,windows,branch,coord,0,None,1)
            assert len(windows)==16 and len(samples)==48016
            if weights is None:
                if opt.solve:
                    print('Solving WHAM:',branch,flush=True)
                    offsets,weights,iterations,delta=pmf.solve_wham(samples,windows,beta,k_values,10000,1e-7)
                    assert delta<1e-7,(branch,iterations,delta)
                else:
                    offsets=np.array([float(r['offset_dimensionless']) for r in read(ROOT/f'reference/pmf/x_complex/{branch}/wham_window_offsets.tsv')])
                    x=np.array([s.subgroup_coords for s in samples]);centers=np.array([w.centers for w in windows])
                    bias=beta*.5*np.sum(np.array(k_values)[None,None,:]*(x[:,None,:]-centers[None,:,:])**2,axis=2)
                    logw=-pmf.numpy_logsumexp(np.log([w.count for w in windows])[None,:]+offsets[None,:]-bias,axis=1)
                    logw-=pmf.numpy_logsumexp(logw,axis=0);weights=np.exp(logw).tolist();iterations=0;delta=None
            rows=pmf.histogram_pmf(samples,weights,beta,100)
            reference=read(ROOT/f'reference/pmf/{coord}/{branch}/{coord}_pmf.tsv')
            err=float(np.max(np.abs(np.array([r[1] for r in rows])-np.array([float(r['pmf_shifted_kj_mol']) for r in reference]))))
            assert err<1e-6,(branch,coord,err)
            with (out/f'{branch}_{coord}.tsv').open('w',newline='') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['bin_center','pmf_kJ_mol','probability_mass','raw_count']);w.writerows(rows)
            report.append(dict(branch=branch,coordinate=coord,samples=len(samples),mode='solve' if opt.solve else 'stored_offsets',iterations=iterations,final_delta=delta,max_PMF_error_kJ_mol=err))
            print(branch,coord,'maximum difference',err,flush=True)
        print(branch,'seconds',round(time.perf_counter()-start,2),flush=True)
    (out/('solve_verification.json' if opt.solve else 'projection_verification.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
