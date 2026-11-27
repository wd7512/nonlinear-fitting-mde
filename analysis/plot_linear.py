import json, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out=sys.argv[1]; r=json.load(open(sys.argv[2])); A,S=r['adam'],r['sym']
col={'train':'#1f77b4','val1':'#ff7f0e','val2':'#2ca02c'}
fig,ax=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
for a,D,title,xmax in ((ax[0],A,f"Adam (lr 1e-3): shown to step 1000 of {len(A['train'])-1:,}; ran {A['secs']:.0f} s",1000),
                       (ax[1],S,f"symbfitrs: all {S['steps']} accepted steps, {S['ms']:.1f} ms",S['steps'])):
    for k,l in (('train','train (60% of angles)'),('val1','validation 1'),('val2','validation 2')):
        a.plot(D[k][:xmax+1],color=col[k],label=l,lw=2)
    a.plot(D['all'][:xmax+1],color='k',ls=':',lw=1.5,label='all-angle refit (train data)')
    a.axvline(D['best'],color='grey',ls='--',lw=1,label=f"val1 best (step {D['best']})")
    if D['stop']<=xmax: a.axvline(D['stop'],color='crimson',lw=1.2,label=f"selected stop (step {D['stop']})")
    a.set_xlim(0,xmax); a.set_xlabel('accepted optimiser step'); a.set_title(title,fontsize=11)
    a.spines[['top','right']].set_visible(False); a.legend(fontsize=8)
ax[0].set_ylabel('MSE (responses normalised to RMS 1)')
fig.suptitle(f"Stringer 2019 neuron {r['cell_id']} (SF {r['sf']:.2f}): test R² Adam {r['test_r2']['adam_stop']:.3f}, symbfitrs {r['test_r2']['sym_capped']:.3f}")
plt.tight_layout(); plt.savefig(out,dpi=140)
