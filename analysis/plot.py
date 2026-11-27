import json, sys, numpy as np, matplotlib, matplotlib.ticker; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out=sys.argv[1]; r=json.load(open(sys.argv[2]))
A,S=r['adam'],r['sym']; col={'train':'#1f77b4','val1':'#ff7f0e','val2':'#2ca02c'}
fig,ax=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
for a,D,title,n in ((ax[0],A,f"Adam (lr 1e-3): {len(A['train'])-1:,} steps, {A['secs']:.0f} s",None),
                    (ax[1],S,f"symbfitrs: {S['steps']} accepted steps, {S['ms']:.1f} ms",None)):
    for k,l in (('train','train (60% of angles)'),('val1','validation 1'),('val2','validation 2')):
        y=np.array(D[k]); a.plot(np.arange(len(y)),y,color=col[k],label=l,lw=1.8)
    a.plot(np.arange(len(D['all'])),D['all'],color='k',ls=':',lw=1.5,label='all-angle refit (train data)')
    a.axvline(D['best'],color='grey',ls='--',lw=1); a.axvline(D['stop'],color='crimson',lw=1.2)
    a.set_xscale('symlog',linthresh=1); a.set_xlim(left=0); a.set_yscale('log'); a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f')); a.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter('%.2f')); a.set_xlabel('accepted optimiser step'); a.set_title(title,fontsize=11)
    a.spines[['top','right']].set_visible(False)
    a.annotate(f"val1 best: {D['best']}",(D['best'],1),xycoords=('data','axes fraction'),xytext=(3,-12),textcoords='offset points',fontsize=8,color='grey')
    a.annotate(f"selected stop: {D['stop']}",(D['stop'],0),xycoords=('data','axes fraction'),xytext=(3,6),textcoords='offset points',fontsize=8,color='crimson')
ax[0].set_ylabel('MSE (responses normalised to RMS 1)'); ax[0].legend(fontsize=8)
fig.suptitle(f"Generalised-Gaussian fit, Stringer 2019 neuron {r['cell_id']} (SF {r['sf']:.2f}, C {r['c']:.2f})  —  test R²: "
             f"Adam {r['test_r2']['adam_stop']:.3f}, symbfitrs {r['test_r2']['sym_capped']:.3f}",fontsize=11)
plt.tight_layout(); plt.savefig(out,dpi=140)
print(r['test_r2']); print({k:(A[k][0],min(A[k]),A[k][-1]) for k in ('train','val1','val2')}); print({k:(S[k][0],min(S[k]),S[k][-1]) for k in ('train','val1','val2')})
