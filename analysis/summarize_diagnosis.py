import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=np.load('analysis/diagnosis.npz',allow_pickle=True); f=np.load('analysis/diagnosis_followup.npz'); s=np.load('analysis/smooth_bounds.npz',allow_pickle=True)
ab=d['ab']; L_a,L_s=d['L_a'],d['L_s']; M={'LM, original (clipped model)':L_s,'LM after reviving dead parameters':f['L_rev'],'LM, smooth-bounds model':s['L_new'],'best of 16 LM starts':d['best16']}
COL=['#8a8a8a','#1a5fb4','#c05a00','#26a269']; idx=np.where(ab)[0]
fig,ax=plt.subplots(2,2,figsize=(13,9.5))
for (lab,L),c in zip(M.items(),COL):
    g=np.sort(100*(L/L_a-1)); ax[0,0].plot(g,np.arange(1,len(g)+1)/len(g),color=c,lw=2,label=lab)
ax[0,0].axvline(0,color='k',lw=.8); ax[0,0].set_xlim(-3,6); ax[0,0].set_xlabel('final train MSE vs Adam (% above Adam; negative = LM lower)'); ax[0,0].set_ylabel('fraction of 300 neurons')
ax[0,0].set_title('Final objective relative to Adam, 300 neurons'); ax[0,0].legend(fontsize=8,loc='lower right')
gap=np.log(L_s[idx])-np.log(L_a[idx])
for k,((lab,L),c) in enumerate(list(zip(M.items(),COL))[1:]):
    cl=(np.log(L_s[idx])-np.log(np.minimum(L,L_s)[idx]))/gap; ax[0,1].scatter(np.full(len(cl),k)+np.random.default_rng(k).normal(0,.06,len(cl)),cl*100,color=c,s=22,alpha=.8); ax[0,1].hlines(np.median(cl)*100,k-.3,k+.3,color='k')
ax[0,1].set_xticks(range(3)); ax[0,1].set_xticklabels(['revive dead\nparameters','smooth-bounds\nmodel','best of 16\nstarts'],fontsize=9); ax[0,1].axhline(100,color='k',ls=':',lw=.8); ax[0,1].set_ylim(-5,115)
ax[0,1].set_ylabel('% of the gap to Adam closed'); ax[0,1].set_title(f'The {len(idx)} neurons where Adam is >1% lower (bars: median)')
e=int(d['ex'][0]); pos=int(np.where(idx==e)[0][0]); tr=d['traces_ab'][pos]; h=d['hist_ab'][:,pos]
ax[1,0].plot(np.arange(1,len(h)+1),h,color='#c05a00',label='Adam'); ax[1,0].plot(np.arange(len(tr))+1,tr,color='#1a5fb4',marker='o',ms=3,label='symbfitrs (accepted steps)')
ax[1,0].axhline(L_a[e],color='#c05a00',ls=':',lw=.8); ax[1,0].axhline(L_s[e],color='#1a5fb4',ls=':',lw=.8); ax[1,0].set_xscale('log'); ax[1,0].set_yscale('log')
ax[1,0].set_xlabel('step (Adam steps; symbfitrs accepted steps)'); ax[1,0].set_ylabel('train MSE'); ax[1,0].legend(); ax[1,0].set_title(f'Largest gap: neuron row {int(d["rows"][e])}, LM {100*(L_s[e]/L_a[e]-1):.0f}% above Adam')
fd=d['first_dead']; cats=[('never',fd<0),('already at\nthe start',fd==0),('LM step 1',fd==1),('LM steps 2-5',(fd>=2)&(fd<=5)),('later',fd>5)]
ax[1,1].bar([c[0] for c in cats],[100*c[1].mean() for c in cats],color='#1a5fb4',alpha=.85); ax[1,1].set_ylabel('% of 300 neurons'); ax[1,1].tick_params(axis='x',labelsize=8)
ax[1,1].set_title('When LM first pushes a parameter outside its clip range\n(zero gradient there)')
for a in ax.flat: a.spines[['top','right']].set_visible(False)
fig.suptitle('Why Adam ends lower than LM in a minority of neurons (generalised-Gaussian model, 300 Stringer neurons)',fontsize=12); plt.tight_layout(); plt.savefig('analysis/diagnosis.png',dpi=140)
