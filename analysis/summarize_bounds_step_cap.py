import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
c=np.load('analysis/bounds_step_cap.npz',allow_pickle=True); n=[str(x) for x in c['names']]; best=c['best']; ab=c['ab']; La=c['L_a']
u=np.load('analysis/bounds_unclipped_rust.npz',allow_pickle=True); un=[str(x) for x in u['names']]
b0=np.load('analysis/bounds_experiments.npz',allow_pickle=True); n0=[str(x) for x in b0['names']]
w=lambda L: 100*np.mean(L<=best*1.001)
# ---- Adam statistics (same neurons, same start, same scoring)
import json, torch, sys; sys.path.insert(0,'analysis')
from _neuro import trials, ROWS, gen_gauss
dg=np.load('analysis/diagnosis.npz',allow_pickle=True); rs=json.load(open('analysis/reliability_speed.json'))
best=np.minimum(best,u['L'].min(0))
cells=[trials(r) for r in ROWS]; TH=np.stack([x[0] for x in cells]); TR=np.stack([x[1][0].mean(1) for x in cells])
LO=np.array([0,0,0,0,.1,.1,.1,.1,.1,.1]); HI=np.array([2*np.pi,np.inf,np.inf,np.inf]+[2*np.pi]*4+[4.,4.])
dead=lambda P: ((P[:,1:]<LO[1:]-1e-12)|(P[:,1:]>HI[1:]+1e-12)).any(1)
P=torch.tensor(dg['P0'].astype(np.float32)).requires_grad_(); opt=torch.optim.Adam([P],lr=1e-3)
th32=torch.tensor(TH,dtype=torch.float32); y32=torch.tensor(TR,dtype=torch.float32); d0=dead(P.detach().numpy())
opt.zero_grad(); (10*((gen_gauss(th32,P).squeeze(0)-y32)**2).mean(1).mean()).backward(); P.grad.nan_to_num_(0,0,0); opt.step(); d1=dead(P.detach().numpy())
adam_step1=100*np.mean(d1&~d0); adam_end=100*np.mean(dead(dg['P_a'])); adam_start=100*np.mean(d0)
match=np.array(rs['adam_steps_to_match_sym']); adam_match=np.median(match[match>0]); adam_ms=1e3*rs['baseline']['adam_secs']/300
print(f'Adam: within 0.1% {w(La):.0f}%; first-step dead-zone entry {adam_step1:.1f}% (dead at start {adam_start:.0f}%); dead at the end {adam_end:.0f}%; median {adam_match:.0f} steps to match LM ({100*np.mean(match>0):.0f}% of neurons do); {adam_ms:.0f} ms/neuron amortised')
caps=[1e-3,3e-3,1e-2,3e-2,0.1,0.3,1.0]; INF=3.0
def curve(prefix,none_label): 
    xs=[];ys=[];st=[];dd=[]
    for cp in caps:
        k=f'{prefix}{cp:g}'
        if k in n: j=n.index(k); xs.append(cp); ys.append(w(c['L'][j])); st.append(np.median(c['steps'][j])); dd.append(100*np.mean(c['first_dead'][j]==1))
    j=n.index(none_label); xs.append(INF); ys.append(w(c['L'][j])); st.append(np.median(c['steps'][j])); dd.append(100*np.mean(c['first_dead'][j]==1)); return xs,ys,st,dd
a=curve('step cap ','LM, clipped model (current symbfitrs)'); bq=curve('bounds + step cap ','bounds (active-set)')
ux=[0.03,0.1,0.3,1.0,INF]; uk=['unclipped model, bounds + cap 0.03','unclipped model, bounds + cap 0.1','unclipped model, bounds + cap 0.3','unclipped model, bounds + cap 1.0','unclipped model, bounds']
uy=[w(u['L'][un.index(k)]) for k in uk]
fig,ax=plt.subplots(1,3,figsize=(15,4.8))
ax[0].plot(a[0],a[1],'o-',color='#8a8a8a',label='clipped model, no bounds (current)'); ax[0].plot(bq[0],bq[1],'o-',color='#1a5fb4',label='clipped model + bounds (theta unbounded)'); ax[0].plot(ux,uy,'o-',color='#26a269',label='unclipped model + bounds (bounds replace clips)')
ax[0].axhline(100*np.mean(La<=best*1.001),color='#c05a00',ls='--',label=f'Adam, 15,000 steps ({w(La):.0f}%)'); ax[0].axhline(w(b0['L'][n0.index('SciPy trf, true bounds')]),color='#6a4ce2',ls=':',label='SciPy trf, true bounds')
ax[0].set_xscale('log'); ax[0].set_xticks(caps+[INF]); ax[0].set_xticklabels([f'{x:g}' for x in caps]+['no cap'],rotation=45); ax[0].set_ylabel('% of neurons within 0.1% of the best result'); ax[0].set_xlabel('max change of any parameter per step'); ax[0].legend(fontsize=7,loc='lower left'); ax[0].set_title('Quality'); ax[0].set_ylim(50,85)
ax[1].plot(a[0],a[2],'o-',color='#8a8a8a'); ax[1].plot(bq[0],bq[2],'o-',color='#1a5fb4'); ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xticks(caps+[INF]); ax[1].set_xticklabels([f'{x:g}' for x in caps]+['no cap'],rotation=45)
ax[1].axhline(15000,color='#c05a00',ls='--',label='Adam: fixed budget, 15,000 steps'); ax[1].axhline(adam_match,color='#c05a00',ls=':',label=f'Adam: median {adam_match:.0f} steps to match LM ({100*np.mean(match>0):.0f}% of neurons)')
ax[1].set_ylim(top=3e4); ax[1].legend(fontsize=7,loc='center right'); ax[1].set_ylabel('median accepted steps'); ax[1].set_xlabel('max change of any parameter per step'); ax[1].set_title(f'Cost: LM 3.4 ms at no cap, 5 ms at 0.1, 19 ms at 0.01; Adam ~{adam_ms:.0f} ms/neuron\n(batched over 300 neurons, amortised; CPU)',fontsize=8)
ax[2].plot(a[0],a[3],'o-',color='#8a8a8a'); ax[2].set_xscale('log'); ax[2].set_xticks(caps+[INF]); ax[2].set_xticklabels([f'{x:g}' for x in caps]+['no cap'],rotation=45); ax[2].axhline(adam_step1,color='#c05a00',ls='--',label=f'Adam at its first step: {adam_step1:.1f}%'); ax[2].text(0.02,0.62,f'Adam: {adam_start:.0f}% of neurons start with a parameter\nin a dead zone and {adam_end:.0f}% end with one',transform=ax[2].transAxes,fontsize=8,color='#c05a00')
ax[2].legend(fontsize=7,loc='upper left'); ax[2].set_ylabel('% of neurons'); ax[2].set_xlabel('max change of any parameter per step')
ax[2].set_title('Parameter leaves its clip range at the first LM step\n(clipped model, no bounds)')
for x in ax: x.spines[['top','right']].set_visible(False)
fig.suptitle('Prototype: active-set bounds and a per-step cap in symbfitrs LM, 300 Stringer neurons',fontsize=12); plt.tight_layout(); plt.savefig('analysis/bounds_step_cap.png',dpi=140); print('ok')
