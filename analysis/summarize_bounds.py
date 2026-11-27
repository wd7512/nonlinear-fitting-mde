import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
b=np.load('analysis/bounds_experiments.npz',allow_pickle=True); n=[str(x) for x in b['names']]; L=b['L']; ab=b['ab']; best=b['best']
La=L[n.index('Adam 15000 steps, clipped model')]
rows=[('Adam (15,000 steps), clipped model','Adam 15000 steps, clipped model','#c05a00'),
 ('symbfitrs LM, clipped model','symbfitrs LM, clipped model (original)','#1a5fb4'),
 ('SciPy lm, clipped, no bounds','SciPy lm, clipped model, no bounds','#8a8a8a'),
 ('SciPy trf, clipped, no bounds','SciPy trf, clipped model, no bounds','#8a8a8a'),
 ('SciPy dogbox, clipped, no bounds','SciPy dogbox, clipped model, no bounds','#8a8a8a'),
 ('SciPy trf, TRUE bounds','SciPy trf, true bounds','#26a269'),
 ('SciPy dogbox, TRUE bounds','SciPy dogbox, true bounds','#26a269')]
fig,ax=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
y=np.arange(len(rows))[::-1]
for yi,(lab,k,c) in zip(y,rows):
    l=L[n.index(k)]; ax[0].barh(yi,100*np.mean(l<=best*1.001),color=c,alpha=.9); ax[0].text(100*np.mean(l<=best*1.001)+1,yi,f'{100*np.mean(l<=best*1.001):.0f}%',va='center',fontsize=9)
    g=100*np.mean((l<=La*1.001)[ab]); ax[1].barh(yi,g,color=c,alpha=.9); ax[1].text(g+1,yi,f'{g:.0f}%',va='center',fontsize=9)
ax[0].set_yticks(y); ax[0].set_yticklabels([r[0] for r in rows]); ax[0].set_xlim(0,100); ax[0].set_xlabel('% of 300 neurons within 0.1% of the best result any method found')
ax[1].set_xlim(0,100); ax[1].set_xlabel(f'% of the {int(ab.sum())} Adam-better neurons brought within 0.1% of Adam')
ax[0].set_title('Reaching the best known objective'); ax[1].set_title('Closing the gap to Adam')
for a in ax: a.spines[['top','right']].set_visible(False)
fig.suptitle('Real bounds help: SciPy with and without bounds on the same neurons, same start, same scoring (clipped model, float64)',fontsize=11)
plt.tight_layout(); plt.savefig('analysis/bounds.png',dpi=140); print('ok')
