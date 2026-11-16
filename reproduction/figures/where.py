import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif'],'font.size':9,'pdf.fonttype':42})
AS=[2.0,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4]
req=[0]*11+[54.01,62.84,70.34,77.07,83.30,89.20]
rot=[4.56,6.52,7.04,7.48,6.64,8.08,7.32,8.40,9.84,9.40,11.0,41.72,44.48,41.36,43.48,44.20,47.20]
x=np.arange(17)
fig,ax=plt.subplots(figsize=(3.35,2.35))
for i in range(17): ax.add_patch(plt.Rectangle((i-0.5,req[i]),1,92-req[i],color='#F6E3A1',lw=0))
ax.step(np.r_[x-0.5,16.5],np.r_[req,req[-1]],where='post',color='#B8860B',lw=1.2,label='required turn')
ax.plot(x,rot,'-o',color='#2F5597',ms=3.2,lw=1.6,label='mean turn of 15 VLMs')
ax.axvline(10.5,color='k',ls=':',lw=0.8)
ax.text(5,50,'any turn fits',ha='center',fontsize=8.5,color='#8a6d00')
ax.text(13.5,14,'blocked unless\nturned enough',ha='center',fontsize=8,color='#555555')
ax.set_xticks(x[::2]); ax.set_xticklabels([f'{a:.1f}' for a in AS[::2]])
ax.set_xlim(-0.5,16.5); ax.set_ylim(0,92); ax.set_yticks([0,30,60,90])
ax.set_xlabel('A/S (opening / shoulder width)'); ax.set_ylabel('max torso rotation (deg)')
ax.legend(fontsize=7.5,loc='upper left',frameon=True,framealpha=0.9); ax.grid(ls='--',alpha=0.3)
fig.tight_layout(pad=0.3); fig.savefig('where_rotate.pdf')
