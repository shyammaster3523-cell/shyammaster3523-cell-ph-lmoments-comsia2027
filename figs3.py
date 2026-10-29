import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fitlib import *
from scipy import stats
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Liberation Sans"],"font.size":8,"axes.labelsize":8,
    "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7,"axes.linewidth":0.6,"lines.linewidth":0.9,
    "ps.fonttype":42,"pdf.fonttype":42,"mathtext.fontset":"custom","mathtext.it":"Liberation Sans:italic","mathtext.rm":"Liberation Sans"})
W=4.6
def tag(ax,s,x=0.02,ha="left"): ax.text(x,0.97,s,transform=ax.transAxes,fontsize=9,fontweight="bold",va="top",ha=ha)
# ---------- Fig 1: runtime (single panel, sidecaption) ----------
T=json.load(open("timing.json")); TT=json.load(open("tri_timing.json"))
fig,ax=plt.subplots(figsize=(2.6,2.15))
for D,mk,lab in [(T["dense"],"o","dense"),(T["spectral"],"s","spectral"),(TT,"^","sparse")]:
    p=sorted((int(k.split(',')[0])**(int(k.split(',')[1])+1),v) for k,v in D.items())
    ax.loglog([q[0] for q in p],[q[1] for q in p],mk,ms=3,mfc="none",mew=0.7,color="k",label=lab)
N=np.logspace(2.6,3.7); ax.loglog(N,1.5*(N/4096)**3,"k--",lw=0.6); ax.text(30,0.05,"slope 3",fontsize=7)
N=np.logspace(4,6.4); ax.loglog(N,0.15*(N/2.1e6),"k:",lw=0.8); ax.text(1.2e5,0.003,"slope 1",fontsize=7)
ax.set_xlabel(r"lifted dimension $N=m^{r+1}$"); ax.set_ylabel("time (s)"); ax.legend(frameon=False,loc="upper left",handletextpad=0.2)
fig.tight_layout(pad=0.3); [fig.savefig("fig1_cost."+e) for e in ("eps","pdf")]
# ---------- Fig 2: ratio diagrams ----------
if not _os.path.exists("ratio_pts.npy"):   # 40,000 random order-two acyclic laws (exact, Theorem 1)
    _rng=np.random.default_rng(0); _P=[]
    for _ in range(40000):
        _a=_rng.uniform(0,1); _u=np.exp(_rng.uniform(np.log(1e-3),0))
        _al,_S=cf1([_u,1.0],[_a,1-_a]); _l=lmoments_from_pwm(pwm_dense(_al,_S,3)); _P.append((_l[1]/_l[0],_l[2]/_l[1],_l[3]/_l[1]))
    np.save("ratio_pts.npy",np.array(_P))
P=np.load("ratio_pts.npy"); E=[]
for k in range(1,9):
    al,S=cf1(np.ones(k),np.r_[1,np.zeros(k-1)]); l=lmoments_from_pwm(pwm_tri(al,S,3)); E.append((l[1]/l[0],l[2]/l[1],l[3]/l[1]))
E=np.array(E)
def lr(x): l=sample_lmoments(x,4); return l[1]/l[0],l[2]/l[1],l[3]/l[1]
pts=[("AutoClaims",lr(load("auto")),"x",dict(color="k")),("Danish, raw",lr(load("danish")),"+",dict(color="k",ms=7)),
     ("Danish, $x-1$",lr(load("danishx")),"o",dict(color="k",mfc="none"))]
fig,ax=plt.subplots(1,2,figsize=(W,2.2))
for a in ax: a.set_rasterization_zorder(0)
ax[0].scatter(P[:,1],P[:,2],s=0.3,c="0.72",zorder=-1,rasterized=True,label="order-two acyclic PH")
t3=np.linspace(-0.05,0.95,100); ax[0].plot(t3,(5*t3**2-1)/4,"k--",lw=0.7,label=r"$\tau_4\geq(5\tau_3^2-1)/4$")
ax[0].plot(E[:,1],E[:,2],"-d",color="k",ms=2.5,lw=0.7,label=r"Erlang, $n\leq8$")
ax[1].scatter(P[:,0],P[:,1],s=0.3,c="0.72",zorder=-1,rasterized=True)
tt=np.linspace(0.5,1,100); ax[1].plot(tt,(5*tt-2)/(3*tt),"k-",lw=1.0,label=r"$(5\tau-2)/(3\tau)$")
ax[1].plot(E[:,0],E[:,1],"-d",color="k",ms=2.5,lw=0.7)
for nm,v,mk,kw in pts:
    ax[0].plot(v[1],v[2],mk,ms=kw.get("ms",5),color="k",mfc=kw.get("mfc","k"),mew=0.9,ls="none",label=nm)
    ax[1].plot(v[0],v[1],mk,ms=kw.get("ms",5),color="k",mfc=kw.get("mfc","k"),mew=0.9,ls="none")
ax[0].set_xlabel(r"L-skewness $\tau_3$"); ax[0].set_ylabel(r"L-kurtosis $\tau_4$"); ax[0].set_xlim(0,0.9); ax[0].set_ylim(0,0.8)
ax[1].set_xlabel(r"L-CV $\tau$"); ax[1].set_ylabel(r"L-skewness $\tau_3$"); ax[1].set_xlim(0.25,1); ax[1].set_ylim(0,1.05)
ax[0].legend(frameon=False,loc="upper left",fontsize=6.5,markerscale=1.5,handletextpad=0.2,borderaxespad=1.4,labelspacing=0.25)
ax[1].legend(frameon=False,loc="lower right",fontsize=7)
tag(ax[0],"a"); tag(ax[1],"b")
fig.tight_layout(pad=0.3); [fig.savefig("fig2_ratio."+e,dpi=300) for e in ("eps","pdf")]
# ---------- Fig 3: comparison ----------
A=json.load(open("auto_em3.json")); F=json.load(open("auto_fits.json"))
x=load("danish"); y=load("danishx"); n=len(x); xs=np.sort(x); surv=1-np.arange(1,n+1)/(n+1)
fx=json.load(open("danishx_fits.json")); ex=json.load(open("danishx_em3.json")); f0=json.load(open("danish_fits.json")); mu=y.mean()
fig,ax=plt.subplots(1,2,figsize=(W,2.1))
lab=[r"$\lambda_1$",r"$\lambda_2$",r"$\lambda_3$",r"$\lambda_4$",r"$\lambda_5$"]; xx=np.arange(5); w=0.27
for j,(nm,v,fc,h) in enumerate([("L-moments",F["L3"]["lerr"],"0.15",""),("EM",A["rand0"]["lerr"],"0.6",""),("classical",F["C3"]["lerr"],"white","////")]):
    ax[0].bar(xx+(j-1)*w,np.clip(v,-12,12),w,color=fc,edgecolor="k",lw=0.5,hatch=h,label=nm)
ax[0].axhline(0,color="k",lw=0.5); ax[0].set_xticks(xx); ax[0].set_xticklabels(lab); ax[0].set_ylabel("relative error (%)"); ax[0].set_ylim(-13,13)
ax[0].legend(frameon=False,loc="upper right",handlelength=1.2,handletextpad=0.3)
g=np.logspace(0,np.log10(300),300)
def sv(al,S,shift): return 1-ph_cdf(np.array(al),np.array(S),np.maximum(g-shift,0))
ax[1].loglog(xs,surv,".",ms=1.5,color="0.55",label="empirical")
ax[1].loglog(g,sv(f0["L3"]["alpha"],np.array(f0["L3"]["S"])/x.mean(),0),"k:",lw=1.0,label="PH3 L-mom., raw")
ax[1].loglog(g,sv(fx["L3"]["alpha"],np.array(fx["L3"]["S"])/mu,1),"k-",lw=1.0,label=r"PH3 L-mom., $x-1$")
ax[1].loglog(g,sv(ex["rand0"]["alpha"],ex["rand0"]["S"],1),"k-.",lw=0.9,label=r"PH3 EM, $x-1$")
c,_,sc=stats.genpareto.fit(y,floc=0); ax[1].loglog(g,stats.genpareto(c,0,sc).sf(g-1),"k--",lw=0.9,label=r"GPD, $x-1$")
ax[1].set_ylim(2e-4,1.5); ax[1].set_xlabel("loss (MDKK)"); ax[1].set_ylabel("survival probability")
ax[1].legend(frameon=False,loc="lower left",fontsize=6.5,handlelength=2.2,labelspacing=0.2)
tag(ax[0],"a"); tag(ax[1],"b",x=0.97,ha="right")
fig.tight_layout(pad=0.3); [fig.savefig("fig3_comparison."+e) for e in ("eps","pdf")]
print("ok")
