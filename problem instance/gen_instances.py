"""gen_instances.py — input_initialmap.py'nin veri-üretim fonksiyonlarıyla (matplotlib'siz)
yeni ızgara örnekleri üretir. scenario.xlsx config'ini taban alır; n_nodes/out (ve isteğe bağlı
araç ölçeği) override edilir.

  .venv/bin/python gen_instances.py
"""
from __future__ import annotations
import os, math, re
from typing import List, Dict, Any, Set, Tuple
import numpy as np
import pandas as pd

SCEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenario.xlsx")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")

# ---------- input_initialmap.py'den birebir (matplotlib'siz) ----------
def validate_square_grid(n):
    g=int(round(math.sqrt(n)))
    if g*g!=n: raise ValueError(f"kare değil: {n}")
    return g
def generate_grid_centers(gs,e):
    xs,ys=[],[]
    for r in range(gs):
        for c in range(gs):
            xs.append((c+0.5)*e); ys.append((r+0.5)*e)
    return xs,ys
def parse_node_list(x):
    if pd.isna(x) or str(x).strip()=="" : return []
    if isinstance(x,(int,float)): return [int(x)]
    return [int(t) for t in re.split(r"[,\.;\s]+",str(x).strip()) if t.isdigit()]
def _rc(n,gs): return (n-1)//gs,(n-1)%gs
def _man(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
def neighbors4(gs,blocked):
    n=gs*gs; neigh=[[] for _ in range(n)]
    for r in range(gs):
        for c in range(gs):
            nid=r*gs+c+1
            if nid in blocked: continue
            cand=[]
            if r>0: cand.append((r-1)*gs+c+1)
            if r<gs-1: cand.append((r+1)*gs+c+1)
            if c>0: cand.append(r*gs+(c-1)+1)
            if c<gs-1: cand.append(r*gs+(c+1)+1)
            neigh[nid-1]=[j for j in cand if j not in blocked]
    return neigh
def rm_roads(neigh,pairs,n):
    for (i,j) in pairs:
        if 1<=i<=n and 1<=j<=n:
            if j in neigh[i-1]: neigh[i-1].remove(j)
            if i in neigh[j-1]: neigh[j-1].remove(i)
    return neigh
def _smooth(f,ni=8,al=.6):
    f=f.astype(float).copy()
    for _ in range(ni):
        up=np.pad(f[:-1,:],((1,0),(0,0)),mode="edge"); dn=np.pad(f[1:,:],((0,1),(0,0)),mode="edge")
        lf=np.pad(f[:,:-1],((0,0),(1,0)),mode="edge"); rt=np.pad(f[:,1:],((0,0),(0,1)),mode="edge")
        f=(1-al)*f+al*.25*(up+dn+lf+rt)
    return f
def _scale(f,lo,hi):
    a,b=f.min(),f.max()
    if abs(b-a)<1e-12: return np.full_like(f,(lo+hi)/2)
    return lo+(f-a)*(hi-lo)/(b-a)
def fields(gs,vmin,vmax,dmin,dmax,rng):
    v=_scale(_smooth(rng.normal(0,1,(gs,gs))),vmin,vmax)
    d=_scale(_smooth(rng.normal(0,1,(gs,gs))),dmin,dmax)
    return np.clip(v,vmin,vmax),np.clip(d,dmin,dmax)
def scattered(gs,k,rng,forbidden,mind):
    cand=[i for i in range(1,gs*gs+1) if i not in forbidden]
    if not cand or k<=0: return []
    chosen=[]; rc=[]
    for _ in range(100000):
        if len(chosen)>=k: break
        node=int(rng.choice(cand)); r=_rc(node,gs)
        if all(_man(r,x)>=mind for x in rc): chosen.append(node); rc.append(r)
    return sorted(chosen)
def road_pairs(gs,blocked,rng,k):
    cand=[]
    for r in range(gs):
        for c in range(gs):
            i=r*gs+c+1
            if i in blocked: continue
            if c<gs-1:
                j=r*gs+c+2
                if j not in blocked: cand.append((i,j))
            if r<gs-1:
                j=(r+1)*gs+c+1
                if j not in blocked: cand.append((i,j))
    if not cand: return []
    idx=rng.choice(len(cand),size=min(k,len(cand)),replace=False)
    return [cand[i] for i in idx]

def read_cfg():
    df=pd.read_excel(SCEN,sheet_name="scenario")
    raw=dict(zip(df["param"].astype(str),df["value"]))
    g=lambda k,d=0: raw.get(k,d) if not pd.isna(raw.get(k,d)) else d
    return dict(n_nodes=int(float(g("n_nodes",2500))),edge=float(g("edge",500)),
        vmin=float(g("vmin",0)),vmax=float(g("vmax",100)),dmin=float(g("dmin",0)),dmax=float(g("dmax",1)),
        amel=float(g("amel",0.7)),seed=int(float(g("seed",42))),out=str(g("out","inputs.xlsx")),
        road_pairs_k=int(float(g("road_pairs_k",0))),n_bases=int(float(g("n_bases",5))),
        fire_auto=int(float(g("fire_auto",0))),n_fire=int(float(g("n_fire",0))),
        fire_min_dist_cells=int(float(g("fire_min_dist_cells",5))),
        protected_auto=int(float(g("protected_auto",0))),n_protected=int(float(g("n_protected",0))),
        protected_min_dist_cells=int(float(g("protected_min_dist_cells",5))),
        water_auto=int(float(g("water_auto",0))),water_every_km=float(g("water_every_km",5.0)),
        fire_nodes=[],water_nodes=[],protected_nodes=[])

def build_inputs_df(cfg):
    gs=validate_square_grid(cfg["n_nodes"]); rng=np.random.default_rng(cfg["seed"])
    x,y=generate_grid_centers(gs,cfg["edge"])
    water=set(cfg["water_nodes"])
    if cfg["water_auto"]:
        step=max(1,int((cfg["water_every_km"]*1000)/cfg["edge"]))
        water=set(r*gs+c+1 for r in range(0,gs,step) for c in range(0,gs,step))
    fire=set(cfg["fire_nodes"]); prot=set(cfg["protected_nodes"])
    if cfg["fire_auto"]: fire=set(scattered(gs,cfg["n_fire"],rng,water|prot,cfg["fire_min_dist_cells"]))
    if cfg["protected_auto"]: prot=set(scattered(gs,cfg["n_protected"],rng,water|fire,cfg["protected_min_dist_cells"]))
    water-=fire; prot-=(fire|water)
    state=np.zeros(cfg["n_nodes"],dtype=int)
    for i in fire: state[i-1]=1
    for i in water: state[i-1]=2
    for i in prot: state[i-1]=3
    blocked=water|prot
    neigh=neighbors4(gs,blocked)
    rp=cfg.get("road_pairs",[])
    if not rp and cfg["road_pairs_k"]>0:
        rp=road_pairs(gs,blocked,rng,cfg["road_pairs_k"]); cfg["road_pairs"]=rp
    neigh=rm_roads(neigh,rp,cfg["n_nodes"])
    v,d=fields(gs,cfg["vmin"],cfg["vmax"],cfg["dmin"],cfg["dmax"],rng)
    vf=v.flatten(); df_=d.flatten()
    for i in blocked: vf[i-1]=0; df_[i-1]=0
    area=cfg["edge"]**2; e=cfg["edge"]
    beta=(vf/e)*df_
    su=np.zeros_like(df_); m=df_>0
    su[m]=0.036322*area*(2*df_[m]/e)
    return pd.DataFrame({"node_id":range(1,cfg["n_nodes"]+1),"x_coordinate":x,"y_coordinate":y,
        "value_at_start":np.round(vf,6),"fire_degradation_rate":np.round(df_,6),
        "fire_amelioration_rate":np.round(df_*cfg["amel"],6),"beta":np.round(beta,10),
        "su":np.round(su,6),"state":state,"neighborhood_list":neigh})

def _base_label(i):
    # A..Z, sonra AA, AB, ... (26'dan fazla üs için)
    s=""; i+=1
    while i>0:
        i,r=divmod(i-1,26); s=chr(ord('A')+r)+s
    return s
def build_bases_df(cfg,gs,rng,blocked,veh_scale=1.0,fe_per_base=3,frv_per_base=2,heli_A=2):
    # varsayılanlar (3 FE + 2 FRV / üs, A üssünde 2 helikopter) eski davranışla BİREBİR aynı.
    nb=cfg["n_bases"]; labels=[_base_label(i) for i in range(nb)]
    coords=scattered(gs,nb,rng,blocked,2)
    rows=[]
    def sc(v): return max(0,int(round(v*veh_scale)))
    for lbl,nid in zip(labels,coords):
        r,c=_rc(nid,gs)
        rows.append(dict(Base=lbl,x_coordinate=c,y_coordinate=r,
            Helicopter=sc(heli_A) if lbl=='A' else 0, **{"Fire Engine":sc(fe_per_base),"FRV":sc(frv_per_base)}))
    return pd.DataFrame(rows)

def build_params_df(cfg):
    n=int(cfg["n_nodes"]); e=float(cfg["edge"]); gs=validate_square_grid(n)
    area=e*e; wreq=36322.0; wm2=wreq/1e6; wcell=wm2*area
    road=";".join(f"{u}-{v}" for (u,v) in cfg.get("road_pairs",[]))
    rows=[("n_nodes",n,"required"),("cell_side_length",e,"m"),("region_side_length",gs*e,""),
        ("node_area",area,""),("time_limit",24,"h"),("water required (for 1km2-lt)",wreq,"given"),
        ("water_required_per_m2",wm2,"derived"),("water_required_per_cell",wcell,"derived"),
        ("road_pairs",road,""),("helicopter_capacity",166.67,""),("helicopter_speed",2000,""),
        ("fire_engine_capacity",117.65,""),("fire_engine_speed",500,""),
        ("FRV_capacity",15,""),("FRV_speed",1000,"")]
    return pd.DataFrame(rows,columns=["parameter","value","note"])

def generate(n_nodes,out,n_bases=None,veh_scale=1.0,fe_per_base=3,frv_per_base=2,heli_A=2,outdir=None):
    cfg=read_cfg(); cfg["n_nodes"]=n_nodes; cfg["out"]=out
    if n_bases is not None: cfg["n_bases"]=n_bases
    cfg["road_pairs"]=[]
    gs=validate_square_grid(n_nodes); rng=np.random.default_rng(cfg["seed"])
    idf=build_inputs_df(cfg)
    blocked=set(idf.loc[idf["state"].isin([2,3]),"node_id"])
    bdf=build_bases_df(cfg,gs,rng,blocked,veh_scale,fe_per_base,frv_per_base,heli_A)
    pdf=build_params_df(cfg)
    od=outdir or OUTDIR
    os.makedirs(od,exist_ok=True)
    path=os.path.join(od,out)
    with pd.ExcelWriter(path,engine="openpyxl") as w:
        idf.to_excel(w,sheet_name="inputs_df",index=False)
        bdf.to_excel(w,sheet_name="bases",index=False)
        pdf.to_excel(w,sheet_name="parameters",index=False)
    return path,idf,bdf

if __name__=="__main__":
    for gs,n in [(10,100),(11,121),(12,144)]:
        path,idf,bdf=generate(n,f"inputs_{gs}x{gs}.xlsx")
        nf=int((idf["state"].isin([0,1])).sum()); na=int((idf["state"]==1).sum())
        veh=int(bdf[["Helicopter","Fire Engine","FRV"]].sum().sum())
        print(f"{os.path.basename(path)}: n_nodes={n} |Nf|={nf} roots={na} bases={len(bdf)} |K|={veh}")
