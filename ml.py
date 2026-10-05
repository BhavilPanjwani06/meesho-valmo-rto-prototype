"""Synthetic data generation + model training (seeded, deterministic). ALL DATA IS SYNTHETIC.
Generation: latent-logit RTO process from decision-time features (no probability leakage), stage-duration
process for ETA, decay process for recovery. sample_orders.csv is schema reference only."""
import numpy as np, pandas as pd, pickle, json, time, os
from sklearn.ensemble import HistGradientBoostingClassifier as HC, HistGradientBoostingRegressor as HR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.inspection import permutation_importance
D = os.path.dirname(os.path.abspath(__file__))
CATS = {"Footwear":.55,"Ethnic wear":.4,"Baby products":.35,"Accessories":.1,"Kids":.1,"Home":0,"Beauty":.1,"Books":-.7}
SEGS = ["Metro","Tier-2","Semi-rural/Village"]
CITIES = {"Mumbai":(0,0),"Delhi":(0,1),"Jaipur":(1,1),"Lucknow":(1,1),"Nashik":(1,0),"Patna":(2,1),"Ahmedabad":(1,0),"Bengaluru":(0,0),"Chennai":(0,0),"Kolhapur":(2,0)}  # (segment idx, north flag)
BASE = np.array([4,2,6,20,6,8,4,3,3.])   # nominal hours step i -> i+1 (ASSUMPTION), 9 steps
RTO_F = ["is_cod","log_value","distance_km","addr_conf","contact_rate","window_success","rider_fit","partner_rel","delay_min","prior_fails","cat_risk","north","stage"]
ETA_F = ["stage","lane_km","distance_km","partner_rel","rider_fit","delay_min","hour"]
REC_F = ["value","condition","hours_since","cons_demand","ret_demand"]
def nominal(lane, dist):
    f = np.ones((len(lane),9)); f[:,3]=lane/600+.3; f[:,5]=lane/1200+.6; f[:,7]=1+dist/15; f[:,8]=1+dist/25
    return BASE*f
def gen(n=20000, seed=7):
    r = np.random.default_rng(seed); city = r.choice(list(CITIES),n); seg = np.array([CITIES[c][0] for c in city]); north = np.array([CITIES[c][1] for c in city])
    seg = np.clip(seg + (r.random(n)<.25),0,2); cat = r.choice(list(CATS),n)
    dist = np.clip(r.gamma(2.2,1.6+seg*1.6),.5,30); lane = r.uniform(80,1800,n)
    addr = np.clip(r.beta(9-seg*2,2+seg*.8),.3,1); contact = np.clip(r.beta(7,1.6,n)-.03*seg,.3,1)
    win = r.beta(5,2.5,n); rider = r.beta(6,2,n); part = r.beta(7,2.5,n); val = np.clip(r.lognormal(5.6,.6,n),99,3000)
    cod = (r.random(n) < .74+.06*north+.03*seg).astype(int); prior = r.poisson(.18,n)
    cr = np.array([CATS[c] for c in cat])
    nom = nominal(lane,dist); act = nom*(1.55-part[:,None])*r.lognormal(0,.12,(n,9)); startH = r.integers(6,22,n)
    delay_tot = ((act-nom)[:,:8].sum(1))*60
    def logit(i): return i+1.45*cod+.045*dist+2.3*(1-addr)+1.5*(1-contact)+1.6*(1-win)+1.4*(1-rider)+1.3*(1-part)+.005*np.clip(delay_tot,0,480)+.45*prior+cr+.3*north*cod-.0003*val
    lo,hi=-9,2
    for _ in range(30):
        m=(lo+hi)/2; (lo,hi)=(m,hi) if (1/(1+np.exp(-logit(m)))).mean()<.17 else (lo,m)
    p = 1/(1+np.exp(-logit(m))); rto = (r.random(n)<p).astype(int)
    cond = np.clip(r.beta(8,2,n),.3,1); cd = r.poisson(4+(seg==0)*6,n); rd = r.poisson(2+(seg==0)*3,n)
    o = pd.DataFrame(dict(order_id=[f"H{i:06d}" for i in range(n)],city=city,segment=[SEGS[s] for s in seg],category=cat,value=val.round(0),is_cod=cod,distance_km=dist.round(1),lane_km=lane.round(0),
        addr_conf=addr.round(3),contact_rate=contact.round(3),window_success=win.round(3),rider_fit=rider.round(3),partner_rel=part.round(3),prior_fails=prior,cat_risk=cr,north=north,delay_min_total=delay_tot.round(0),
        condition=cond.round(2),cons_demand=cd,ret_demand=rd,rto=rto))
    cn, ca = nom.cumsum(1), act.cumsum(1)
    ev = pd.DataFrame(dict(order_id=np.repeat(o.order_id.values,9),stage=np.tile(np.arange(1,10),n),planned_h=cn.ravel().round(2),actual_h=ca.ravel().round(2)))
    ev["delay_min"]=((ev.actual_h-ev.planned_h)*60).round(0)
    ff = r.random(n) < np.where(rto==1,1,p*.9)
    tag = np.where(rto==0, np.where(ff,"SAVE","PREVENT"), np.where(r.random(n)<np.clip(cd/25,0,.5),"RECOVER","CHEAPER"))
    out = pd.DataFrame(dict(order_id=o.order_id,outcome=tag,rto=rto,total_hours=ca[:,-1].round(2)))
    return o, ev, out, nom, act, startH
def train():
    o, ev, out, nom, act, startH = gen(); r=np.random.default_rng(11); n=len(o)
    o.to_csv(f"{D}/data/orders.csv",index=False); ev.to_csv(f"{D}/data/parcel_events.csv",index=False); out.to_csv(f"{D}/data/outcomes.csv",index=False)
    cumall = np.concatenate([np.zeros((n,1)),(act-nom).cumsum(1)],1)*60; rows=[]
    for _ in range(3):   # decision points at random stages; delay = accumulated so far (known at that time)
        s = r.integers(0,9,n); d = o[["is_cod","distance_km","addr_conf","contact_rate","window_success","rider_fit","partner_rel","prior_fails","cat_risk","north"]].copy()
        d["log_value"]=np.log(o.value); d["delay_min"]=cumall[np.arange(n),s]; d["stage"]=s; d["y"]=o.rto; rows.append(d)
    R = pd.concat(rows); Xtr,Xte,ytr,yte = train_test_split(R[RTO_F],R.y,test_size=.2,random_state=1)
    m1 = HC(max_iter=160,learning_rate=.07,random_state=1).fit(Xtr,ytr); auc = roc_auc_score(yte,m1.predict_proba(Xte)[:,1])
    imp = permutation_importance(m1,Xte.iloc[:4000],yte.iloc[:4000],n_repeats=2,random_state=1,scoring="roc_auc").importances_mean
    E=[pd.DataFrame(dict(stage=s,lane_km=o.lane_km,distance_km=o.distance_km,partner_rel=o.partner_rel,rider_fit=o.rider_fit,delay_min=cumall[:,s],hour=(startH+act[:,:s].sum(1))%24,y=act[:,s:].sum(1))) for s in range(9)]
    E=pd.concat(E); Xa,Xb,ya,yb = train_test_split(E[ETA_F],E.y,test_size=.2,random_state=2)
    m2 = HR(max_iter=160,random_state=2).fit(Xa,ya); mae = mean_absolute_error(yb,m2.predict(Xb))
    t = r.uniform(0,48,n); frac = np.exp(-t/34)*(.45+.5*np.clip((o.cons_demand+2*o.ret_demand)/30,0,1))*r.uniform(.9,1.05,n)
    Rc = pd.DataFrame(dict(value=o.value,condition=o.condition,hours_since=t,cons_demand=o.cons_demand,ret_demand=o.ret_demand)); yv = o.value*o.condition*frac
    Xc,Xd,yc,yd = train_test_split(Rc,yv,test_size=.2,random_state=3); m3=HR(max_iter=120,random_state=3).fit(Xc,yc); mae3=mean_absolute_error(yd,m3.predict(Xd))
    pickle.dump((m1,m2,m3),open(f"{D}/models/models.pkl","wb"))
    meta=dict(version="v1-synthetic",trained=time.strftime("%Y-%m-%d %H:%M:%S"),n_orders=n,n_rto_rows=len(R),n_eta_rows=len(E),model="sklearn HistGradientBoosting (clf: RTO; reg: ETA remaining hours; reg: recovery value)",
        rto_auc_holdout=round(auc,3),eta_mae_hours=round(mae,2),recovery_mae_inr=round(mae3,1),synthetic_rto_rate=round(float(o.rto.mean()),3),rto_features=RTO_F,
        importance={f:round(float(v),4) for f,v in sorted(zip(RTO_F,imp),key=lambda x:-x[1])},disclaimer="Trained on SYNTHETIC data generated by ml.py. Metrics measure recovery of the synthetic process, NOT production accuracy.")
    json.dump(meta,open(f"{D}/models/meta.json","w"),indent=1); return meta
if __name__=="__main__": print(json.dumps(train(),indent=1))
