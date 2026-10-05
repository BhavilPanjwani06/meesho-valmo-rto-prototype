"""DICE parcel engine + local HTTP API (stdlib). One parcel -> one journey -> continuously updated decision.
Prototype / synthetic. Evidence tags: CASE-PACK, ASSUMED, SYNTHETIC, PROPOSED."""
import json, pickle, os, sys, math, re, threading, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from ml import *
M1, M2, M3 = pickle.load(open(f"{D}/models/models.pkl","rb")); META = json.load(open(f"{D}/models/meta.json"))
ST = ["Order placed","First-mile hub","First-mile carting","Source sort","National linehaul","Destination sort","Regional linehaul","LMDC","Out for delivery","Delivered"]
SUNK = [0,4,6,11,19,24,29,50,50,50]; NODE = [0,4,2,5,8,5,5,21,0,0]   # CASE-PACK node costs
FWD, REV, PLAN_R = 50, 120, .15   # CASE-PACK; PLAN_R = planned-reverse saving fraction (ASSUMED)
GATES = ["","Reachable & Locatable","Assigned & On-plan","Attempt-ready","Handover-able","Changed-condition","Recoverable","Reverse-planned"]
STATE_BY_STAGE = lambda s: "COMMITTED" if s==0 else "IN-PIPELINE" if s<7 else "STAGED" if s==7 else "ATTEMPTING"
# id: (name, module, cost INR, success p, feature effects, evidence)  costs/p are ASSUMED prototype placeholders
ACT = {"contact_repair":("Call customer: confirm address and phone","M1",3,.8,dict(addr_min=.9,contact_min=.85)),
 "geo_pin":("Get an exact map pin (DIGIPIN) on WhatsApp","M1",1.5,.75,dict(addr_min=.9)),
 "phone_verify":("Verify the phone (WhatsApp, then IVR)","M1",1,.85,dict(contact_min=.88)),
 "window_confirm":("Confirm delivery window","M1",1,.9,dict(window_add=.12,contact_add=.06)),
 "cod_nudge":("Offer prepaid with a small incentive","EXT",4,.2,dict(cod=0)),
 "hubfit":("Switch to a faster lane","M3",4,.75,dict(rel_min=.9,gain_h=4)),
 "common_cause":("Move to the next delivery wave","M4",2,.85,dict(gain_h=2)),
 "reroute_hub":("Move to the correct hub","M3",5,.8,dict(gain_h=3,rider_min=.8)),
 "zonefit":("Give to a rider who knows the area","M5",2,.8,dict(rider_min=.9)),
 "ldp":("Hand to a local delivery partner","M5",6,.75,dict(rider_min=.85,addr_add=.1)),
 "retry_window":("Retry in preferred window","M8",3,None,{}),"neighbour":("Neighbour Fallback (OTP)","M6",1,.35,{}),
 "payment_rescue":("Payment Rescue (UPI / scam-proof)","M6",1.5,.45,{}),"pickup_node":("Trusted-node pickup / Pay-at-Counter","M10",4,.4,{}),
 "refusal_call":("Call to save a refused order","EXT",3,.2,{}),"forward_stock":("Forward to a hub with demand for this item","EXT",10,.5,{}),
 "local_recovery":("Local Recovery (sell nearby)","M9",30,None,{}),"planned_reverse":("Planned Reverse","M11",2,1.0,{})}
EXT={"geo_pin","phone_verify","cod_nudge","reroute_hub","refusal_call","forward_stock"}   # research-derived extensions: evidence PROPOSED (market-informed); all p / cost values ASSUMED
DOES={"contact_repair":"Hub agent calls the customer in their language and confirms the full address and a working number.",
 "geo_pin":"WhatsApp asks the customer to drop a map pin; the 10-character DIGIPIN is saved so the rider can navigate to the door.",
 "phone_verify":"WhatsApp confirmation first, IVR call if unanswered, so a working number is confirmed before dispatch.",
 "window_confirm":"Customer picks a delivery window in which they will be home.",
 "cod_nudge":"WhatsApp or IVR offers a small prepaid discount with a payment link. Only helps if the customer agrees.",
 "hubfit":"Parcel moves to an express linehaul slot with a more reliable partner, recovering about 4 hours.",
 "common_cause":"Parcel and customer move to the next delivery wave, adding about 2 hours of slack. Customer confirms the new slot.",
 "reroute_hub":"Parcel goes from the wrong hub to the correct destination hub on the next transfer, recovering about 3 hours.",
 "zonefit":"Parcel is handed to a rider who has delivered in this pincode before.",
 "ldp":"A local delivery partner who knows the lanes takes the last mile.",
 "retry_window":"Reattempt in the customer's preferred window after they confirm they will be home.",
 "neighbour":"Hand over to a neighbour after the customer shares an OTP.",
 "payment_rescue":"Customer pays by UPI through a scam-proof link instead of cash.",
 "pickup_node":"Customer collects from a trusted nearby shop and pays at the counter.",
"forward_stock":"Instead of returning it to the seller, send the parcel to another hub that has open orders for the same item, so the stock is sold there.",
 "refusal_call":"Call to learn why the customer refused, then offer an exchange or an open-box discount.",
 "local_recovery":"Offer the parcel to nearby buyers or a local retailer instead of sending it back.",
 "planned_reverse":"Send it back in a consolidated batch instead of an individual return."}
FAILWHY={"contact_repair":"Customer did not pick up","geo_pin":"Customer did not share a pin","phone_verify":"Number could not be verified","window_confirm":"Customer did not confirm a window","cod_nudge":"Customer preferred cash on delivery",
 "hubfit":"Express lane slot was taken","common_cause":"Customer declined the later wave","reroute_hub":"Transfer vehicle was missed","zonefit":"No familiar rider was free","ldp":"Local partner declined the area",
 "retry_window":"Customer still unreachable","neighbour":"Neighbour was not available","payment_rescue":"Customer had no UPI or cash","pickup_node":"Customer would not collect","refusal_call":"Customer still refuses","local_recovery":"No nearby buyer accepted","forward_stock":"The other hub filled its orders from other stock"}   # SIMULATED failure reasons
CAP={"forward_stock":4,"common_cause":6,"hubfit":4,"reroute_hub":4,"zonefit":6,"ldp":4,"pickup_node":6}; USED={}   # ASSUMED seats per hub; one seat frees per simulation step
REASONS = {"unavailable":"Customer unavailable","address":"Address / locatability","refused":"Customer refused","cod_cash":"COD cash unavailable","weak":"Weak attempt (integrity flag)","damaged":"Parcel damaged"}
OPTS = {"unavailable":["retry_window","neighbour","pickup_node","local_recovery","forward_stock","planned_reverse"],"address":["geo_pin","ldp","retry_window","local_recovery","forward_stock","planned_reverse"],"refused":["refusal_call","local_recovery","forward_stock","planned_reverse"],
 "cod_cash":["payment_rescue","pickup_node","local_recovery","forward_stock","planned_reverse"],"weak":["retry_window","local_recovery","planned_reverse"],"damaged":["planned_reverse"]}
P = {}; SEQ = [0]; T0 = 10.0
clamp = lambda x,a=0,b=1: max(a,min(b,x))
def clock(h): d=int((T0+h)//24); t=(T0+h)%24; hh=int(t); mm=int((t-hh)*60); return f"{(hh%12) or 12}:{mm:02d} {'AM' if hh<24 and hh<12 else 'PM'}"+(f" (+{d}d)" if d else "")
def rto_x(f, stage=None): return [f["is_cod"],math.log(f["value"]),f["distance_km"],f["addr_conf"],f["contact_rate"],f["window_success"],f["rider_fit"],f["partner_rel"],f["delay_min"],f["prior_fails"],f["cat_risk"],f["north"],(f["stage"] if stage is None else stage)]   # order == ml.RTO_F
def rto_p(f, stage=None): return float(M1.predict_proba(np.array([rto_x(f,stage)],dtype=float))[0,1])
def rto_pm(fs): return [float(v) for v in M1.predict_proba(np.array([rto_x(f) for f in fs],dtype=float))[:,1]]   # batched
def eta_f(p,f): return float(max(0,M2.predict(np.array([[p["stage"],f["lane_km"],f["distance_km"],f["partner_rel"],f["rider_fit"],max(0,f["delay_min"]),(T0+p["now"])%24]],dtype=float))[0]))
def eta_rem(p): return eta_f(p,p["f"])
def dk(h):
    """Time-decay of rescue success after a failed attempt (ASSUMED curve; market sources say acting inside hours recovers far more than acting after a day)."""
    if h<=4: return 1.0
    if h<=12: return 1-.12*(h-4)/8
    if h<=24: return .88-.16*(h-12)/12
    return max(.35,.72-.32*(h-24)/24)

import zlib, random
GEO={"Mumbai":(19.076,72.878),"Delhi":(28.614,77.209),"Jaipur":(26.912,75.787),"Lucknow":(26.847,80.947),"Nashik":(19.998,73.790),"Patna":(25.594,85.138),"Ahmedabad":(23.023,72.572),"Bengaluru":(12.972,77.594),"Chennai":(13.083,80.271),"Kolhapur":(16.705,74.243)}
def km(a,b):
    """Road distance estimate between two hub cities: great-circle x 1.25. SYNTHETIC geography; real city positions, invented hubs and partners."""
    if a==b: return 0
    la1,lo1=map(math.radians,GEO[a]); la2,lo2=map(math.radians,GEO[b]); h=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return int(round(2*6371*math.asin(math.sqrt(h))*1.25))
def rg(p,a=""): return random.Random(zlib.crc32((p["id"]+a).encode()))
PARTNERS=["Northline Logistics","Sarathi Freight","BlueRoute Express","Kaveri Carriers","Trishul Cargo","Ganga Haulage","SwiftAxle Movers","Mitra Roadways"]
RIDERS=["Imran K.","Suresh P.","Anita D.","Rahul M.","Farhan S.","Deepa N.","Vikas T.","Meena R.","Arjun B.","Salim H."]
SHOPS=["Shree Ganesh Traders","Anand Fashion Point","Maa Durga Stores","New Style Emporium","Royal Collection","Sai Krupa Retail","Balaji Bazaar","Gupta Family Store","Jyoti Gift Corner","Kavya Kids World"]
NODES=["Kirana store","Mobile recharge shop","Pharmacy","Stationery shop","Tailoring shop","Sweet shop"]
LOCS=["Ward 12","Old Market Road","Station Road","Green Park","Civil Lines","Lake View Colony","Model Town","Shivaji Nagar","Gandhi Chowk","Rajiv Nagar"]
def origin_city(p):
    c=p["city"]; l=p["f"]["lane_km"]; return min((x for x in GEO if x!=c),key=lambda x:abs(km(x,c)-l))
def via_city(o,c):
    return min((x for x in GEO if x not in(o,c)),key=lambda x:km(o,x)+km(x,c))
def demand_at(sku,city,day=0):
    """Open orders for this item waiting at another hub. SYNTHETIC, deterministic per item, hub and day."""
    return random.Random(zlib.crc32(f"{sku}|{city}|{day}".encode())).randint(0,13)
def sku_of(p): return f"{p['product']} / {p['category']}"
def redir(p):
    """Hubs (other than the parcel's own) with open orders for the same item, ranked by net saving. Forward cost = 8 + 0.045/km (ASSUMED); handling 6 h + 45 km/h."""
    sku=sku_of(p); out=[]
    for c in GEO:
        if c==p["city"]: continue
        n=demand_at(sku,c,int(p["now"]//24)); d=km(p["city"],c); cost=round(8+.045*d,1); hrs=round(6+d/45,1)
        pr=clamp(.15+.04*n,.15,.7); out.append(dict(city=c,hub="LMDC "+c,km=d,orders=n,cost=cost,hrs=hrs,p=round(pr,2),net=round(pr*(REV-cost),1)))
    return sorted([o for o in out if o["orders"]>=3],key=lambda o:-o["net"])
WAVES=[("Morning wave","9 AM-1 PM"),("Afternoon wave","1-5 PM"),("Evening wave","5-9 PM")]
def wave_i(h): t=(T0+h)%24; return 0 if t<13 else 1 if t<17 else 2
def wave_txt(h,nxt=False):
    i=wave_i(h); d=int((T0+h)//24)
    if nxt: d+=1 if i==2 else 0; i=(i+1)%3
    return f"{WAVES[i][0]} ({WAVES[i][1]})"+(f", day +{d}" if d else "")
def alt_city(c): cs=list(CITIES); return cs[(cs.index(c)+1)%len(cs)]
FEED=[]; SIM=dict(tick=0,auto="routine",budget=45,mode="sample",spawn=4)   # auto: off | routine (cheap pre-attempt gate fixes) | full   # budget = rescue minutes per hub per shift (ASSUMED)
MINS={"contact_repair":3,"window_confirm":1,"hubfit":10,"common_cause":5,"zonefit":4,"ldp":25,"retry_window":15,"neighbour":10,"payment_rescue":8,"pickup_node":12,"local_recovery":20,"planned_reverse":2,"geo_pin":3,"phone_verify":2,"cod_nudge":3,"reroute_hub":8,"refusal_call":6,"forward_stock":6}   # ASSUMED rescue-capacity minutes per action
LOCK=threading.RLock()
def ev(p, txt, kind="info"):
    e=dict(t=clock(p["now"]),h=round(p["now"],2),txt=txt,kind=kind); p["events"].append(e); FEED.append(dict(e,id=p["id"],city=p["city"],tick=SIM["tick"]))
    if len(FEED)>400: del FEED[:150]
def log(p, txt): p["log"].append(dict(t=clock(p["now"]),txt=txt))
def new_order(a, seed=None):
    SEQ[0]+=1; oid=f"M{100230+SEQ[0]}"; r=np.random.default_rng(seed if seed is not None else SEQ[0]*97+5)
    seg,north = CITIES[a["city"]]; dist=float(np.clip(r.gamma(2.2,1.6+seg*1.6),.5,30)); lane=float(r.uniform(80,1800))
    addr={"clear":.93,"landmark":.78,"incomplete":.6}[a.get("address","clear")]+r.normal(0,.015)
    f=dict(is_cod=int(a["payment"]=="COD"),value=float(a["value"]),distance_km=round(dist,1),lane_km=round(lane),addr_conf=clamp(addr,.3,.99),contact_rate=clamp(r.normal(.9,.04),.3,.99),
      window_success={"7-9 PM":.85,"11 AM-2 PM":.65,"Any time":.5}.get(a["window"],.7),rider_fit=clamp(r.normal(.82,.05),.3,.97),partner_rel=clamp(r.normal(.8,.06),.3,.98),delay_min=0.,prior_fails=0,
      cat_risk=CATS[a["category"]],north=north,stage=0)
    for k in ("addr_conf","contact_rate","window_success","rider_fit","partner_rel"): f[k]=a.get("ov",{}).get(k,f[k])
    nom=nominal(np.array([f["lane_km"]]),np.array([dist]))[0]
    p=dict(id=oid,product=a["product"],city=a["city"],segment=SEGS[seg],category=a["category"],payment=a["payment"],value=a["value"],window=a["window"],f=f,stage=0,state="COMMITTED",now=0.,nom=nom.tolist(),
      dur=(nom*(1.55-f["partner_rel"])*r.lognormal(0,.15,9)).tolist(),events=[],log=[],spent=0.,rev=0.,outcome=None,reason=None,used=[],fails=0,rng=int(r.integers(1e9)),arr={},risk_hist={},cond=float(clamp(r.normal(.85,.07),.4,1)),
      cd=int(r.poisson(4+(seg==0)*6)),rd=int(r.poisson(2+(seg==0)*3)),fail_h=None,plan_arr=[float(x) for x in nom.cumsum()],learn=[],recover_via=None,hold=0,decisions=[],first_act_h=None,late_cause=None,late_at=None,wrong_hub=None)
    f0=rto_p(f); p["risk0"]=f0; rem=eta_rem(p); p["promised"]=rem+max(1.5,.06*rem); p["arr"][0]=0.; p["risk_hist"][0]=f0
    ev(p,f"Order placed · ₹{a['value']} · {a['payment']}","ok"); log(p,f"Wave Sync set: promise {clock(p['promised'])} (forecast {clock(rem)} + buffer). Initial RTO risk {f0:.0%}.")
    P[oid]=p; return p
def slack(p): return p["promised"]+p.get("rec_h",0)-(p["now"]+eta_rem(p))
def gate(p):
    f=p["f"]; s=p["stage"]
    if p["outcome"] or p["state"] in("REVERSING","RECOVERING"): return (7 if p["state"]=="REVERSING" else 6),True,"Terminal / post-failure path active"
    if p["state"]=="HELD-OPEN": return 5,False,f"Failure: {REASONS[p['reason']]}. Change-or-Rescue decides next state."
    if s==0:
        ok = f["addr_conf"]>=.85 and f["contact_rate"]>=.8
        return 1,ok,"Address confidence %d%% (need ≥85%%), contactability %d%% (need ≥80%%)"%(f["addr_conf"]*100,f["contact_rate"]*100)
    if s<7: sl=slack(p); return 2,sl>=0,"Plan slack %+d min (need ≥0 vs promised wave)"%(sl*60)
    if s==7: return 3,f["rider_fit"]>=.75,"Rider-zone fit %d%% (need ≥75%%)"%(f["rider_fit"]*100)
    return 4,True,"Handover attempt in progress"
def contributors(p):
    f=p["f"]; refs=dict(Address=dict(addr_conf=.97),Customer=dict(window_success=.9,contact_rate=.95),COD=dict(is_cod=0),Rider_fit=dict(rider_fit=.92),Network=dict(partner_rel=.92,delay_min=0),Distance=dict(distance_km=2.))
    ps=rto_pm([f]+[{**f,**v} for v in refs.values()]); base=ps[0]; out={k:round((base-x)*100,1) for k,x in zip(refs,ps[1:])}
    return base,{k:v for k,v in sorted(out.items(),key=lambda x:-x[1]) if v>.05}
def reason_weights(p):
    f=p["f"]; w=dict(unavailable=(1-f["window_success"])+(1-f["contact_rate"]),address=(1-f["addr_conf"])*1.5,weak=(1-f["rider_fit"]),refused=.25*f["is_cod"],cod_cash=.3*f["is_cod"]); return w
def recovery(p):
    t=(p["now"]-(p["fail_h"] or p["now"])); hs=(0,6,12,24)
    X=np.array([[p["value"],p["cond"],t+h,p["cd"],p["rd"]] for h in hs]+[[p["value"],p["cond"],0,p["cd"],p["rd"]]],dtype=float); y=M3.predict(X); dec=[(h,float(v)) for h,v in zip(hs,y[:4])]; v0=float(y[4])
    opts=[dict(id="consumer",name="Nearby consumer (hyperlocal offer)",p=clamp(p["cd"]/18,0,.75),hrs=6,cost=25),dict(id="retailer",name="Local retailer buys / consigns",p=clamp(p["rd"]/8,0,.7),hrs=10,cost=30),
      dict(id="node",name="Trusted node / pickup point",p=.4,hrs=18,cost=15),dict(id="reverse",name="Planned reverse",p=1.,hrs=48,cost=REV*(1-PLAN_R))]
    for o in opts: o["ev"]=round(o["p"]*(REV-o["cost"]) if o["id"]!="reverse" else REV*PLAN_R-2,1)
    return dict(now_value=round(dec[0][1]),decay=[dict(h=h,v=round(v)) for h,v in dec],options=opts,best=max(opts,key=lambda o:o["ev"])["id"],v0=round(v0))
def conf_of(sl,rem): return clamp(.5*(1+math.erf((sl/(.25*rem+.6))/1.414)))
HEAD={"missed_connection":"Missed the line-haul connection","weather":"Weather or traffic is holding this lane","wrong_hub":"Parcel went to the wrong hub","slow_leg":"Running slow in the network",
 "addr_incomplete":"Address is incomplete, the rider cannot find the door","addr_landmark":"Address depends on a landmark, the rider may not find it","phone_weak":"Customer phone is rarely answered, delivery cannot be confirmed",
 "addr_drift":"Address doubt found in transit, confirm before the rider goes out","phone_drift":"Customer phone looks unreliable, verify before out-for-delivery","rider_mismatch":"Assigned rider does not know this area",
 "cod_risky":"High-risk cash order, likely to be refused","thin_slack":"Little time to spare, one more delay breaks the promise","window_weak":"Customer may not be home in the chosen slot",
 "fail_unavailable":"Customer was not home, needs a new time","fail_address":"Rider could not find the address, needs a map pin","fail_refused":"Customer refused the parcel at the door",
 "fail_cod_cash":"Customer did not have the cash ready","fail_weak":"Attempt looks weak, verify before retrying","fail_damaged":"Parcel arrived damaged, cannot be delivered"}
CLABEL={"missed_connection":"Missed connection","weather":"Weather or traffic","wrong_hub":"Wrong hub","slow_leg":"Slow leg","addr_incomplete":"Address incomplete","addr_landmark":"Landmark-only address","phone_weak":"Phone unreachable",
 "addr_drift":"Address doubt","phone_drift":"Phone doubt","rider_mismatch":"Rider mismatch","cod_risky":"High-risk cash order","thin_slack":"Thin slack","window_weak":"Slot mismatch"}
def diagnose(p,base=None):
    """Why does this parcel need attention? Gate-blocking causes first, then optional ones. Thresholds PROPOSED."""
    if p["outcome"] or p["state"] in("REVERSING","RECOVERING","RECOVERED","DELIVERED"): return []
    f=p["f"]; s=p["stage"]; out=[]
    def add(i,label,head,evid,block,sev): out.append(dict(id=i,label=label,headline=head,evidence=evid,block=bool(block),sev=sev))
    if p["state"]=="HELD-OPEN":
        r=p["reason"]; add("fail_"+r,REASONS[r],HEAD["fail_"+r],"Failed at the door. %d h of the 48 h rescue window left"%max(0,round(48-(p["now"]-p["fail_h"]))),True,3); return out
    g,ok,_=gate(p)
    if s==0 and not ok:
        if f["addr_conf"]<.85: add("addr_incomplete" if f["addr_conf"]<.7 else "addr_landmark",CLABEL["addr_incomplete" if f["addr_conf"]<.7 else "addr_landmark"],HEAD["addr_incomplete" if f["addr_conf"]<.7 else "addr_landmark"],"Address confidence %d%%, need 85%%"%round(f["addr_conf"]*100),True,2)
        if f["contact_rate"]<.8: add("phone_weak",CLABEL["phone_weak"],HEAD["phone_weak"],"Contactability %d%%, need 80%%"%round(f["contact_rate"]*100),True,2)
    elif 0<s<7 and not ok:
        sl=slack(p); c=p.get("late_cause") or "slow_leg"; m=round(-sl*60); rem=eta_rem(p)
        add(c,CLABEL[c],HEAD[c]+(" (in %s)"%p["late_at"] if c=="slow_leg" and p.get("late_at") else "")+", %d min behind the promise"%m,"Arrives %s, promised %s"%(clock(p["now"]+rem-p.get("rec_h",0)),clock(p["promised"])),True,2)
    elif s==7 and not ok: add("rider_mismatch",CLABEL["rider_mismatch"],HEAD["rider_mismatch"],"Rider-zone fit %d%%, need 75%%"%round(f["rider_fit"]*100),True,2)
    if 0<s<7 and ok:
        sl_=slack(p)
        if sl_<1: add("thin_slack",CLABEL["thin_slack"],HEAD["thin_slack"]+" (%d min left)"%round(sl_*60),"Slack %+d min against the promised wave"%round(sl_*60),False,1)
    if s<=7:
        if s>0 and f["addr_conf"]<.85: add("addr_drift",CLABEL["addr_drift"],HEAD["addr_drift"],"Address confidence %d%%"%round(f["addr_conf"]*100),False,1)
        if s>0 and f["contact_rate"]<.8: add("phone_drift",CLABEL["phone_drift"],HEAD["phone_drift"],"Contactability %d%%"%round(f["contact_rate"]*100),False,1)
        if f["is_cod"]:
            b=rto_p(f) if base is None else base
            if b>=.2: add("cod_risky",CLABEL["cod_risky"],HEAD["cod_risky"],"Cash order, %d%% chance of return"%round(b*100),False,1)
        if f["window_success"]<.6: add("window_weak",CLABEL["window_weak"],HEAD["window_weak"],"Slot success %d%%"%round(f["window_success"]*100),False,1)
    return out
def fx_project(f,a):
    """What an action changes, without mutating: returns (new features, hours of slack recovered)."""
    fx=ACT[a][4]; nf=dict(f)
    if "addr_min" in fx: nf["addr_conf"]=max(nf["addr_conf"],fx["addr_min"])
    if "addr_add" in fx: nf["addr_conf"]=clamp(nf["addr_conf"]+fx["addr_add"])
    if "contact_min" in fx: nf["contact_rate"]=max(nf["contact_rate"],fx["contact_min"])
    if "contact_add" in fx: nf["contact_rate"]=clamp(nf["contact_rate"]+fx["contact_add"])
    if "window_add" in fx: nf["window_success"]=clamp(nf["window_success"]+fx["window_add"])
    if "cod" in fx: nf["is_cod"]=fx["cod"]
    if "rel_min" in fx: nf["partner_rel"]=max(nf["partner_rel"],fx["rel_min"])
    if "rider_min" in fx: nf["rider_fit"]=max(nf["rider_fit"],fx["rider_min"])
    g=fx.get("gain_h",0)
    if g: nf["delay_min"]=max(0,nf["delay_min"]-g*60)
    return nf,g
def seats(p,a):
    c=CAP.get(a); return None if c is None else max(0,c-USED.get((p["city"],a),0))

def details(p,a,x=None):
    """Exact specifics of one alternative: who, where, how far, when. All names, places and slots are SYNTHETIC; chances and costs ASSUMED."""
    f=p["f"]; c=p["city"]; r=rg(p,a); pr=p["promised"]+p.get("rec_h",0); D=dict(frm=None,to=None,how="",specs=[],cands=None,cutoff=None,note=None)
    def sp(k,v): D["specs"].append([k,str(v)])
    if a=="hubfit":
        o=origin_city(p); d=km(o,c); m=via_city(o,c); p1,p2=r.sample(PARTNERS,2); rel=round(f["partner_rel"]*100)
        if km(o,m)+km(m,c)>1.35*d: m=None
        sd=(km(o,m)+km(m,c)) if m else d; sh=round(sd/35+6,1); eh=round(max(2.0,sh-4),1)
        cur=f"{o} Sort > {m} Sort > LMDC {c}" if m else f"{o} Sort > LMDC {c}"
        D.update(frm=f"Standard: {cur}, {p1} ({rel}% reliable), about {sh} h",to=f"Express: {o} Sort direct to LMDC {c}, {p2} (90% reliable), about {eh} h",how=f"about {round(sh-eh,1)} h earlier at the LMDC")
        sp("Current lane",f"{cur} ({sd} km, {p1}, reliability {rel}%)"); sp("New lane",f"{o} Sort > LMDC {c} direct ({d} km, dedicated truck, {p2}, reliability 90%)")
        sp("Next express departure",clock(p["now"]+1.5)); sp("Release by",clock(p["now"]+1.0)); sp("Expected transit",f"{eh} h express vs {sh} h standard")
        st=seats(p,a); sp("Express seats at this hub",f"{st} of {CAP[a]} left"); sp("Extra cost","Rs 4 per parcel (ASSUMED)"); D["cutoff"]=f"Release by {clock(p['now']+1.0)} to make the {clock(p['now']+1.5)} departure"
    elif a=="common_cause":
        i=wave_i(pr); j=(i+1)%3; used=26+r.randint(0,14)+USED.get((c,a),0); nxt=wave_txt(pr,True)
        D.update(frm=wave_txt(pr),to=nxt,how="about 2 h of slack added")
        sp("Current wave",f"{wave_txt(pr)}, promised {clock(p['promised'])}"); sp("New wave",nxt); sp("Slots in the new wave",f"{used} of 48 used, {48-used} left"); sp("Customer",f"Asked once on WhatsApp to accept {WAVES[j][1]}; if no reply in 30 min the parcel stays in the current wave")
        sp("Seats to re-slot at this hub",f"{seats(p,a)} of {CAP[a]} left"); sp("Cost","Rs 2 per parcel (ASSUMED)"); D["cutoff"]=f"Customer must accept before {clock(p['now']+.5)}"
    elif a=="reroute_hub":
        w=p.get("wrong_hub") or alt_city(c); d=km(w,c); hrs=round(d/45+3,1); pn=r.choice(PARTNERS)
        D.update(frm=f"LMDC {w} (wrong hub)",to=f"LMDC {c} (correct hub)",how="about 3 h recovered")
        sp("Now at",f"LMDC {w}"); sp("Send to",f"LMDC {c}, {d} km away"); sp("Transfer truck",f"{pn}, departs {clock(p['now']+2)}, transit about {hrs} h"); sp("Seats",f"{seats(p,a)} of {CAP[a]} left"); sp("Cost","Rs 5 per parcel (ASSUMED)")
        D["cutoff"]=f"Load by {clock(p['now']+1.5)}"
    elif a in("zonefit","ldp"):
        n=3 if a=="zonefit" else 2; rows=[]; fits=[.92,.88,.84]
        for k in range(n):
            nm=(r.choice(RIDERS) if a=="zonefit" else r.choice(["Gali Express","LaneRunner","Pahiya Delivery","QuickStep Riders"])); dn=r.randint(9,60) if a=="zonefit" else r.randint(40,180)
            rows.append([nm,f"{dn} in this pincode (30 days)",f"{round(r.uniform(.4,4.5),1)} km from hub",f"{round(fits[k]*100)}%"])
        D["cands"]=dict(head=["Who","Local experience","Distance","Area fit"],rows=rows,note="Top row is assigned. The others are backups.")
        D.update(frm=f"Assigned rider, area fit {round(f['rider_fit']*100)}%",to=f"{rows[0][0]}, area fit {rows[0][3]}"+(" (local partner)" if a=="ldp" else ""),how=rows[0][1])
        sp("Rider today",f"area fit {round(f['rider_fit']*100)}%, below the 75% needed"); sp("Seats",f"{seats(p,a)} of {CAP[a]} left"); sp("Cost","Rs 2 (ZoneFit) or Rs 6 (local partner) per parcel (ASSUMED)")
        if a=="ldp": sp("Extra effect","Local partner knows the lanes, so address confidence also rises about 10 points")
    elif a=="contact_repair":
        D.update(frm=f"Address {round(f['addr_conf']*100)}%, phone {round(f['contact_rate']*100)}%",to="Address 90%+, phone 85%+",how="confirmed by call")
        sp("Who calls","Hub desk agent, in the customer's language"); sp("Attempts","2 calls, 10 minutes apart, then WhatsApp"); sp("What is confirmed","House or flat number, floor, landmark, an alternate phone number"); sp("Cost","Rs 3 (ASSUMED)")
    elif a=="geo_pin":
        D.update(frm=f"Address confidence {round(f['addr_conf']*100)}%",to="Exact map pin saved, confidence 90%",how="DIGIPIN saved")
        sp("Channel","WhatsApp message in the customer's language with a Share Location button"); sp("What is saved","Map pin plus its 10-character DIGIPIN (about 4 m by 4 m), format like XXX-XXX-XXXX"); sp("Rider sees","Pin on the rider app, so no landmark hunting")
        sp("Why it helps","Address errors are a leading cause of failed delivery (vendor sources); a pin removes the guesswork"); sp("Cost","Rs 1.5 (ASSUMED)"); D["note"]="DIGIPIN is India Post's geo-coded address code, launched May 2025. Prototype only simulates the capture."
    elif a=="phone_verify":
        ans=round(f["contact_rate"]*10); D.update(frm=f"Contactability {round(f['contact_rate']*100)}%",to="Verified number, 88%",how="WhatsApp then IVR")
        sp("Recent calls answered",f"{ans} of 10 (simulated)"); sp("Step 1","WhatsApp confirm now"); sp("Step 2","IVR call after 30 min if no reply"); sp("Step 3","SMS with a reply code; the number is marked unverified if all three fail"); sp("Cost","Rs 1 (ASSUMED)")
    elif a=="window_confirm":
        wins=[("7-9 PM",.85),("11 AM-2 PM",.65),("Any time",.5)]
        D.update(frm=f"Slot '{p['window']}', success {round(f['window_success']*100)}%",to=f"Customer-picked slot, about {round(min(1,f['window_success']+.12)*100)}%",how="")
        D["cands"]=dict(head=["Slot","Typical success"],rows=[[w,f"{round(s*100)}%"]+([] if False else []) for w,s in wins],note="Customer chooses on WhatsApp. Success rates are ASSUMED.")
        sp("Cost","Rs 1 (ASSUMED)")
    elif a=="cod_nudge":
        D.update(frm="Cash on delivery",to="Prepaid (if the customer agrees)",how="small incentive")
        sp("Channel","WhatsApp with UPI link; IVR call if unread after 30 min"); sp("Offer","Small prepaid discount, budgeted inside the Rs 4 cost (ASSUMED)"); sp("Chance customer agrees","20% (ASSUMED; vendor blogs claim 15 to 30%)")
        sp("If it works","Cash risk goes away: prepaid orders return far less often (vendor sources say 3 to 5 times less)"); D["note"]="Only worth doing on high-risk cash orders. The prototype offers it only when the expected gain is positive."
    elif a=="retry_window":
        D.update(frm="Failed attempt, not delivered",to=f"Reattempt in '{p['window']}' window",how="")
        sp("Reattempt slot",f"next '{p['window']}' window, from {clock(p['now']+4)}"); sp("Confirmation","Customer confirms on WhatsApp that someone will be home"); sp("Rider","Same rider if area fit is 75% or more, else ZoneFit"); sp("Cost","Rs 3 (ASSUMED)")
    elif a=="neighbour":
        D.update(frm="At the door, nobody home",to="Neighbour receives it with the customer's OTP",how="")
        sp("Who","A neighbour nominated by the customer on WhatsApp"); sp("Proof","OTP sent to the customer, photo at the door"); sp("Not allowed","Cash parcels are not handed to neighbours"); sp("Cost","Rs 1 (ASSUMED)")
    elif a=="payment_rescue":
        D.update(frm="Customer had no cash",to=f"Pays Rs {p['value']} by UPI link",how="")
        sp("How","In-app UPI link or QR shown by the rider; link valid 2 h"); sp("Rule","Payment only inside the app or UPI, never to a personal number"); sp("Cost","Rs 1.5 (ASSUMED)")
    elif a=="pickup_node":
        rows=[[f"{r.choice(SHOPS)} ({r.choice(NODES)})",f"{round(r.uniform(.3,1.8),1)} km from the customer",f"Open till {r.choice(['8 PM','9 PM','10 PM'])}",f"{r.randint(82,97)}%"] for _ in range(3)]
        D["cands"]=dict(head=["Trusted node","Distance","Hours","Trust score"],rows=rows,note="Customer picks one. Trust score is simulated.")
        D.update(frm="Home delivery failed",to=f"Pickup at {rows[0][0]}",how=rows[0][1]); sp("Payment","Pay at the counter (cash or UPI)"); sp("Holding period","48 h, then returns"); sp("Seats",f"{seats(p,a)} of {CAP[a]} left"); sp("Cost","Rs 4 (ASSUMED)")
    elif a=="refusal_call":
        D.update(frm="Customer refused at the door",to="Customer accepts after a call or offer",how="")
        sp("Who calls","Hub desk agent, within 4 h of the refusal"); sp("Script","Ask the reason. Offer an exchange, an open-box discount, or a new delivery time"); sp("Chance","20% (ASSUMED; vendor sources say refusals recover least, about 15 to 30%)"); sp("Cost","Rs 3 (ASSUMED)")
    elif a=="local_recovery":
        rc=recovery(p); nv=rc["now_value"]; sc=1.5 if p["segment"]!="Metro" else 1.0; cons=[]; rets=[]
        for k in range(min(3,p["cd"])): cons.append([f"Buyer B-{k+1}",r.choice(LOCS),f"{round(r.uniform(.6,4.5)*sc,1)} km",f"Rs {round(nv*r.uniform(.92,1.0))}",f"viewed similar items {r.randint(2,6)} times today"])
        for k in range(min(3,p["rd"])): rets.append([r.choice(SHOPS),r.choice(LOCS),f"{round(r.uniform(1,7)*sc,1)} km",f"Rs {round(nv*r.uniform(.72,.86))}",f"buys or consigns {p['category'].lower()}"])
        cons.sort(key=lambda z:float(z[2].split()[0])); rets.sort(key=lambda z:float(z[2].split()[0]))
        rows=[["Buyer"]+z for z in []]
        D["cands"]=dict(head=["Who","Area","Distance","Would pay","Signal"],rows=[["Consumer: "+z[0]]+z[1:] for z in cons]+[["Retailer: "+z[0]]+z[1:] for z in rets],note=f"{p['cd']} nearby buyers and {p['rd']} retailers are interested in total; top matches shown, nearest first. Names and places are SYNTHETIC.")
        nearest=(cons+rets)[0] if (cons or rets) else None
        D.update(frm="Return to seller at Rs 120",to=(f"Sell to {nearest[0]}, {nearest[2]} away" if nearest else "Sell nearby (no buyer signal right now)"),how=f"value kept about Rs {nv}")
        sp("Parcel value if sold now",f"Rs {nv} (falls with time)"); sp("Handling",f"Rs 30 only if it sells"); sp("Time to sell","about 6 h to a buyer, 10 h to a retailer")
    elif a=="forward_stock":
        hs=redir(p); sku=sku_of(p)
        rows=[[o["hub"],f"{o['km']} km",f"{o['orders']} open orders",f"{o['hrs']} h",f"Rs {o['cost']:g}",f"{round(o['p']*100)}%"] for o in hs[:4]]
        D["cands"]=dict(head=["Send to","Distance","Demand for this item","Transit","Forward cost","Chance it is used"],rows=rows,note=f"Item: {sku}. Open orders per hub are SYNTHETIC. The top row is used.")
        if hs:
            b=hs[0]; D.update(frm=f"Return to seller at Rs {REV}",to=f"Forward to {b['hub']}, {b['km']} km, fills one of {b['orders']} open orders",how=f"forward Rs {b['cost']:g} instead of Rs {REV}")
            sp("Best hub",f"{b['hub']} ({b['km']} km, about {b['hrs']} h)"); sp("Open orders there",f"{b['orders']} for this item"); sp("Forward cost",f"Rs {b['cost']:g} (8 + 0.045 per km, ASSUMED)"); sp("Also avoids","A fresh shipment from the seller for that order (not counted in the numbers)")
        sp("Condition",f"Parcel condition score {round(p['cond']*100)}%, needs 60% or more"); sp("Seats",f"{seats(p,a)} of {CAP[a]} left"); D["note"]="Counts as Recover (inventory kept), not as an RTO cut. It moves stock to where it sells."
    elif a=="planned_reverse":
        o=origin_city(p); n=r.randint(6,24); D.update(frm=f"Individual return at Rs {REV}",to=f"Consolidated return batch to the seller warehouse in {o}",how="15% cheaper")
        sp("Batch",f"RB-{c[:3].upper()}-{int(p['now']//24)+1:02d}, about {n} parcels"); sp("Destination",f"Seller warehouse, {o} ({km(c,o)} km)"); sp("Truck",f"Next consolidated truck at {clock(p['now']+10)}"); sp("Cost",f"Rs {round(REV*(1-PLAN_R))} plus Rs 2 planning (saves Rs {round(REV*PLAN_R)}, ASSUMED)")
    else: return None
    return D
def where(p,a):
    D=details(p,a)
    if D and D["frm"]: return dict(frm=D["frm"],to=D["to"],how=D["how"])
    return WHERE2.get(a)
def actions_ex(p):
    """Feasible next actions, ranked. Returns (actions, unavailable). Pre-attempt EV = P(success) x dRisk x 120 - cost. Gate-blocking actions are recommended even if EV <= 0."""
    if p["outcome"] or p["state"] in("RECOVERING","REVERSING"): return [],[]
    return post_actions(p) if p["state"]=="HELD-OPEN" else pre_actions(p)
def actions(p): return actions_ex(p)[0]
def pre_actions(p):
    f=p["f"]; g,ok,_=gate(p); s=p["stage"]; block=not ok; dg={d["id"] for d in diagnose(p,-1)}; must=set()
    if block:
        if g==1:
            na=f["addr_conf"]<.85; nc=f["contact_rate"]<.8
            if na: must.add("geo_pin")
            if nc: must.add("phone_verify")
            if na or nc: must.add("contact_repair")
        elif g==2:
            if "wrong_hub" in dg: must.add("reroute_hub")
            must|={"hubfit","common_cause"}
        elif g==3: must|={"zonefit","ldp"}
    cand=sorted(must)
    if s<=7:
        for a,c in(("geo_pin",f["addr_conf"]<.85),("phone_verify",f["contact_rate"]<.8),("window_confirm",f["window_success"]<.6),("cod_nudge",bool(f["is_cod"]))):
            if c and a not in cand: cand.append(a)
    un=[]; keep=[]
    for a in cand:
        st=seats(p,a)
        if st is not None and st<=0: un.append(dict(id=a,name=ACT[a][0],why="No seats left at this hub right now (0 of %d). One frees each step."%CAP[a])); continue
        keep.append(a)
    proj=[fx_project(f,a) for a in keep]; pp=rto_pm([f]+[x[0] for x in proj]); base=pp[0]; rem=eta_rem(p); sl0=slack(p); out=[]
    for a,(nf,gh),ra in zip(keep,proj,pp[1:]):
        nm,mod,cost,pr,_=ACT[a]; dp=base-ra; evv=pr*dp*REV-cost; eta_a=eta_f(p,nf); sl_a=p["promised"]+p.get("rec_h",0)+gh-(p["now"]+eta_a); m=a in must
        clears=(nf["addr_conf"]>=.85 and nf["contact_rate"]>=.8) if (g==1 and block) else (sl_a>=0) if (g==2 and block) else (nf["rider_fit"]>=.75) if (g==3 and block) else False
        if not m and evv<=0: continue
        out.append(dict(id=a,name=nm,module=mod,cost=cost,p=pr,dp=round(dp*100,1),ev=round(evv,1),gate=g if m else 0,clears=bool(clears),must=bool(m),partial=bool(m and not clears),mins=MINS[a],
          evidence="PROPOSED (market-informed); p and cost ASSUMED" if a in EXT else "ASSUMED",gain_h=gh,risk_now=round(base,3),risk_after=round(ra,3),slack_now_min=round(sl0*60),slack_after_min=round(sl_a*60),
          eta_after_h=eta_a,rem_h=rem,seats=seats(p,a),cap=CAP.get(a)))
    out.sort(key=lambda x:(not x["must"],not x["clears"],-(x["slack_after_min"] if (x["must"] and not x["clears"]) else 0),-x["ev"]))
    return out,un
WHERE2={"retry_window":dict(frm="Failed attempt, not delivered",to="Reattempt in the customer's chosen window",how=""),"neighbour":dict(frm="At the door, nobody home",to="Neighbour receives it with the customer's OTP",how=""),
 "payment_rescue":dict(frm="Customer had no cash",to="Customer pays by UPI link",how=""),"pickup_node":dict(frm="Home delivery failed",to="Customer collects from a nearby trusted shop",how=""),
 "refusal_call":dict(frm="Customer refused at the door",to="Customer accepts after a call or exchange offer",how=""),"local_recovery":dict(frm="Return to seller",to="Sold to a nearby buyer or retailer",how=""),
 "planned_reverse":dict(frm="Individual return at full cost",to="Consolidated return batch",how="15% cheaper")}
SUCC={"forward_stock":"Forwarded to a hub with open orders, stock kept","retry_window":"Delivered in the customer's window","neighbour":"Delivered to a neighbour","payment_rescue":"Delivered, paid by UPI","pickup_node":"Collected from the nearby shop","ldp":"Delivered by the local partner","geo_pin":"Delivered using the map pin","refusal_call":"Customer accepts the parcel",
 "local_recovery":"Sold nearby, parcel value kept","planned_reverse":"Returns in a planned batch"}
def post_actions(p):
    f=p["f"]; h=p["now"]-p["fail_h"]; left=48-h; rc=recovery(p); k=dk(h); out=[]; un=[]; bl=max(o["p"] for o in rc["options"] if o["id"]!="reverse")
    for a in OPTS[p["reason"]]:
        if a in p["used"]: continue
        nm,mod,cost,pr,_=ACT[a]; dec=a not in("local_recovery","planned_reverse")
        if a=="forward_stock":
            hs=redir(p)
            if p["cond"]<.6: un.append(dict(id=a,name=nm,why="Parcel condition %d%%, needs 60%% or more to resell"%round(p["cond"]*100))); continue
            if not hs: un.append(dict(id=a,name=nm,why="No other hub has 3 or more open orders for this item right now")); continue
            cost=hs[0]["cost"]; pr=hs[0]["p"]*math.sqrt(k); need=6
        elif a=="retry_window": pr=clamp(f["window_success"]*.85*f["contact_rate"]*(.6 if p["reason"]=="weak" else 1))*k; need=18
        elif a=="local_recovery": pr=bl; need=6
        elif a=="planned_reverse": need=4
        elif a=="ldp": pr*=k; need=12
        else: pr*=k; need=4
        if left<need: un.append(dict(id=a,name=nm,why="Needs %d h, only %d h of rescue window left"%(need,max(0,round(left))))); continue
        st=seats(p,a)
        if st is not None and st<=0: un.append(dict(id=a,name=nm,why="No seats left at this hub right now (0 of %d)."%CAP[a])); continue
        cond_only=a in("local_recovery","forward_stock")     # cost is only incurred when it sells
        if cond_only: evv=pr*(REV-cost); tot=pr*cost+(1-pr)*REV
        elif a=="planned_reverse": evv=REV*PLAN_R-cost; tot=REV*(1-PLAN_R)+cost
        else: evv=pr*REV-cost; tot=cost+(1-pr)*REV
        f12=(dk(h+12)/k if dec else (math.sqrt(dk(h+12)/k) if a=="forward_stock" else 1.0)); p12=pr*f12
        if left-12<need: e12=None
        elif cond_only: e12=p12*(REV-cost)
        elif a=="planned_reverse": e12=evv
        else: e12=p12*REV-cost
        out.append(dict(id=a,name=nm,module=mod,cost=cost,p=round(pr,2),ev=round(evv,1),dp=None,gate=5,need_h=need,mins=MINS[a],must=False,clears=False,partial=False,evidence="PROPOSED (market-informed); p and cost ASSUMED" if a in EXT else "ASSUMED",
          exp_total=round(tot,1),succ=SUCC.get(a,"Resolved"),p12=round(p12,2),ev12=None if e12 is None else round(e12,1),seats=seats(p,a),cap=CAP.get(a)))
    return sorted(out,key=lambda x:-x["ev"]),un
def apply_fx(p,a):
    nf,g=fx_project(p["f"],a); p["f"].update(nf)
    if g: p["rec_h"]=p.get("rec_h",0)+g   # time genuinely recovered by re-slot / lane switch / re-route
    if "cod" in ACT[a][4]: p["payment"]="Prepaid"
    if a=="reroute_hub":
        p["wrong_hub"]=None
        if p.get("late_cause")=="wrong_hub": p["late_cause"]=None
def finish(p,outcome,rev=0.):
    p["outcome"]=outcome; p["rev"]=rev
    p["state"]={"PREVENT":"DELIVERED","SAVE":"DELIVERED","RECOVER":"RECOVERED","CHEAPER":"REVERSING","RTO":"REVERSING"}[outcome]
    if outcome in("PREVENT","SAVE"): p["stage"]=9
    L=p["learn"]; f=p["f"]
    if outcome=="SAVE": L.append(f"Failure '{REASONS[p['reason']]}' resolved by {p['used'][-1]} → preferred-window & door memory updated for this customer (Failure DNA).")
    if outcome=="PREVENT": L.append("Clean delivery: address & window confidence reinforced for this locality." + (" Upstream rescue logged in Contribution Ledger." if p["spent"]>0 else ""))
    if outcome=="RECOVER": L.append((f"Stock forwarded to {p['recover_via']} where the item had open orders: cross-hub demand signal fed to Seller Delivery Intelligence." if p.get("recover_via") else f"Local demand absorbed the parcel in {p['city']}: seller SKU × pincode refusal signal fed to Seller Delivery Intelligence."))
    if outcome=="CHEAPER": L.append("Unrecoverable RTO made cheaper via planned reverse; seller gets SKU-geography risk flag (request prepaid / partial COD).")
    if outcome=="RTO": L.append("Rescue window closed with no action: default reverse at the baseline Rs 120. Missed-rescue value recorded in the Contribution Ledger.")
    if f["rider_fit"]<.75: L.append("Rider-zone mismatch associated with this failure in locality – ZoneFit weight raised.")
    if f["partner_rel"]<.7: L.append("Partner reliability below baseline on this lane – Hub-Fit priors updated.")
def fail(p,reason):
    p["reason"]=reason; p["state"]="HELD-OPEN"; p["fail_h"]=p["now"]; p["fails"]+=1; p["f"]["prior_fails"]=p["fails"]; p["used"]=[]
    ev(p,f"Attempt failed: {REASONS[reason]} → HELD-OPEN (not RTO yet)","bad"); log(p,f"Held open: {REASONS[reason]}. 48h rescue window (ASSUMED) opened; Change-or-Rescue ranks next states.")
def advance(p):
    if p["outcome"] or p["state"] in("REVERSING","RECOVERING"): raise ValueError("Parcel is terminal / on reverse path")
    if p["state"]=="HELD-OPEN": raise ValueError("HELD-OPEN: execute a Change-or-Rescue action (no repeat without change)")
    g,ok,why=gate(p)
    if not ok: raise ValueError(f"LOCKED at Gate {g} ({GATES[g]}): {why}. Execute the recommended action first.")
    s=p["stage"]
    if s<8:
        d=p["dur"][s]; (p.update(late_cause="slow_leg",late_at=ST[s]) if d-p["nom"][s]>.75 else None); p["now"]+=d; p["f"]["delay_min"]=max(0,p["f"]["delay_min"]+(d-p["nom"][s])*60); p["stage"]=s+1; p["state"]=STATE_BY_STAGE(s+1); p["arr"][s+1]=p["now"]
        p["f"]["stage"]=p["stage"]; p["risk_hist"][p["stage"]]=rto_p(p["f"]); ev(p,f"Reached {ST[s+1]} (+{d:.1f}h)","ok")
        log(p,f"Gate {g} cleared → {ST[s+1]}. Risk {p['risk_hist'][p['stage']]:.0%}.")
    else:
        r=np.random.default_rng(p["rng"]+p["fails"]); risk=rto_p(p["f"])
        if r.random()>min(.9,risk*.9): ev(p,"Delivered on first handover","ok"); log(p,"Handover completed."); p["stage"]=9; finish(p,"PREVENT" if p["fails"]==0 else "SAVE")
        else:
            w=reason_weights(p); ks=list(w); pr=np.array([w[k] for k in ks])+1e-6; fail(p,ks[int(r.choice(len(ks),p=pr/pr.sum()))])
def event(p,e):
    f=p["f"]
    if p["outcome"]: raise ValueError("Parcel already resolved")
    late=p["stage"]>=8
    E={"delay":lambda:(f.update(delay_min=f["delay_min"]+300),p.update(now=p["now"]+5),"Missed connection: network delay +5h"),
     "weather":lambda:(f.update(delay_min=f["delay_min"]+60,partner_rel=max(.3,f["partner_rel"]-.05)),p.update(now=p["now"]+1.0),"Weather/traffic disruption +60 min"),
     "address":lambda:(f.update(addr_conf=max(.3,f["addr_conf"]-.25)),None,"Address problem detected (landmark mismatch)"),
     "rider":lambda:(f.update(rider_fit=max(.3,f["rider_fit"]-.3)),None,"Rider-zone mismatch (unfamiliar rider)"),
     "wronghub":lambda:(f.update(delay_min=f["delay_min"]+180,rider_fit=max(.3,f["rider_fit"]-.15)),p.update(now=p["now"]+3),"Routed to wrong hub (+3h)"),
     "phone":lambda:(f.update(contact_rate=max(.3,f["contact_rate"]-.3)),None,"Customer phone unreachable (off or no answer)")}
    if e in E:
        t=E[e]()[-1]; ev(p,t,"bad"); log(p,f"Event: {t}. Model recalculated.")
        if e=="delay": p["late_cause"]="missed_connection"
        elif e=="weather": p["late_cause"]="weather"
        elif e=="wronghub": p["late_cause"]="wrong_hub"; p["wrong_hub"]=alt_city(p["city"])
        return
    if e=="deliver":
        if p["stage"]<8: raise ValueError("Advance parcel to Out for delivery first")
        if p["state"]!="ATTEMPTING": raise ValueError("Use rescue actions on HELD-OPEN parcels")
        p["stage"]=9; ev(p,"Delivered (simulated)","ok"); finish(p,"PREVENT" if p["fails"]==0 else "SAVE"); return
    m={"unavailable":"unavailable","refused":"refused","cod_cash":"cod_cash","weak":"weak","damaged":"damaged"}
    if e in m:
        if p["stage"]<8 or p["state"]!="ATTEMPTING": raise ValueError("Failure events apply at the delivery attempt (advance to Out for delivery)")
        fail(p,m[e]); return
    raise ValueError("unknown event")
def act(p,a,mode="success",who=None):
    """Execute one action. Returns a result dict: what was tried, whether it worked, and the before/after state (worked and cleared are different things)."""
    acts,_=actions_ex(p); av={x["id"]:x for x in acts}
    if a not in av: raise ValueError(f"Action '{a}' is not available in the current state/gate")
    x=av[a]; sfx=" (autopilot)" if who=="auto" else ""; held=p["state"]=="HELD-OPEN"; r=np.random.default_rng(p["rng"]+len(p["log"])*13)
    ok = True if mode=="success" else False if mode=="fail" else bool(r.random()<x["p"])
    b=dict(risk=round(rto_p(p["f"]),3),slack_min=round(slack(p)*60),gate_ok=gate(p)[1],state=p["state"],stage=p["stage"],stage_name=ST[min(p["stage"],9)])
    chg=x["cost"] if (ok or a not in("local_recovery","forward_stock")) else 0     # sale-conditional costs are only paid when it sells
    p["spent"]+=chg; log(p,f"DICE action {x['name']} ({x['module']}): p={x['p']}, cost ₹{chg}, EV ₹{x['ev']} → {'SUCCESS' if ok else 'FAILED'}")
    if held and p.get("first_act_h") is None: p["first_act_h"]=round(p["now"]-p["fail_h"],1)
    why=None if ok else FAILWHY.get(a,"It did not work")
    if not held:
        if ok: apply_fx(p,a); ev(p,f"{x['name']} executed{sfx}","act")
        else: ev(p,f"{x['name']} attempted{sfx}: {why}. Nothing changed","bad")
    else:
        p["used"].append(a)
        if a=="planned_reverse": ok=True; why=None; p["state"]="REVERSING"; ev(p,"Planned Reverse scheduled (consolidated, condition evidenced)","act"); finish(p,"CHEAPER",REV*(1-PLAN_R))
        elif a=="local_recovery":
            if ok: p["state"]="RECOVERING"; ev(p,"Sold locally – value preserved","ok"); finish(p,"RECOVER",0); p["state"]="RECOVERED"
            else: ev(p,f"Local recovery found no buyer","bad")
        elif a=="forward_stock":
            hs=redir(p); b=hs[0] if hs else None
            if ok and b: p["state"]="RECOVERING"; p["recover_via"]=b["hub"]; ev(p,f"Forwarded to {b['hub']} ({b['km']} km) to fill one of {b['orders']} open orders","ok"); finish(p,"RECOVER",0); p["state"]="RECOVERED"
            else: ev(p,f"Forward found no taker: {why}","bad")
        elif ok: p["stage"]=9; ev(p,f"{x['name']} succeeded{sfx} -> delivered after failed attempt","ok"); finish(p,"SAVE")
        else: ev(p,f"{x['name']} did not resolve{sfx}: {why}","bad"); p["f"]["prior_fails"]+=1
    if ok and a in CAP: USED[(p["city"],a)]=USED.get((p["city"],a),0)+1
    p["decisions"].append(dict(t=clock(p["now"]),tick=SIM["tick"],kind="acted",action=a,name=x["name"],ev=x["ev"],p=x["p"],cost=x["cost"],ok=bool(ok),who="autopilot" if who=="auto" else "operator",why=why))
    if len(p["decisions"])>30: del p["decisions"][:10]
    aft=dict(risk=round(rto_p(p["f"]),3),slack_min=round(slack(p)*60) if not p["outcome"] else None,gate_ok=gate(p)[1],state=p["state"],stage=p["stage"],stage_name=ST[min(p["stage"],9)],outcome=p["outcome"])
    nx=actions(p); nxr=nx[0] if nx and (nx[0]["ev"]>0 or nx[0].get("must")) else None
    if not ok and nx:                      # after a failed try, offer a different option first (the same one stays available as a retry)
        alt=[y for y in nx if y["id"]!=a and (y["ev"]>0 or y.get("must"))]; nxr=alt[0] if alt else nx[0]
    if p["outcome"]: kind="resolved"
    elif not ok: kind="failed"
    elif held: kind="failed"
    elif aft["gate_ok"]: kind="fixed"
    else: kind="partial"
    OT={"PREVENT":"delivered on the first try","SAVE":"delivered after rescue","RECOVER":"sold nearby, value kept","CHEAPER":"returning in a planned batch (cheaper)","RTO":"returning at full cost"}
    if kind=="resolved": msg=f"{x['name']}: {OT[p['outcome']]}."
    elif kind=="failed": msg=f"{x['name']} did not work: {why}. ₹{chg:g} spent, nothing changed. This had a {round((1-x['p'])*100)}% chance of happening."
    elif kind=="fixed": msg=f"{x['name']} worked. Risk {b['risk']:.0%} to {aft['risk']:.0%}, slack {b['slack_min']:+d} to {aft['slack_min']:+d} min, cost ₹{x['cost']:g}. The parcel can move on."
    else: msg=f"{x['name']} worked, but the parcel is still {abs(aft['slack_min']) if aft['slack_min'] is not None else 0} min behind. ₹{x['cost']:g} spent."+(f" Next: {nxr['name']}." if nxr else "")
    res=dict(kind=kind,worked=bool(ok),action=a,name=x["name"],cost=chg,p=x["p"],why_failed=why,before=b,after=aft,msg=msg,next=dict(id=nxr["id"],name=nxr["name"],must=bool(nxr.get("must"))) if nxr else None,mode=mode)
    p["last_result"]=res; return res
def hold(p,steps=3,note=None):
    """Operator decides to keep the parcel as it is for a few simulation steps. Logged so the decision (and what was given up) is visible."""
    if p["outcome"]: raise ValueError("Parcel already resolved")
    acts=actions(p); best=acts[0] if acts and (acts[0]["ev"]>0 or acts[0].get("must")) else None; steps=max(1,min(12,int(steps)))
    p["hold"]=SIM["tick"]+steps
    p["decisions"].append(dict(t=clock(p["now"]),tick=SIM["tick"],kind="kept",action=None,name="Keep as it is",ev=best["ev"] if best else 0,p=None,cost=0,ok=None,who="operator",why=note))
    ev(p,f"Operator kept it as it is for {steps} steps"+(f" (best option was worth ₹{best['ev']:g})" if best and best["ev"]>0 else ""),"info")
    worth=best and best["ev"]>0 and not best.get("must")
    return dict(kind="kept",steps=steps,msg=f"Kept as it is for {steps} steps."+(f" The best alternative was worth about ₹{best['ev']:g}." if worth else (" Note: the recommended fix was required to keep the promise." if best and best.get("must") else "")))
def triage(p):
    """Four-State Triage (M8): GREEN keep trying / BLUE change how we try / ORANGE rescue nearby / GREY let go. Rule mapping is PROPOSED."""
    if p["outcome"]: return None
    if p["state"]!="HELD-OPEN": return "GREEN"
    r=p["reason"]
    if r=="damaged" or (r=="refused" and max(p["cd"]/18,p["rd"]/8)<.15): return "GREY"
    if r=="refused" or p["fails"]>=2 or p["used"]: return "ORANGE"
    return "BLUE"
def urgent(p,sl):
    if p["outcome"]: return None
    if p["state"]=="HELD-OPEN": return round((48-(p["now"]-p["fail_h"]))*60)
    return round(sl*60)
VC={}
def view(p,full=False):
    k=(p['id'],full,len(p['events']),len(p['log']),p['stage'],p['state'],p['spent'],round(p.get('rec_h',0),2),len(p['used']),round(p['now'],2),p['outcome'],tuple(USED.get((p['city'],a),0) for a in CAP))
    if k not in VC:
        if len(VC)>6000: VC.clear()
        VC[k]=_view(p,full)
    return VC[k]
def _view(p,full=False):
    f=p["f"]; g,ok,why=gate(p); base,contrib=contributors(p) if full else (rto_p(f),{}); rem=eta_rem(p) if not p["outcome"] else 0; sl=slack(p) if not p["outcome"] else 0
    conf=clamp(.5*(1+math.erf((sl/(.25*rem+.6))/1.414))) if not p["outcome"] else 1
    acts,un=actions_ex(p); rec=acts[0] if acts and (acts[0]["ev"]>0 or acts[0].get("must")) else None; s=p["stage"]; sunk=SUNK[min(s,9)]; avoid=FWD-sunk
    if p["state"]=="HELD-OPEN": sunk,avoid=50,0
    health="Resolved" if p["outcome"] else "At Risk" if (sl<0 or not ok) else "Watch" if sl<1 or base>.25 else "Healthy"
    nodes=[]
    for i in range(1,10):
        st="done" if i<=s else "current" if i==s+1 and not p["outcome"] else "future"
        nodes.append(dict(name=ST[i-1] if i>1 else ST[0],idx=i-1,status="done" if i-1<s or (p["outcome"] and i-1<=s) else "current" if i-1==s else "future",planned=clock(p["plan_arr"][i-2]) if i>1 else clock(0),
          actual=clock(p["arr"][i-1]) if (i-1) in p["arr"] else None,delta=round((p["plan_arr"][i-2]-p["arr"][i-1])*60) if i>1 and (i-1) in p["arr"] else None,
          risk=round(p["risk_hist"].get(i-1,0)*100) if (i-1) in p["risk_hist"] else None,sunk=SUNK[i-1],avoid=FWD-SUNK[i-1]))
    nodes=nodes[:9]; nodes.append(dict(name="Delivered",idx=9,status="done" if p["outcome"] in("PREVENT","SAVE") else "future",planned=clock(p["promised"]),actual=clock(p["now"]) if p["outcome"] in("PREVENT","SAVE") else None,delta=None,risk=None,sunk=50,avoid=0))
    nodes=[n for n in nodes if n["idx"]<=9]; nodes=nodes[:1]+nodes[1:]
    dn=rto_p({**f}) ; best=rec
    dnp = None
    if rec and rec["dp"] is not None: dnp=dict(rto_pp=rec["dp"],eta_slip_min=round(max(0,-sl)*60),text="Without action: RTO risk stays %d%%, forecast slips %d min past promise"%(base*100,max(0,-sl)*60))
    econ=dict(sunk=sunk,avoidable=avoid,reverse=REV,ceiling=avoid+REV,rto_p=round(base,3),exp_reverse=round(base*REV,1),action_cost=rec["cost"] if rec else 0,ev=rec["ev"] if rec else 0,
      decision="ACT" if rec else ("CLEARED – no action" if not p["outcome"] else p["outcome"]),spent=round(p["spent"],1),reverse_paid=round(p["rev"],1),
      formula="EV = P(success) × ΔRTO-risk × ₹120 (case reverse convention) − action cost. ₹170 is NOT credited as savings: ₹50 forward is sunk-or-paid in the successful counterfactual.")
    inputs={k:(round(v,2) if isinstance(v,float) else v) for k,v in f.items()}
    d=dict(id=p["id"],hub="LMDC "+p["city"],triage=triage(p),urgent_min=urgent(p,sl),distance_km=f["distance_km"],product=p["product"],city=p["city"],segment=p["segment"],category=p["category"],payment=p["payment"],value=p["value"],window=p["window"],state=p["state"],stage=s,stage_name=ST[min(s,9)],
      gate=g,gate_name=GATES[g],gate_ok=ok,gate_why=why,risk=round(base,3),risk0=round(p["risk0"],3),health=health,outcome=p["outcome"],reason=REASONS.get(p["reason"]) if p["reason"] else None,
      eta=dict(promised=clock(p["promised"]),forecast=clock(p["now"]+rem-p.get("rec_h",0)),slack_min=round(sl*60),conf=round(conf,2),deadline=clock(p["promised"]-max(.5,.25*rem)),remaining_h=round(rem,1),now=clock(p["now"])),
      nodes=nodes,contrib=contrib,econ=econ,rec=rec,alts=acts[1:4] if rec else acts[:3],all_actions=acts,donothing=dnp,events=p["events"][-30:],log=p["log"][-30:],learn=p["learn"],
      why=dict(inputs=inputs,model=f"RTO probability = {base:.1%} (HistGradientBoosting, synthetic-trained)",rule="Act only if feasible at this gate, before rescue deadline, and EV>0: P(success)×ΔRisk×₹120 − cost.",evidence=dict(costs="CASE-PACK (₹50/₹120, node costs)",action_p_cost="ASSUMED",model="SYNTHETIC",gates_thresholds="PROPOSED")))
    if p["state"]=="HELD-OPEN": 
        d["failure"]=dict(triage=triage(p),demand=dict(consumers=p["cd"],retailers=p["rd"]),diagnosis=REASONS[p["reason"]],integrity="FLAGGED – needs verification" if p["reason"]=="weak" else "VERIFIED (geo + call log, simulated)",last_contact=f"{int(20+ (1-f['contact_rate'])*100)} min ago (simulated)",window_left_h=round(48-(p["now"]-p["fail_h"]),1),recovery=recovery(p))
    d["diag"]=diagnose(p,base); d["unavailable"]=un; d["decision"]=decision(p,acts,un,base,sl,rem,full,d.get("failure"),rec)
    return d
def decision(p,acts,un,base,sl,rem,full,failure,rec):
    """Now vs after: what is happening, what each option changes, and what it costs. Money in Rs, EV convention unchanged. Nothing here is a forecast of real Valmo data."""
    held=p["state"]=="HELD-OPEN"; opts=[]; n=6 if full else 1
    sel=acts[:n] if not (rec and not full) else [x for x in acts if x["id"]==rec["id"]][:1]
    if held:
        rc=failure["recovery"]; left=failure["window_left_h"]; v12=next((d["v"] for d in rc["decay"] if d["h"]==12),rc["now_value"])
        now=dict(kind="post",stage=ST[min(p["stage"],9)],hub="LMDC "+p["city"],status="Delivery failed: "+REASONS[p["reason"]],window_left_h=left,value_now=rc["now_value"],value_12h=v12,decay=rc["decay"],exp_cost=REV,risk=None,slack_min=None,eta=None,promised=clock(p["promised"]),conf=None,
          text=f"The parcel is held open. Doing nothing sends it back at the full ₹{REV} when the rescue window closes in {left:.0f} h.")
        opts.append(dict(id="hold",name="Keep as it is",does="Nothing changes. The rescue window keeps running and every option gets less likely to work.",where=None,cost=0,p=None,mins=0,must=False,clears=False,partial=False,module=None,evidence="",
          after=dict(status=f"Returns at full cost in {left:.0f} h",exp_cost=REV,risk=None,slack_min=None,eta=None,conf=None),fail=None,exp=dict(cost_total=REV,gain=0.0,risk=None),breakeven_p=None,wait=None))
        for x in sel:
            a=x["id"]; fin=a=="planned_reverse"; lr=a in("local_recovery","forward_stock"); cost_txt=x["cost"]
            aft_cost=REV*(1-PLAN_R)+x["cost"] if fin else x["cost"]
            opts.append(dict(id=a,name=x["name"],does=DOES[a],where=where(p,a),module=x["module"],evidence=x["evidence"],cost=x["cost"],p=x["p"],mins=x["mins"],must=False,clears=False,partial=False,seats=x.get("seats"),cap=x.get("cap"),
              after=dict(status=x["succ"]+(f", value kept about ₹{rc['now_value']}" if a=="local_recovery" else ""),exp_cost=round(aft_cost,1),risk=None,slack_min=None,eta=None,conf=None),
              fail=None if fin else dict(text=(("No buyer found. Nothing is spent, the window keeps running." if a=="local_recovery" else f"{FAILWHY.get(a)}. Nothing is spent, the window keeps running.") if lr else f"{FAILWHY.get(a,'It did not work')}. ₹{x['cost']:g} is spent and the window keeps running."),extra_cost=0 if lr else x["cost"]),
              exp=dict(cost_total=x["exp_total"],gain=x["ev"],risk=None),breakeven_p=None if fin else round(x["cost"]/REV,2) if not lr else 0.0,
              wait=dict(p12=x["p12"],ev12=x["ev12"],value_12h=v12 if lr else None,note=("Not possible after 12 more hours" if x["ev12"] is None else f"In 12 h: {x['p12']:.0%} likely, worth ₹{x['ev12']:g}"+(f", parcel value falls to ₹{v12}" if lr else "")))))
    else:
        g,ok,why=gate(p); f=p["f"]; nf3=dict(f); nf3["delay_min"]=f["delay_min"]+(180 if not ok else 0); r3=rto_p(nf3) if not ok else base; rec_h=p.get("rec_h",0); conf=conf_of(sl,rem)
        now=dict(kind="pre",stage=ST[min(p["stage"],9)],hub="LMDC "+p["city"],status=why,eta=clock(p["now"]+rem-rec_h),promised=clock(p["promised"]),slack_min=round(sl*60),risk=round(base,3),conf=round(conf,2),exp_cost=round(base*REV,1),gate=g,gate_ok=bool(ok),
          text=(f"Blocked at Gate {g} ({GATES[g]}): {why}. It cannot move until this is fixed." if not ok else f"Moving normally. {round(base*100)}% chance of coming back, worth ₹{round(base*REV)} in expected return cost."))
        opts.append(dict(id="hold",name="Keep as it is",does=("Nothing changes. The parcel stays locked and waiting costs about 60 min of slack per step." if not ok else "Nothing changes. The parcel keeps moving with its current risk."),where=None,cost=0,p=None,mins=0,must=False,clears=False,partial=False,module=None,evidence="",
          after=dict(status=("Still blocked; risk %d%% in 3 h"%round(r3*100)) if not ok else "Unchanged",risk=round(r3,3),slack_min=round(sl*60)-(180 if not ok else 0),eta=now["eta"],conf=round(conf,2),exp_cost=round(r3*REV,1)),fail=None,exp=dict(cost_total=round(base*REV,1),gain=0.0,risk=round(base,3)),breakeven_p=None,wait=dict(p12=None,ev12=None,note="Risk in 3 h if nothing changes: %d%%"%round(r3*100))))
        for x in sel:
            a=x["id"]; sla=x["slack_after_min"]; eta_h=x["eta_after_h"]-(rec_h+x["gain_h"]); rx=x["risk_after"]; rexp=x["p"]*rx+(1-x["p"])*base
            opts.append(dict(id=a,name=x["name"],does=DOES[a],where=where(p,a),module=x["module"],evidence=x["evidence"],cost=x["cost"],p=x["p"],mins=x["mins"],must=x["must"],clears=x["clears"],partial=x["partial"],seats=x.get("seats"),cap=x.get("cap"),
              after=dict(status=("Gate cleared, parcel moves on" if x["clears"] else "Improved, but still blocked" if x["must"] else "Risk lowered"),risk=rx,slack_min=sla,eta=clock(p["now"]+eta_h),conf=round(conf_of(sla/60,x["eta_after_h"]),2),exp_cost=round(rx*REV,1)),
              fail=dict(text=f"{FAILWHY.get(a,'It did not work')}. ₹{x['cost']:g} is spent and nothing changes.",extra_cost=x["cost"]),exp=dict(cost_total=round(x["cost"]+rexp*REV,1),gain=x["ev"],risk=round(rexp,3)),
              breakeven_p=round(x["cost"]/(max(1e-6,base-rx)*REV),2) if base>rx else None,wait=None))
    best=next((o for o in opts if rec and o["id"]==rec["id"]),None)
    for o in opts:
        o["best"]=bool(best and o["id"]==best["id"])
        if full and o["id"]!="hold": o["detail"]=details(p,o["id"])
    if full: opts[0]["detail"]=dict(frm=None,to=None,how="",specs=([["Rescue window",f"{failure['window_left_h']:.0f} h left, then it returns at Rs {REV}"],["Success chance later","Falls with every hour (see the 12 h row)"]] if held else [["Gate",f"{gate(p)[2]}"],["What waiting costs","About 60 min of slack per step while blocked"]] if not gate(p)[1] else [["Status","Moving normally, no change"]]),cands=None,cutoff=None,note=None)
    if best:
        must=best["must"]; kind="ACT"; gn=best["exp"]["gain"]
        if must and gn<=0: text=f"Do this now. It is required: the parcel cannot move on and would miss its promised slot. Expected cost goes from ₹{now['exp_cost']:g} to ₹{best['exp']['cost_total']:g}, about ₹{abs(gn):g} to keep the promise. {round(best['p']*100)}% chance it works."
        else: text=("Do this now. " if must else "Worth doing. ")+f"Expected cost falls from ₹{now['exp_cost']:g} to ₹{best['exp']['cost_total']:g}, a gain of ₹{gn:g}, with a {round(best['p']*100)}% chance it works."+(" The parcel cannot move on until it is fixed." if must else "")
    else: kind="KEEP"; text=("Keep as it is. No option pays for itself right now." if acts else "Keep as it is. No option is available right now.")+(f" Best available gain is ₹{acts[0]['ev']:g}." if acts else "")
    return dict(now=now,options=opts,best=best["id"] if best else None,verdict=dict(kind=kind,text=text),unavailable=un,sku=sku_of(p))
def fview(p):
    d=dict(view(p,True)); d["hold_left"]=max(0,p["hold"]-SIM["tick"]); d["decisions"]=p["decisions"][-8:][::-1]; d["last_result"]=p.get("last_result"); return d
def lane(v): return "done" if v["outcome"] else "fix" if (v["state"]=="HELD-OPEN" or v["health"]=="At Risk") else "watch" if v["health"]=="Watch" else "ok"
def row(p):
    v=view(p); r=v["rec"]; dg=v["diag"]; pc=dg[0] if dg else None; dec=v["decision"]; ro=next((o for o in dec["options"] if r and o["id"]==r["id"]),None)
    held=p["hold"]>SIM["tick"]; ln=lane(v); urg=v["urgent_min"]
    if ln=="ok" and r and r["ev"]>0: ln="watch"          # optional, worth-doing action on a healthy parcel
    if held and ln=="fix" and (v["state"]!="HELD-OPEN" or urg is None or urg>360): ln="watch"
    pre=dec["now"]["kind"]=="pre"
    return {k:v[k] for k in("id","hub","product","city","segment","category","payment","value","state","stage","stage_name","gate","gate_name","risk","risk0","health","outcome","reason","triage","urgent_min","distance_km")}|dict(
      lane=ln,eta=v["eta"]["forecast"],promised=v["eta"]["promised"],slack=v["eta"]["slack_min"],gate_ok=v["gate_ok"],gate_why=v["gate_why"],ev=v["econ"]["ceiling"] if not p["outcome"] else 0,
      rec=r["name"] if r else "—",rec_id=r["id"] if r else None,rec_ev=r["ev"] if r else 0,rec_p=r["p"] if r else 0,rec_cost=r["cost"] if r else 0,rec_mins=r.get("mins",5) if r else 0,rec_must=bool(r and r.get("must")),
      window_left_h=v["failure"]["window_left_h"] if "failure" in v else None,fail_value=v["failure"]["recovery"]["now_value"] if "failure" in v else None,
      cause=pc["id"] if pc else None,cause_label=pc["label"] if pc else None,headline=pc["headline"] if pc else None,evidence=pc["evidence"] if pc else None,more=max(0,len(dg)-1),block=bool(pc and pc["block"]),
      c_now=dec["now"]["exp_cost"] if not p["outcome"] else None,c_after=ro["exp"]["cost_total"] if ro else None,gain=ro["exp"]["gain"] if ro else None,
      r_now=dec["now"]["risk"] if pre else None,r_after=ro["after"]["risk"] if (ro and pre) else None,s_now=dec["now"]["slack_min"] if pre else None,s_after=ro["after"]["slack_min"] if (ro and pre) else None,
      where_from=(ro["where"] or {}).get("frm") if ro else None,where_to=(ro["where"] or {}).get("to") if ro else None,held=held,hold_left=max(0,p["hold"]-SIM["tick"]),seats=r.get("seats") if r else None)
def analytics(ps=None):
    ps=list(P.values()) if ps is None else ps; done=[p for p in ps if p["outcome"]]; c={k:sum(1 for p in done if p["outcome"]==k) for k in("PREVENT","SAVE","RECOVER","CHEAPER","RTO")}
    base_exp=sum(p["risk0"] for p in ps)*REV; paid=sum(p["rev"] for p in ps); spend=sum(p["spent"] for p in ps)
    rev_avoided=sum(REV for p in done if p["outcome"] in("SAVE","RECOVER")) + sum(REV*PLAN_R for p in done if p["outcome"]=="CHEAPER")
    return dict(n=len(ps),done=len(done),counts=c,inflight=len(ps)-len(done),at_risk=sum(1 for p in ps if not p["outcome"] and view(p)["health"]=="At Risk"),
      exp_rto=round(sum(view(p)["risk"] for p in ps if not p["outcome"]),1),spend=round(spend,1),reverse_paid=round(paid,1),baseline_expected_reverse=round(base_exp,1),reverse_avoided_resolved=round(rev_avoided,1),
      rec_value=round(sum(recovery(p)["now_value"] for p in ps if p["outcome"]=="RECOVER")),net=round(rev_avoided-spend,1),by_stage=[dict(stage=ST[i],avoid=FWD-SUNK[i],sunk=SUNK[i],n=sum(1 for p in ps if not p["outcome"] and p["stage"]==i)) for i in range(8)],
      note="Reverse avoided/net are prototype-model outcomes on synthetic parcels, not Valmo results. Case cohort: 17 RTO/100 × ₹120 = ₹2,040.")
PRODUCTS=["Kurti set","Sneakers","Kids toy","Handbag","Kitchen set","Bedsheet","Sandals","Baby romper"]
def rand_order(r):
    cs=list(CITIES); return dict(product=str(r.choice(PRODUCTS)),city=cs[int(r.integers(len(cs)))],category=str(r.choice(list(CATS))),payment="COD" if r.random()<.8 else "Prepaid",
           value=int(r.integers(199,1500)),window=str(r.choice(["7-9 PM","11 AM-2 PM","Any time"])),address=str(r.choice(["clear","clear","landmark","incomplete"])),
           **({"ov":{"contact_rate":float(r.uniform(.55,.78))}} if r.random()<.18 else {}))
def seed(n=64):
    r=np.random.default_rng(3)
    for i in range(n):
        p=new_order(rand_order(r))
        for _ in range(int(r.integers(1,11))):
            try: advance(p)
            except ValueError:
                a=actions(p)
                if a and r.random()<.9: act(p,a[0]["id"],"success")
                else: break
def seed_failures():
    """Give the shift a realistic history: a few doorstep failures, some already rescued, some still open (deterministic)."""
    r=np.random.default_rng(5); order=["unavailable","refused","refused","cod_cash","unavailable","address","damaged","weak","refused","unavailable","refused"]; used=set(); made=[]
    for reason in order:
        c=[p for p in P.values() if p["id"] not in used and not p["outcome"] and p["state"] not in DONE_STATES+("HELD-OPEN",) and (p["f"]["is_cod"] or reason in("unavailable","address","damaged","weak"))]
        if not c: continue
        p=c[int(r.integers(len(c)))]; used.add(p["id"])
        try:
            to_door(p)
            if p["state"]=="ATTEMPTING": event(p,reason); made.append(p)
        except ValueError: pass
    def ok(p,a,m="success"):
        try: act(p,a,m)
        except ValueError: pass
    for i,p in enumerate(made):
        if p["state"]!="HELD-OPEN": continue
        if i==0 and any(x["id"]=="retry_window" for x in actions(p)): ok(p,"retry_window")
        elif i==1: ok(p,"local_recovery")
        elif i==2: ok(p,"local_recovery","fail"); ok(p,"planned_reverse")
        elif i in(4,6): p["now"]+=float(r.integers(20,34))
def seed_events():
    """Give in-flight parcels a mix of real problems (delay, weather, wrong hub, address doubt, rider mismatch) so the floor shows different causes."""
    r=np.random.default_rng(11)
    for p in list(P.values()):
        if p["outcome"] or p["state"] in DONE_STATES+("HELD-OPEN",): continue
        s=p["stage"]; x=r.random()
        try:
            if 1<=s<=6:
                if x<.10: event(p,"delay")
                elif x<.16: event(p,"weather")
                elif x<.22: event(p,"wronghub")
                elif x<.30: event(p,"address")
                elif x<.36: event(p,"phone")
            elif s==7 and x<.2: event(p,"rider")
        except ValueError: pass
def seed_all(): seed(); seed_events(); seed_failures(); USED.clear()
def reset(): P.clear(); SEQ[0]=0; VC.clear(); FEED.clear(); USED.clear(); SIM.update(tick=0); seed_all()
SR=np.random.default_rng(42)
DONE_STATES=("REVERSING","RECOVERING","RECOVERED","DELIVERED")
def to_door(p):
    """fast-forward a parcel to the delivery attempt, applying the recommended rescue whenever a gate locks (used by the simulator only)"""
    for _ in range(14):
        if p["outcome"] or p["state"] in("HELD-OPEN","ATTEMPTING"): return
        try: advance(p)
        except ValueError:
            a=actions(p)
            if not a: return
            act(p,a[0]["id"],"success")
def tick(spawn=None,auto=None,mode=None):
    """One simulation step = every parcel that can move goes one stop further. Parcel clocks are per-parcel (each leg adds that leg's simulated hours).
    Autopilot: off = nothing automatic; routine = cheap pre-attempt gate fixes run themselves; full = also post-failure decisions."""
    spawn=SIM["spawn"] if spawn is None else spawn; auto=SIM["auto"] if auto is None else auto; auto={True:"routine",False:"off"}.get(auto,auto); mode=mode or SIM["mode"]; SIM["tick"]+=1
    out=dict(tick=SIM["tick"],moved=0,acted=0,waiting=0,new=0,expired=0)
    for k in list(USED): USED[k]=max(0,USED[k]-1)          # one seat frees per step (a wave / truck leaves)
    for p in list(P.values()):
        if p["outcome"] or p["state"] in DONE_STATES: continue
        try:
            if p["state"]=="HELD-OPEN":
                if auto=="full":
                    a=actions(p)
                    if a: act(p,a[0]["id"],mode,"auto"); out["acted"]+=1; continue
                p["now"]+=2                                   # unattended failures age: the 48h rescue window (ASSUMED) is running
                if 48-(p["now"]-p["fail_h"])<=0:
                    ev(p,"Rescue window closed with no action -> default reverse at baseline cost","bad"); p["state"]="REVERSING"; finish(p,"RTO",REV); out["expired"]+=1
                else: out["waiting"]+=1
                continue
            g,ok,_=gate(p)
            if not ok:
                a=actions(p)
                if auto in("routine","full") and a: act(p,a[0]["id"],mode,"auto"); out["acted"]+=1; g,ok,_=gate(p)
                if not ok: p["now"]+=1; out["waiting"]+=1; continue      # locked: waiting time passes, slack shrinks
            x_=SR.random(); s_=p["stage"]                    # natural noise: a mix of causes, not one
            if s_==0 and x_<.05: event(p,"address")
            elif 0<s_<7 and x_<.10: event(p,str(SR.choice(["delay","delay","weather","address","phone","wronghub"])))
            elif s_==7 and x_<.06: event(p,"rider")
            if auto=="full" and p["stage"]<=7 and not p["outcome"]:      # full autopilot also takes cheap worth-doing options
                a_=actions(p)
                if a_ and a_[0]["ev"]>0 and not a_[0].get("must"): act(p,a_[0]["id"],mode,"auto"); out["acted"]+=1
            advance(p); out["moved"]+=1
        except ValueError: out["waiting"]+=1
    for _ in range(spawn): new_order(rand_order(SR)); out["new"]+=1
    return out
INJ={"phone":("phone",range(0,8),0),"network_delay":("delay",range(1,7),0),"weather":("weather",range(1,9),0),"wronghub":("wronghub",range(3,7),0),"address":("address",range(0,8),0),"rider":("rider",range(6,9),0),
    "not_home":("unavailable",None,1),"refusals":("refused",None,1),"no_cash":("cod_cash",None,1)}
def inject(kind,hub=None,n=5):
    if kind not in INJ: raise ValueError("unknown disruption")
    e,stages,door=INJ[kind]; cand=[p for p in P.values() if not p["outcome"] and p["state"] not in DONE_STATES+("HELD-OPEN",) and (not hub or p["city"]==hub)]
    if door: cand=[p for p in cand if (kind=="not_home" or p["f"]["is_cod"])]
    else: cand=[p for p in cand if p["stage"] in stages]
    if hub and len(cand)<3:      # quiet hub: bring a few new parcels through the network so the disruption has something to act on
        for _ in range(4-len(cand)):
            o=rand_order(SR); o["city"]=hub; o["address"]="clear"; q=new_order(o)
            for _ in range(8):
                if not door and q["stage"] in stages: break
                try: advance(q)
                except ValueError:
                    a=actions(q)
                    if not a: break
                    act(q,a[0]["id"],"success")
            if door or q["stage"] in stages: cand.append(q)
    cand=sorted(cand,key=lambda p:-p["stage"])[:n]; hit=[]
    for p in cand:
        try:
            if door: to_door(p)
            if door and p["state"]!="ATTEMPTING": continue
            event(p,e)
            if kind in("network_delay","wronghub"):        # cascade until the promised wave is actually at risk
                for _ in range(3):
                    if not gate(p)[1]: break
                    event(p,e)
            hit.append(p["id"])
        except ValueError: pass
    return dict(kind=kind,hub=hub,affected=hit,n=len(hit))
def bulk(ids,mode=None):
    mode=mode or SIM["mode"]; res=[]
    for i in ids:
        p=P.get(i)
        if not p: continue
        a=actions(p)
        if not a or p["outcome"]: res.append(dict(id=i,kind="skipped")); continue
        try: r=act(p,a[0]["id"],mode); res.append(dict(id=i,kind=r["kind"],ok=bool(r["worked"]),action=a[0]["id"],name=r["name"],why=r["why_failed"]))
        except ValueError as e: res.append(dict(id=i,kind="stale",error=str(e)))
    n=lambda k:sum(1 for r in res if r["kind"]==k)
    return dict(done=sum(1 for r in res if r.get("ok")),tried=len(res),fixed=n("fixed")+n("resolved"),partial=n("partial"),failed=n("failed"),skipped=n("skipped")+n("stale"),results=res)
def dash(hub=None):
    allrows={p["id"]:row(p) for p in P.values()}; ps=[p for p in P.values() if not hub or p["city"]==hub]; rows=[allrows[p["id"]] for p in reversed(ps)]
    live=[r for r in rows if not r["outcome"]]; fix=[r for r in live if r["lane"]=="fix"]
    # Intervention Budget (M8): rank by value per scarce rescue-minute; gate-blocking (promise-integrity) actions are always funded first
    fix.sort(key=lambda r:(not r["rec_must"],-(r["rec_ev"]/max(1,r["rec_mins"])) if r["rec_id"] else 9e9,r["urgent_min"] if r["urgent_min"] is not None else 0))
    used={}; queue=[]
    for k,r in enumerate(fix):
        h=r["hub"]; m=r["rec_mins"] if r["rec_id"] else 0; used[h]=used.get(h,0)+m
        queue.append(dict(id=r["id"],rank=k+1,cum_min=used[h],funded=bool(r["rec_must"] or used[h]<=SIM["budget"] or not r["rec_id"]),ev_min=round(r["rec_ev"]/max(1,r["rec_mins"]),1) if r["rec_id"] else 0))
    hubs_in=len({r["hub"] for r in rows}) or 1; cap=dict(per_hub=SIM["budget"],hubs=hubs_in,total=SIM["budget"]*hubs_in,used=sum(min(v,SIM["budget"]) for v in used.values()),queued=sum(used.values()),unfunded=sum(1 for q in queue if not q["funded"]),funded=sum(1 for q in queue if q["funded"]))
    stages=[dict(i=i,name=ST[i],n=sum(1 for r in live if r["stage"]==i and r["state"]!="HELD-OPEN"),risk=sum(1 for r in live if r["stage"]==i and r["state"]!="HELD-OPEN" and r["lane"]=="fix"),sunk=SUNK[i],avoid=FWD-SUNK[i],cost=NODE[i]) for i in range(9)]
    held=sum(1 for r in live if r["state"]=="HELD-OPEN")
    sl=[r["slack"] for r in live if r["state"]!="HELD-OPEN" and r["stage"]<8]
    a=analytics(ps); c=a["counts"]; deliv=c["PREVENT"]+c["SAVE"]; nd=a["done"]
    cost_total=FWD*nd+a["reverse_paid"]+a["spend"]
    kpi=dict(fix=len(fix),watch=sum(1 for r in live if r["lane"]=="watch"),ok=sum(1 for r in live if r["lane"]=="ok"),done=nd,live=len(live),held=held,exposed=round(sum(r["risk"]*REV for r in fix)),
      avg_slack=round(sum(sl)/len(sl)) if sl else 0,late=sum(1 for x in sl if x<0),on_plan=round(100*sum(1 for x in sl if x>=0)/len(sl)) if sl else 100,
      deliv_per_100=round(100*deliv/nd,1) if nd else None,ndr_conv=(round(100*c["SAVE"]/max(1,sum(1 for p in ps if p["fails"] and p["outcome"]))) if any(p["fails"] and p["outcome"] for p in ps) else None),
      resp_h=(round(sorted(x)[len(x)//2],1) if (x:=[p["first_act_h"] for p in ps if p.get("first_act_h") is not None]) else None),first_attempt=round(100*c["PREVENT"]/deliv) if deliv else None,cost_per_delivered=round(cost_total/deliv,1) if deliv>=8 else None,spend=a["spend"],net=a["net"])
    # clusters (M4 Failure Contagion / Common-Cause Rescue; Cluster Recovery Broadcast = research-derived extension)
    cl=[]; by={}
    for r in live: by.setdefault(r["hub"],[]).append(r)
    for h,rs in by.items():
        off=[r for r in rs if r["gate"]==2 and not r["gate_ok"]]
        if len(off)>=3: cl.append(dict(id=f"cc-{h}",hub=h,city=h[5:],kind="common_cause",sev="high" if len(off)>=5 else "med",n=len(off),ids=[r["id"] for r in off],tag="M4 - PROPOSED"))
        ho={}
        for r in rs:
            if r["state"]=="HELD-OPEN": ho.setdefault(r["reason"],[]).append(r["id"])
        for reason,ids in ho.items():
            if len(ids)>=3: cl.append(dict(id=f"cb-{h}-{reason}",hub=h,city=h[5:],kind="broadcast",reason=reason,sev="high" if len(ids)>=5 else "med",n=len(ids),ids=ids,tag="M4 extension - PROPOSED"))
    fe=[dict(f,seq=i) for i,f in enumerate(FEED) if not hub or f["city"]==hub][-40:][::-1]
    hubs=[]
    for c_ in CITIES:
        rs=[r for r in allrows.values() if r["city"]==c_]; lv=[r for r in rs if not r["outcome"]]
        hubs.append(dict(hub="LMDC "+c_,city=c_,segment=SEGS[CITIES[c_][0]],n=len(rs),live=len(lv),fix=sum(1 for r in lv if r["lane"]=="fix"),held=sum(1 for r in lv if r["state"]=="HELD-OPEN"),
          avg_risk=round(sum(r["risk"] for r in lv)/len(lv),3) if lv else 0,delivered=sum(1 for r in rs if r["outcome"] in("PREVENT","SAVE")),recovered=sum(1 for r in rs if r["outcome"]=="RECOVER"),reversed=sum(1 for r in rs if r["outcome"] in("CHEAPER","RTO"))))
    mix={}
    for r in live:
        if r["cause"] and r["lane"] in("fix","watch"):
            m=mix.setdefault(r["cause"],dict(id=r["cause"],label=r["cause_label"],n=0,fix=0,gain=0.,block=r["block"],gate=r["gate"])); m["n"]+=1; m["fix"]+=r["lane"]=="fix"; m["gain"]=round(m["gain"]+(r["gain"] or 0),1)
    causes=sorted(mix.values(),key=lambda m:-m["n"])
    dl=[dict(d,id=p["id"],hub="LMDC "+p["city"]) for p in ps for d in p["decisions"]]; dl.sort(key=lambda d:(-d["tick"],d["id"]))
    dsum=dict(acted=sum(1 for d in dl if d["kind"]=="acted"),kept=sum(1 for d in dl if d["kind"]=="kept"),worked=sum(1 for d in dl if d["kind"]=="acted" and d["ok"]),failed=sum(1 for d in dl if d["kind"]=="acted" and not d["ok"]))
    return dict(hub=hub,rows=rows,queue=queue,causes=causes,decisions=dl[:14],dsum=dsum,capacity=cap,stages=stages,kpi=kpi,counts=c,analytics=a,clusters=cl,feed=fe,hubs=hubs,sim=dict(SIM),
      rescue=rescue(ps),ledger=ledger(ps))
def rescue(ps):
    held=[]; fut=0.; fut_n=0; rec=[]; rev={}; fwd=[]
    for p in ps:
        if p["state"]=="HELD-OPEN":
            v=view(p); f=v["failure"]; rc=f["recovery"]; best=max((o for o in rc["options"] if o["id"]!="reverse"),key=lambda o:o["p"])
            hs=redir(p) if p["cond"]>=.6 and p["reason"]!="damaged" else []
            fwd.append(dict(id=p["id"],product=p["product"],sku=sku_of(p),value=p["value"],from_hub=v["hub"],to_hub=hs[0]["hub"],km=hs[0]["km"],orders=hs[0]["orders"],cost=hs[0]["cost"],hrs=hs[0]["hrs"],p=hs[0]["p"],net=hs[0]["net"],window_left_h=f["window_left_h"])) if hs else None
            held.append(dict(id=p["id"],home=(dict(hub=hs[0]["hub"],km=hs[0]["km"],orders=hs[0]["orders"]) if hs else None),hub=v["hub"],product=p["product"],city=p["city"],value=p["value"],payment=p["payment"],category=p["category"],reason=f["diagnosis"],reason_id=p["reason"],triage=f["triage"],integrity=f["integrity"],window_left_h=f["window_left_h"],
              now_value=rc["now_value"],decay=rc["decay"],options=rc["options"],best=rc["best"],demand=f["demand"],rec=v["rec"],alts=v["alts"],distance_km=v["distance_km"]))
            if f["triage"]=="ORANGE": fut+=best["p"]*rc["decay"][2]["v"]; fut_n+=1
        if p["outcome"]=="RECOVER": rec.append(dict(id=p["id"],hub="LMDC "+p["city"],product=p["product"],value=p["value"],kept=recovery(p)["now_value"],via=p.get("recover_via")))
        if p["outcome"] in("CHEAPER","RTO"):
            g=rev.setdefault("LMDC "+p["city"],dict(hub="LMDC "+p["city"],n=0,cost=0.,base=0.,planned=0)); g["n"]+=1; g["cost"]+=p["rev"]; g["base"]+=REV; g["planned"]+=1 if p["outcome"]=="CHEAPER" else 0
    for g in rev.values(): g["cost"]=round(g["cost"]); g["saved"]=round(g["base"]-g["cost"])
    order={"ORANGE":0,"BLUE":1,"GREY":2}; held.sort(key=lambda h:(order.get(h["triage"],3),h["window_left_h"]))
    fwd.sort(key=lambda z:-z["net"])
    return dict(forwards=fwd[:8],held=held,recovered=rec[-12:],recovered_n=len(rec),futures=dict(n=fut_n,value=round(fut)),reverse=sorted(rev.values(),key=lambda g:-g["n"]))
def ledger(ps):
    dna={}; seller={}; learn=[]
    for p in ps:
        if p["fails"] or p["reason"]: dna[p["reason"]]=dna.get(p["reason"],0)+1
        g=seller.setdefault(p["category"],dict(category=p["category"],n=0,cod=0,risk=0.,fails=0,reasons={}))
        g["n"]+=1; g["cod"]+=p["f"]["is_cod"]; g["risk"]+=p["risk0"]
        if p["fails"]: g["fails"]+=1; g["reasons"][p["reason"]]=g["reasons"].get(p["reason"],0)+1
        for l in p["learn"]: learn.append(dict(id=p["id"],txt=l))
    sel=[]
    for g in seller.values():
        rs=g["reasons"]; top=max(rs,key=rs.get) if rs else None; cod=g["cod"]/g["n"]; risk=g["risk"]/g["n"]
        act="Request prepaid for high-risk SKU / geography" if risk>=.2 and cod>=.6 else "Use a local delivery partner for hard addresses" if top=="address" else "Allow smaller / partial COD where policy permits" if top in("refused","cod_cash") else "No seller action needed - keep watching"
        sel.append(dict(category=g["category"],n=g["n"],cod=round(cod*100),risk=round(risk*100,1),fails=g["fails"],top=REASONS.get(top,"-") if top else "-",action=act))
    sel.sort(key=lambda x:-x["risk"])
    return dict(dna=[dict(reason=REASONS.get(k,k),id=k,n=v) for k,v in sorted(dna.items(),key=lambda x:-x[1]) if k],seller=sel[:6],learn=learn[-10:][::-1])
def qs(path): return {k:v[0] for k,v in __import__("urllib.parse").parse.parse_qs(__import__("urllib.parse").parse.urlparse(path).query).items()}
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def send(self,code,obj,ct="application/json"):
        b=obj if isinstance(obj,bytes) else json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type",ct); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        with LOCK: return self._get()
    def _get(self):
        u=self.path.split("?")[0]
        try:
            if u=="/" or u=="/index.html": return self.send(200,open(f"{D}/static/index.html","rb").read(),"text/html; charset=utf-8")
            if u=="/api/parcels": return self.send(200,[row(p) for p in reversed(list(P.values()))])
            if u=="/api/dash": q=qs(self.path); return self.send(200,dash(q.get("hub") or None))
            if u=="/api/analytics": q=qs(self.path); return self.send(200,analytics([p for p in P.values() if not q.get("hub") or p["city"]==q["hub"]]))
            if u=="/api/meta": return self.send(200,META|dict(cities=list(CITIES),categories=list(CATS),budget=SIM["budget"]))
            m=re.match(r"/api/parcel/(\w+)$",u)
            if m: return self.send(200,fview(P[m.group(1)]))
            m=re.match(r"/api/whatif/(\w+)$",u)
            if m:
                p=P[m.group(1)]; q=qs(self.path); f=dict(p["f"]); b=rto_p(f)
                for k,pk in(("addr","addr_conf"),("window","window_success"),("rider","rider_fit"),("partner","partner_rel")): f[pk]=clamp(f[pk]+float(q.get(k,0)))
                a=rto_p(f); return self.send(200,dict(before=round(b,3),after=round(a,3),d_pp=round((b-a)*100,1),per100_saved=round((b-a)*100,1),value_per100=round((b-a)*100*REV),label="MODEL SIMULATION (not measured causal impact)"))
            self.send(404,{"error":"not found"})
        except KeyError: self.send(404,{"error":"unknown parcel"})
    def do_POST(self):
        with LOCK: return self._post()
    def _post(self):
        u=self.path.split("?")[0]; body=json.loads(self.rfile.read(int(self.headers.get("Content-Length",0)) or 0) or b"{}")
        try:
            if u=="/api/sim/tick": return self.send(200,tick(body.get("spawn"),body.get("auto"),body.get("mode")))
            if u=="/api/sim/inject": return self.send(200,inject(body["kind"],body.get("hub") or None,int(body.get("n",5))))
            if u=="/api/sim/reset": reset(); return self.send(200,dict(ok=True))
            if u=="/api/sim/config":
                for k in("auto","budget","mode","spawn"):
                    if k in body: SIM[k]=type(SIM[k])(body[k])
                if SIM["auto"] not in("off","routine","full"): SIM["auto"]="routine"
                return self.send(200,dict(SIM))
            if u=="/api/bulk": return self.send(200,bulk(body.get("ids",[]),body.get("mode")))
            if u=="/api/order": p=new_order(body,body.get("seed")); return self.send(200,fview(p))
            m=re.match(r"/api/parcel/(\w+)/hold$",u)
            if m: p=P[m.group(1)]; res=hold(p,body.get("steps",3),body.get("note")); d=fview(p); d["result"]=res; return self.send(200,d)
            m=re.match(r"/api/parcel/(\w+)/(advance|event|act)$",u)
            if m:
                p=P[m.group(1)]; k=m.group(2)
                if k=="advance": advance(p)
                elif k=="event": event(p,body["event"])
                elif k=="act":
                    try: res=act(p,body["action"],body.get("mode","success"))
                    except ValueError as e:
                        if "not available" not in str(e): raise
                        nx=actions(p); res=dict(kind="stale",worked=False,action=body["action"],msg=("Already handled: this parcel is resolved." if p["outcome"] else f"Already handled: the parcel moved on and is now at {ST[min(p['stage'],9)]}. Showing the latest options."),next=dict(id=nx[0]["id"],name=nx[0]["name"]) if nx else None)
                    d=fview(p); d["result"]=res; return self.send(200,d)
                return self.send(200,fview(p))
            self.send(404,{"error":"not found"})
        except ValueError as e: self.send(409,{"error":str(e)})
        except KeyError: self.send(404,{"error":"unknown parcel"})
if __name__=="__main__":
    seed_all(); port=int(os.environ.get("PORT",8000)); print(f"DICE prototype (synthetic data) → http://localhost:{port}"); ThreadingHTTPServer(("127.0.0.1",port),H).serve_forever()
