"""End-to-end test of the 5 demo scenarios against the running API: python test_flow.py [port]. Mirrors the UI's demo runner."""
import json,sys,urllib.request as u
PORT=sys.argv[1] if len(sys.argv)>1 else "8000"; API=f"http://localhost:{PORT}/api/"
def post(p,b):
    try: return json.load(u.urlopen(u.Request(API+p,json.dumps(b).encode(),{"content-type":"application/json"})))
    except Exception as e: return json.loads(e.read())
B=dict(product="Kurti",city="Jaipur",category="Ethnic wear",value=699,window="7-9 PM")
SC=[("PREVENT clean",dict(address="clear",payment="Prepaid"),[("adv",8),("ev","deliver")],"PREVENT"),
 ("Network delay rescue",dict(address="clear",payment="COD"),[("adv",3),("ev","delay"),("rec",),("adv",5),("ev","deliver")],"PREVENT"),
 ("Unavailable->retry",dict(address="clear",payment="COD"),[("adv",8),("ev","unavailable"),("act","retry_window")],"SAVE"),
 ("Refuse->local recovery",dict(address="clear",payment="COD"),[("adv",8),("ev","refused"),("act","local_recovery")],"RECOVER"),
 ("Recovery fails->reverse",dict(address="landmark",payment="COD"),[("rec",),("adv",8),("ev","refused"),("act","local_recovery","fail"),("act","planned_reverse")],"CHEAPER")]
ok=True
for name,o,steps,want in SC:
    v=post("order",B|o); pid=v["id"]; trace=[f"G{v['gate']}{'✓' if v['gate_ok'] else '✗'}"]
    for st in steps:
        for _ in range(st[1] if st[0]=="adv" else 1):
            if st[0]=="adv":
                r=post(f"parcel/{pid}/advance",{})
                for _t in range(3):
                    if "error" in r and v["rec"]: trace.append("lock→"+v["rec"]["id"]); v=post(f"parcel/{pid}/act",{"action":v["rec"]["id"]}); r=post(f"parcel/{pid}/advance",{})
            elif st[0]=="ev": r=post(f"parcel/{pid}/event",{"event":st[1]})
            elif st[0]=="rec": r=post(f"parcel/{pid}/act",{"action":v["rec"]["id"]})
            else: r=post(f"parcel/{pid}/act",{"action":st[1],"mode":st[2] if len(st)>2 else "success"})
            if "error" in r: print("  ERROR",st,r["error"][:90]); break
            v=r
    good=v["outcome"]==want; ok&=good; e=v["econ"]
    print(("PASS" if good else "FAIL"),name,"->",v["outcome"],v["state"],f"risk {v['risk0']:.0%}→{v['risk']:.0%} spent ₹{e['spent']} reverse ₹{e['reverse_paid']}",trace[:3])
a=json.load(u.urlopen(API+"analytics")); print("analytics",a["counts"],"net",a["net"]); sys.exit(0 if ok else 1)
