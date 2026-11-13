"""Lock the 1/3/5 schedule and run paired fixed-rate or common-rate controls."""
from pathlib import Path
from collections import defaultdict
import csv
import sys
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
import simulate as s

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def save(path,rows):
    tmp=path.with_suffix(".tmp")
    with tmp.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)

grid=[round(float(x),2) for x in s.GRID]
params=[]
for bo in grid:
    band,m=("low",1) if bo<=.25 else ("middle",3) if bo<=.65 else ("high",5)
    params.append(dict(beta_o=bo,band=band,M_fixed=3,M_increasing=m,beta_r_fixed="0.15 / 0.30 / 0.50",
        beta_r_common=bo,tau=10,readout=50,repost_updates=40,repeats=100,master_seed=s.MASTER))
lock=OUT/"locked_parameters.csv"
if not lock.exists():
    save(lock,params)
else:
    assert [{k:str(v) for k,v in x.items()} for x in params]==read(lock)
bandpath=OUT/"locked_NO_bands.csv"
if not bandpath.exists():
    save(bandpath,[dict(band=label,lower=lo,upper_exclusive=hi)
        for label,lo,hi in [("low",0,300),("middle",300,600),("high",600,920)]])
if len(sys.argv)<2 or sys.argv[1]=="lock":
    print("Locked 21 scan points",flush=True)
    sys.exit()
phase=sys.argv[1]
assert phase in ("fixed_rate","common_rate")
prior=Path(sys.argv[2]) if len(sys.argv)>2 else OUT.parent/"supplement_2026-11-11/01_increasing_M/trials.csv"
metrics=("m_started","no","nr_total","nr_per_planned","nr_per_started","active_original","active_repost")
model_fields=(*metrics,"preferred_already_original","replacements","original_before_activation","steps","active_at_horizon","repost_sizes")
def key(row):
    return (int(row["replicate"]),float(row["beta_o"]),float(row["beta_r"]),row["pool"],int(row["m_planned"]))
cache={}
sources=[] if "--fresh" in sys.argv[2:] else [prior]
if phase=="common_rate":
    sources.append(OUT/"fixed_rate_trials.csv")
for source in sources:
    for row in read(source):
        assert row["tau"]=="10" and row["read_step"]=="50" and row["repost_updates"]=="40"
        k=key(row)
        if k in cache:
            assert all(cache[k][field]==row[field] for field in (*model_fields,"rng_seed","original_seed")),k
        else:
            cache[k]=row
def noise(rng,t):
    return (rng.random((2,t,s.N,4),dtype=np.float32),
            rng.random((2,t,s.N),dtype=np.float32),
            rng.random((2,t,s.N,4),dtype=np.float32))
parts=OUT/(phase+"_checkpoints")
parts.mkdir(exist_ok=True)
for rep in range(100):
    part=parts/f"{rep:03}.csv"
    if part.exists():
        continue
    seed=int(np.random.SeedSequence([s.MASTER,rep]).generate_state(1,dtype=np.uint64)[0])
    rng=np.random.default_rng(seed)
    original=int(rng.integers(s.N))
    nodes=np.arange(s.N)
    eligible=nodes[(abs(nodes//s.SIZE-original//s.SIZE)+abs(nodes%s.SIZE-original%s.SIZE))>=s.DISTANCE]
    priority=rng.permutation(eligible)
    rng.integers(1,6)
    first,extra=noise(rng,40),noise(rng,10)
    random_arrays=tuple(np.concatenate((a,b),axis=1) for a,b in zip(first,extra))
    rows=[]
    for point in params:
        bo=point["beta_o"]
        rates=(.15,.30,.50) if phase=="fixed_rate" else (bo,)
        pools=("independent","shared") if phase=="fixed_rate" else ("shared",)
        for br in rates:
            for pool in pools:
                memo={}
                for rule in ("fixed","increasing"):
                    m=point["M_"+rule]
                    old=cache.get((rep,bo,br,pool,m))
                    if old:
                        assert int(old["rng_seed"])==seed and int(old["original_seed"])==original
                        result={k:old[k] for k in model_fields}
                        source="verified_archive"
                    elif m in memo:
                        result=memo[m]
                        source="identical_condition"
                    else:
                        result=s.case(original,priority,m,10,pool=="shared","replace",bo,br,random_arrays,horizon=50)
                        source="new_simulation"
                    memo[m]=result
                    assert int(result["m_started"])==m
                    assert int(result["nr_total"])==sum(map(int,result["repost_sizes"].split(";")))
                    rows.append(dict(replicate=rep,rng_seed=seed,original_seed=original,rule=rule,pool=pool,
                        beta_o=bo,beta_r=br,m_planned=m,tau=10,read_step=50,repost_updates=40,source=source,**result))
    save(part,rows)
    if rep%10==9:
        print(phase,rep+1,flush=True)
trials=[row for part in sorted(parts.glob("*.csv")) for row in read(part)]
assert len(trials)==(25200 if phase=="fixed_rate" else 4200)
save(OUT/(phase+"_trials.csv"),trials)
print("Verified",phase,len(trials),flush=True)

