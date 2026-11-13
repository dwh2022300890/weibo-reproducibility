"""Network SIR reimplementation for the submitted Weibo model."""
import argparse,csv,json,math,os
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np

SIZE=30
N=SIZE*SIZE
T=40
GAMMA=.1
MASTER=20261110
REPS=100
DISTANCE=5
GRID=np.linspace(0,1,21).round(2)
NB=np.arange(N).reshape(SIZE,SIZE)
PAD=np.pad(NB,1,constant_values=-1)
ADJ=np.stack([PAD[:-2,1:-1],PAD[2:,1:-1],PAD[1:-1,:-2],PAD[1:-1,2:]],axis=-1).reshape(N,4)
NORMAL=1.959963984540054

def case(o,priority,planned,tau,shared,policy,bo,br,noise,tie_rule="contact_lottery",horizon=T):
    # 0 susceptible, -1 reserved, label 1 original, labels >=2 reposts.
    # Recovery removes the node from the active set but preserves first ownership.
    a=np.zeros(N,dtype=np.int16)
    b=a if shared else np.zeros_like(a)
    a[o]=1
    oa=np.array([o],dtype=np.int64)
    ra=np.empty(0,dtype=np.int64)
    preferred=priority[:planned]
    if policy=="reserve" and shared and tau:
        b[preferred]=-1
    p=np.array([bo,br])
    contact,heal,lottery=noise
    started=blocked=replaced=pre_n=steps=0
    for t in range(horizon):
        if t==tau:
            pre_n=int(np.count_nonzero(a==1))
            blocked=int(np.count_nonzero(a[preferred]==1))
            if policy=="replace":
                chosen=priority[b[priority]==0][:planned]
            elif policy=="reserve":
                chosen=preferred
                assert np.all((b[chosen]==0)|(b[chosen]==-1))
            elif policy=="skip":
                chosen=preferred[b[preferred]==0]
            else:
                raise ValueError(policy)
            started=len(chosen)
            replaced=int(np.count_nonzero(~np.isin(chosen,preferred)))
            b[chosen]=np.arange(2,2+started)
            ra=chosen
        if not (len(oa) or len(ra)):
            if t<tau:
                continue
            break
        steps=t+1
        def proposals(active,kind,owner):
            dest=ADJ[active]
            safe=np.maximum(dest,0)
            accepted=(dest>=0)&(owner[safe]==0)&(contact[kind,t,active]<p[kind])
            row,slot=np.nonzero(accepted)
            src=active[row]
            key=lottery[kind,t,src,slot]
            if tie_rule=="original_first":
                key=key+kind
            return dest[row,slot],owner[src],key
        def winners(owner,dest,labels,key):
            if not len(dest):
                return dest
            order=np.lexsort((key,dest))
            d=dest[order]
            win=order[np.r_[True,d[1:]!=d[:-1]]]
            assert np.all(owner[dest[win]]==0)
            owner[dest[win]]=labels[win]
            return dest[win]
        x=proposals(oa,0,a)
        y=proposals(ra,1,b)
        if shared:
            joined=[np.concatenate(v) for v in zip(x,y)]
            new=winners(a,*joined)
            on,rn=new[a[new]==1],new[a[new]>=2]
        else:
            on,rn=winners(a,*x),winners(b,*y)
        oa=np.concatenate((oa[heal[0,t,oa]>=GAMMA],on))
        ra=np.concatenate((ra[heal[1,t,ra]>=GAMMA],rn))
    no=int(np.count_nonzero(a==1))
    rt=int(np.count_nonzero(b>=2))
    assert no>=1 and rt>=started and (not shared or no+rt<=N)
    assert started<=planned and (policy!="reserve" or started==planned)
    assert (policy!="skip" or started+int(shared)*blocked==planned)
    sizes=np.bincount(b[b>=2],minlength=2+started)[2:].tolist()
    assert sum(sizes)==rt and len(sizes)==started
    return dict(no=no,nr_total=rt,nr_per_planned=rt/planned,
                nr_per_started=rt/started if started else None,m_started=started,
                preferred_already_original=blocked,replacements=replaced,original_before_activation=pre_n,
                steps=steps,active_original=len(oa),active_repost=len(ra),active_at_horizon=len(oa)+len(ra),repost_sizes=";".join(map(str,sizes)))

def specs(stage):
    if stage=="baseline":
        return [("manuscript_baseline",True,10,"replace",float(x),float(x),"contact_lottery") for x in GRID]
    cells=[]
    for br in (.15,.30,.50):
        for shared in (True,False):
            for tau in (0,10):
                for bo in GRID:
                    cells.append(("factorial",shared,tau,"replace",float(bo),br,"contact_lottery"))
    for policy,tie in (("reserve","contact_lottery"),("skip","contact_lottery"),("replace","original_first")):
        for bo in GRID:
            cells.append(("rule_sensitivity",True,10,policy,float(bo),.30,tie))
    return cells

def trials(stage,reps):
    rows=[]
    for rep in range(reps):
        seed=int(np.random.SeedSequence([MASTER,rep]).generate_state(1,dtype=np.uint64)[0])
        rng=np.random.default_rng(seed)
        o=int(rng.integers(N))
        v=np.arange(N)
        candidates=v[(abs(v//SIZE-o//SIZE)+abs(v%SIZE-o%SIZE))>=DISTANCE]
        priority=rng.permutation(candidates)
        paper_m=int(rng.integers(1,6))
        noise=(rng.random((2,T,N,4),dtype=np.float32),rng.random((2,T,N),dtype=np.float32),
               rng.random((2,T,N,4),dtype=np.float32))
        for mode,shared,tau,policy,bo,br,tie in specs(stage):
            m=paper_m if mode=="manuscript_baseline" else 3
            z=case(o,priority,m,tau,shared,policy,bo,br,noise,tie)
            rows.append(dict(mode=mode,replicate=rep,rng_seed=seed,original_seed=o,pool="shared" if shared else "independent",
                             tau=tau,seed_policy=policy,tie_rule=tie,beta_o=bo,beta_r=br,m_planned=m,**z))
        if rep%20==19:
            print(stage,rep+1,flush=True)
    return rows

def csv_write(path,rows):
    with path.open("x",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def pilot():
    deg=(ADJ>=0).sum(1)
    assert np.count_nonzero(deg==2)==4 and deg.sum()==3480
    zeros=np.zeros((2,T,N,4),dtype=np.float32)
    rec=np.ones((2,T,N),dtype=np.float32)
    noise=(zeros,rec,zeros)
    seeds=np.array([1,2,3,200,400,600])
    for shared in (True,False):
        for tau in (0,10):
            z=case(0,seeds,3,tau,shared,"replace",0,0,noise)
            assert (z["no"],z["nr_total"],z["m_started"])==(1,3,3)
    delayed=case(0,seeds,3,10,True,"replace",0,0,(zeros,np.zeros_like(rec),zeros))
    assert delayed["m_started"]==3 and delayed["steps"]==11
    for policy in ("replace","reserve","skip"):
        z=case(0,seeds,3,3,True,policy,1,0,noise)
        assert z["m_started"]==(0 if policy=="skip" else 3)
        assert z["replacements"]==(3 if policy=="replace" else 0)
    print("Pilot passed: open lattice; no transmission; waiting for delayed activation; seed collision policies.",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("pilot","baseline","mechanism"))
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True)
    if args.stage=="pilot":
        pilot()
    else:
        rows=trials(args.stage,REPS)
        csv_write(args.out/(args.stage+"_trials.csv"),rows)
        print("Completed",args.stage,len(rows),flush=True)

if __name__=="__main__":
    main()

