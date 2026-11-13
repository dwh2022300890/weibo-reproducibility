from pathlib import Path
import sys,os,csv,json
os.environ['OPENBLAS_NUM_THREADS']='1'
OLD=Path('D:/weibo_april_filtered/figure2_inference_2026-11-09')
sys.path[:0]=[str(Path(__file__).resolve().parent),str(OLD/'runtime_deps')]
import model_basis as old
import numpy as np,duckdb,pyarrow.parquet as pq
OUT=Path(__file__).resolve().parent
WORK=Path('work/supplement_2026-11-11')
def save(path,head,rows):
 with path.with_suffix('.tmp').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(head);w.writerows(rows)
 path.with_suffix('.tmp').replace(path)
def bins(v):return np.where(v>0,np.floor(np.log2(np.maximum(v,1))),-1).astype(int)
def scan(folder):
 fs=sorted(folder.glob('???.parquet'))
 if len(fs)!=128:raise RuntimeError(f'{folder}: {len(fs)} shards, expected 128')
 return fs
def run(name,folder,heat,where,out,coverage_only=False):
 out.mkdir(exist_ok=True)
 checkpoint=WORK/('fit_'+name);checkpoint.mkdir(exist_ok=True)
 fs=scan(folder)
 c=duckdb.connect();c.execute("SET threads=1;SET memory_limit='64MB';SET enable_external_file_cache=false")
 pieces=[]
 for i,f in enumerate(fs):
  target=checkpoint/f'support_{i:03}.json'
  if not target.exists():
   rows=c.execute(f"""WITH p AS (SELECT oid,CASE WHEN {heat}=0 THEN -1 ELSE floor(log2({heat}))::INT END k,
      count(*) n,sum(y)::DOUBLE y,sum((y>0)::INT) p
      FROM read_parquet('{f.as_posix()}') WHERE {where} AND {heat}>=0 GROUP BY oid,k)
      SELECT k,sum(n)::BIGINT,count(*),sum(n::DOUBLE*n),max(n),sum(y),sum(p)::BIGINT FROM p GROUP BY k ORDER BY k""").fetchall()
   target.write_text(json.dumps(rows))
  pieces.extend(json.loads(target.read_text()))
 c.close()
 totals={}
 for k,n,g,n2,mx,y,p in pieces:
  a=totals.setdefault(k,[0,0,0,0,0,0]);a[0]+=n;a[1]+=g;a[2]+=n2;a[3]=max(a[3],mx);a[4]+=y;a[5]+=p
 support=[]
 for k,a in sorted(totals.items()):
  n,g,n2,mx,y,p=a;effective=n*n/n2;share=mx/n;ok=n>=10000 and g>=100 and effective>=30 and share<=.2
  support.append([k,n,g,effective,share,y/n,p/n,ok])
 save(out/(name+'_support.csv'),['heat_bin','reposts','unique_originals','effective_originals','max_original_share','mean_outcome','positive_probability','passes'],support)
 cap=-2
 for k,n,g,eff,share,y,p,ok in support:
  if k!=cap+1 or not ok:break
  cap=k
 allowed=list(range(-1,cap+1))
 if name=='age_25_h':
  allowed=[r[0] for r in support if r[-1]]
  cap=max(allowed,default=-2)
 if coverage_only:
  (out/(name+'_status.json')).write_text(json.dumps(dict(status='完成',analysis_role='覆盖率对照',cap=cap),ensure_ascii=False));return
 if cap<0:
  (out/(name+'_status.json')).write_text(json.dumps(dict(status='覆盖不足',cap=cap),ensure_ascii=False));return
 sp=old.model(cap)
 if folder.name=='counts':
  sp[0].cache(min(2**(cap+1),262144),1);sp[1].cache(86401,3600);sp[2].cache(172801,3600)
 dim=1+sum(z.size for z in sp)
 G=np.zeros((dim,dim));v=np.zeros(dim);xb=np.zeros(dim);n=0
 def data(f):
  table=pq.read_table(f);a={k:table[k].to_numpy(zero_copy_only=False) for k in table.column_names}
  mask=np.ones(len(a['y']),dtype=bool)
  if 'win' in a: mask&=a['win']==int(where.split('win=')[1].split()[0])
  mask&=(a[heat]>=0)&np.isin(bins(a[heat]),allowed)
  return {k:v[mask] for k,v in a.items()}
 for i,f in enumerate(fs):
  dest=checkpoint/f'gram_{i:03}.npz'
  if not dest.exists():
   a=data(f);gg=np.zeros_like(G);vv=np.zeros_like(v);xx=np.zeros_like(v)
   for j in range(0,len(a['y']),100000):
    sl=slice(j,j+100000);X=old.design(sp,a[heat][sl],a['delay'][sl],a['age'][sl]);y=a['y'][sl]
    gg+=X.T@X;vv+=X.T@y;xx+=X.sum(axis=0)
   np.savez(dest,G=gg,v=vv,xb=xx,n=len(a['y']))
  z=np.load(dest);G+=z['G'];v+=z['v'];xb+=z['xb'];n+=int(z['n'])
 scale=np.sqrt(np.diag(G));scale[scale==0]=1
 normalized=G/np.outer(scale,scale)
 inv=np.linalg.pinv(normalized,rcond=1e-10)/np.outer(scale,scale)
 beta=inv@v
 projection=np.linalg.pinv(normalized,rcond=1e-10)@normalized
 def estimability(q):
  qs=q/scale
  return float(np.linalg.norm(qs-projection@qs)/max(np.linalg.norm(qs),1e-30))
 meat=np.zeros_like(G);clusters=0
 for i,f in enumerate(fs):
  dest=checkpoint/f'meat_{i:03}.npz'
  if not dest.exists():
   a=data(f);ids=np.unique(a['oid']);scores=np.zeros((len(ids),dim))
   for j in range(0,len(a['y']),100000):
    sl=slice(j,j+100000);oo=a['oid'][sl];X=old.design(sp,a[heat][sl],a['delay'][sl],a['age'][sl])
    residual=a['y'][sl]-X@beta
    starts=np.r_[0,np.flatnonzero(oo[1:]!=oo[:-1])+1]
    score=np.add.reduceat(X*residual[:,None],starts,axis=0)
    scores[np.searchsorted(ids,oo[starts])]+=score
   np.savez(dest,meat=scores.T@scores,g=len(ids))
  z=np.load(dest);meat+=z['meat'];clusters+=int(z['g'])
 covariance=inv@meat@inv*clusters/(clusters-1)*(n-1)/(n-np.linalg.matrix_rank(normalized,tol=1e-10))
 def vector(h):
  q=xb/n;q=q.copy();q[1:1+sp[0].size]=sp[0].basis(np.array([np.log1p(h)]))[0];return q
 curves=[]
 for k in allowed:
  h=0 if k==-1 else 2**k;q=vector(h);mu=q@beta;se=np.sqrt(max(0,q@covariance@q))
  if estimability(q)>1e-5:raise RuntimeError(f'{name}: curve not estimable at {k}')
  curves.append((k,h,mu,se,mu-1.96*se,mu+1.96*se))
 save(out/(name+'_curve.csv'),['heat_bin','heat_lower','adjusted_mean','se','ci95_low','ci95_high'],curves)
 effects=[]
 for label,low,high in [('rise',(2,5),(10,13)),('fall',(14,17),(18,20))]:
  if not all(k in allowed for k in range(low[0],low[1]+1)) or not all(k in allowed for k in range(high[0],high[1]+1)):effects.append((label,'覆盖不足',None,None,None,None,None));continue
  def avg(band):
   h=np.expm1(np.linspace(np.log1p(2**band[0]),np.log1p(2**(band[1]+1)),300))
   return np.mean([vector(t) for t in h],axis=0)
  q=avg(high)-avg(low)
  if estimability(q)>1e-5:raise RuntimeError(f'{name}: contrast not estimable')
  eff=q@beta;se=np.sqrt(max(0,q@covariance@q));effects.append((label,'完成',eff,eff-1.96*se,eff+1.96*se,eff-2.241402727604945*se,eff+2.241402727604945*se))
 save(out/(name+'_effects.csv'),['contrast','status','difference','ci95_low','ci95_high','simultaneous95_low','simultaneous95_high'],effects)
 (out/(name+'_status.json')).write_text(json.dumps(dict(status='完成',cap=cap,n=n,unique_originals=clusters,rank=int(np.linalg.matrix_rank(normalized,tol=1e-10)),columns=dim,gram_condition=float(np.linalg.cond(normalized)),coefficients=beta.tolist(),allowed_bins=allowed,max_curve_estimability_error=max(estimability(vector(0 if k==-1 else 2**k)) for k in allowed)),ensure_ascii=False,indent=2))
 print(name,'done',n,cap,flush=True)
if __name__=='__main__':
 task=sys.argv[1]
 if task=='counts':
  for heat in ('h','h2'):run('reposts_vs_'+heat,WORK/'counts',heat,'true',OUT/'03_repost_counts')
 elif task=='age_capture':
  run('age_25_h',WORK/'age_ordered','h','win=25',OUT/'02_age24')
 else:
  for win in (25,27):
   for heat in ('h','hs'):run(f'age_{win}_{heat}',WORK/'age_ordered',heat,f'win={win}',OUT/'02_age24',coverage_only=win==27)

