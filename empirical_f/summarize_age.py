from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json,duckdb,numpy as np
ROOT=Path('D:/weibo_april_filtered')
OUT=Path(__file__).resolve().parent/'02_age24'
WORK=Path('work/supplement_2026-11-11')
c=duckdb.connect();c.execute("SET threads=1;SET memory_limit='64MB';SET enable_external_file_cache=false")
def save(name,head,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(head);w.writerows(rows)
src="read_parquet('"+(WORK/'age_ordered/*.parquet').as_posix()+"')"
count_parts=[]
for path in sorted((WORK/'age_ordered').glob('???.parquet')):
 count_parts.extend(c.execute(f"""SELECT win,count(*),count(DISTINCT oid),count(*) FILTER (WHERE hs>=0),
 count(DISTINCT oid) FILTER (WHERE hs>=0),min(age),max(age),sum(age),
 count(*) FILTER (WHERE y=0) FROM read_parquet('{path.as_posix()}') GROUP BY win""").fetchall())
counts=[]
for win in (25,27):
 rr=[r for r in count_parts if r[0]==win]
 n=sum(r[1] for r in rr)
 counts.append((win,n,sum(r[2] for r in rr),sum(r[3] for r in rr),sum(r[4] for r in rr),
  min(r[5] for r in rr if r[1]),max(r[6] for r in rr if r[1]),sum(r[7] for r in rr)/n,sum(r[8] for r in rr)))
save('window_coverage.csv',['window_upper_hours','reposts','unique_originals','pre_snapshot_reposts','pre_snapshot_unique_originals','min_age_s','max_age_s','mean_age_s','zero_likes'],counts)
hist=c.execute(f"SELECT win,age,count(*) n FROM {src} GROUP BY ALL ORDER BY ALL").fetchall()
save('actual_age_histogram.csv',['window_upper_hours','age_s','reposts'],hist)
qs=[]
for win in (25,27):
 rows=[(a,n) for w,a,n in hist if w==win];ages=np.array([a for a,n in rows]);ns=np.array([n for a,n in rows]);cum=np.cumsum(ns)
 for p in (0,.25,.5,.75,.9,.95,.99,1):
  rank=p*(cum[-1]-1);lo=int(np.floor(rank));hi=int(np.ceil(rank))
  av=ages[np.searchsorted(cum,lo+1)];bv=ages[np.searchsorted(cum,hi+1)]
  qs.append((win,p,float(av+(bv-av)*(rank-lo))))
save('actual_age_quantiles.csv',['window_upper_hours','quantile','age_s'],qs)
with (ROOT/'analysis/tables/bins_latest.csv').open(encoding='utf-8-sig') as f:
 denom={int(r['bin']):int(r['n']) for r in csv.DictReader(f) if r['hours']=='0'}
assert sum(denom.values())==328136749
cov_parts=[]
for path in sorted((WORK/'age_ordered').glob('???.parquet')):
 cov_parts.extend(c.execute(f"""SELECT win,CASE WHEN last_heat=0 THEN -1 ELSE floor(log2(last_heat))::INT END k,
 count(*) n,count(DISTINCT oid),count(*) FILTER(WHERE hs>=0) FROM read_parquet('{path.as_posix()}') GROUP BY ALL""").fetchall())
cov=[]
for win in (25,27):
 for k in sorted(denom):
  rr=[r for r in cov_parts if r[0]==win and r[1]==k]
  cov.append((win,k,sum(r[2] for r in rr),sum(r[3] for r in rr),sum(r[4] for r in rr)))
save('coverage_by_last_observation_heat.csv',['window_upper_hours','last_observation_heat_bin','eligible_month_posts_denominator','observed_near24','coverage','window_snapshot_unique_originals','pre_snapshot_matched'],
 [(w,k,denom[k],n,n/denom[k],g,p) for w,k,n,g,p in cov])
observed_parts=[]
for path in sorted((WORK/'age_ordered').glob('???.parquet')):
 for clock,heat in [('capture','h'),('pre','hs')]:
  observed_parts.extend(c.execute(f"""SELECT win,'{clock}',CASE WHEN {heat}=0 THEN -1 ELSE floor(log2({heat}))::INT END k,
 count(*),count(DISTINCT oid),sum(y),sum((y>0)::INT) FROM read_parquet('{path.as_posix()}')
 WHERE {heat}>=0 GROUP BY ALL""").fetchall())
observed=[]
for key in sorted({tuple(r[:3]) for r in observed_parts}):
 rr=[r for r in observed_parts if tuple(r[:3])==key];n=sum(r[3] for r in rr)
 observed.append((*key,n,sum(r[4] for r in rr),sum(r[5] for r in rr)/n,sum(r[6] for r in rr)/n))
save('observed_heat_support_descriptive.csv',['window_upper_hours','clock','heat_bin','reposts','unique_originals','mean_likes','positive_likes_probability'],observed)
save('sample_flow.csv',['stage','count'],[('unique_april_repost_denominator',328136749)]+[(f'window_{r[0]}',r[1]) for r in counts]+[(f'window_{r[0]}_pre_6h',r[3]) for r in counts])
print('Age coverage ready',counts,flush=True)

lag=c.execute(f'SELECT win,count(*),min(lag),max(lag),avg(lag),quantile_cont(lag,[.25,.5,.75,.95]) FROM {src} WHERE hs>=0 GROUP BY win ORDER BY win').fetchall()
save('pre_snapshot_lag.csv',['window_upper_hours','matched','min_lag_ms','max_lag_ms','mean_lag_ms','p25_p50_p75_p95_ms'],lag)

