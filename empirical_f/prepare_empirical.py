from pathlib import Path
import duckdb,csv,json,os,time,sys
ROOT=Path('D:/weibo_april_filtered')
OUT=Path('outputs/supplement_2026-11-11')
WORK=Path('work/supplement_2026-11-11');WORK.mkdir(exist_ok=True)
c=duckdb.connect();c.execute("SET threads=1;SET memory_limit='96MB';SET preserve_insertion_order=false;SET enable_external_file_cache=false;SET temp_directory='work/supplement_2026-11-11/spill'")
def files(paths):
 return "read_parquet(["+','.join("'"+p.as_posix()+"'" for p in paths)+"],hive_partitioning=false)"
def export(path,query):
 if path.exists():return
 temporary=Path(str(path)+'.tmp')
 if temporary.exists():temporary.unlink()
 c.execute(f"COPY ({query}) TO '{path.as_posix()}.tmp' (FORMAT PARQUET,COMPRESSION ZSTD)")
 Path(str(path)+'.tmp').replace(path)
def csvsave(path,query):
 r=c.execute(query)
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow([x[0] for x in r.description]);w.writerows(r.fetchall())
# Stage the new repost-count outcomes, without repeating any prior model fits.
parts=WORK/'counts_batches';parts.mkdir(exist_ok=True)
for i in range(0,64,8):
 dst=parts/f'b{i:02}'
 if dst.exists():continue
 src=files(sorted((ROOT/'analysis/work/latest').glob('bucket_*.parquet'))[i:i+8])
 c.execute(f"""COPY (SELECT try_cast(r_weibo_id AS UBIGINT) oid,n_zhuan y,n_r_zhan h,n_r_zhuan h2,age_s age,delay_s delay,
   (try_cast(right(r_weibo_id,7) AS UINTEGER)%128)::UTINYINT shard FROM {src}
   WHERE age_s>=0 AND delay_s>=0 AND n_zhuan>=0 AND n_r_zhan>=0 AND n_r_zhuan>=0)
   TO '{dst.as_posix()}' (FORMAT PARQUET,COMPRESSION ZSTD,PARTITION_BY(shard))""")
 print('counts batch',i,flush=True)
dest=WORK/'counts';dest.mkdir(exist_ok=True)
for j in range(128):
 src=files(sorted(parts.glob(f'b*/shard={j}/*.parquet')))
 export(dest/f'{j:03}.parquet',f'SELECT * FROM {src} ORDER BY oid')
# Window snapshots: filter first, deduplicate independently within each requested win.
age=WORK/'age24';age.mkdir(exist_ok=True)
for k in range(int(os.environ.get('AGE_START','0')),int(os.environ.get('AGE_STOP','64'))):
 paths=sorted((ROOT/'analysis/work/latest_partition_v1').glob(f'day_??/bucket={k}/*.parquet'))
 src=files(paths)
 latest=ROOT/f'analysis/work/latest/bucket_{k:02}.parquet'
 export(age/f'{k:03}.parquet',f"""
 WITH candidates AS (
 SELECT * FROM {src} WHERE age_s>=86400 AND age_s<97200 AND n_zhan>=0 AND n_r_zhan>=0 AND delay_s>=0
 ), windows AS (
 SELECT *,25 AS win FROM candidates WHERE age_s<90000 UNION ALL SELECT *,27 AS win FROM candidates
 ), pick AS (
 SELECT weibo_id rid,win,arg_max(struct_pack(oid:=r_weibo_id,y:=n_zhan,h:=n_r_zhan,age:=age_s,delay:=delay_s,
       event_s:=epoch(t_crawler_time)::BIGINT-28800-age_s),
       (-age_s,t_crawler_time,crawler_ms,selected_id,source_part,file_row_number)) p
 FROM windows GROUP BY rid,win)
 SELECT rid,win,p.*,l.n_r_zhan last_heat FROM pick LEFT JOIN read_parquet('{latest.as_posix()}') l ON rid=l.weibo_id""")
 if k%8==0:print('age bucket',k,flush=True)
if os.environ.get('AGE_STAGE_ONLY')=='1':
 print('age stage worker complete',flush=True)
 sys.exit(0)
while len(list(age.glob('???.parquet')))<64:
 time.sleep(10)
# Copy the selected post snapshots into original-post shards and attach prior snapshot matches.
part=WORK/'age_partition'
if not part.exists():
 c.execute(f"""COPY (SELECT *, (try_cast(right(oid,7) AS UINTEGER)%128)::UTINYINT shard
   FROM read_parquet('{age.as_posix()}/*.parquet')) TO '{part.as_posix()}'
   (FORMAT PARQUET,COMPRESSION ZSTD,PARTITION_BY(shard))""")
final=WORK/'age_ordered';final.mkdir(exist_ok=True)
for j in range(128):
 src=files(sorted(part.glob(f'shard={j}/*.parquet')))
 match=ROOT/f'age_snapshot_2026-11-08/work/matched/{j:03}.parquet'
 export(final/f'{j:03}.parquet',f"""
 WITH target AS MATERIALIZED (SELECT * FROM {src}),
 m AS MATERIALIZED (SELECT * FROM read_parquet('{match.as_posix()}') WHERE repost_id IN(SELECT rid FROM target))
 SELECT try_cast(a.oid AS UBIGINT) oid,a.y,a.h,a.age,a.delay,a.win,a.last_heat,
  CASE WHEN m.lag_ms BETWEEN 1 AND 21600000 THEN m.snapshot_likes ELSE -1 END hs,
  CASE WHEN m.lag_ms BETWEEN 1 AND 21600000 THEN m.lag_ms ELSE -1 END lag
 FROM target a LEFT JOIN m
 ON a.rid=m.repost_id AND a.oid=m.oid AND a.event_s=m.event_s ORDER BY a.oid""")
 print('age match',j,flush=True)
print('empirical preparation complete',flush=True)

