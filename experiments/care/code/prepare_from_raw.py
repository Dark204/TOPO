"""Create CARE input Parquet files from benchmark traces and metric CSV."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json
import numpy as np
from input_mapping import install
import input_helpers

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--system',required=True,choices=['OnlineBoutique','TrainTicket'])
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--inject-time',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    adapter=SimpleNamespace(**{k:getattr(input_helpers,k) for k in ('_trace_columns','normalize_service','_numeric_array','_sorted_profile')})
    install(adapter)
    source={'trace_path':str(args.trace),'profile_path':str(args.profile),'system':args.system}
    edges,trace_meta=adapter.read_edge_spans(source,args.inject_time)
    series,profile_meta=adapter.load_profile_series(source)
    rows=input_helpers.attach_profile(edges,series)
    args.output.mkdir(parents=True,exist_ok=True)
    for part,mask in [('normal',rows.time_sec<args.inject_time),('fault',rows.time_sec>=args.inject_time)]:
        frame=rows.loc[mask].reset_index(drop=True)
        frame['trace_id']=frame['trace_id'].astype(str)
        if part=='fault':frame['__row_id']=np.arange(len(frame),dtype=np.int64)
        frame.to_parquet(args.output/(part+'.parquet'),index=False,compression='zstd')
    (args.output/'mapping.json').write_text(json.dumps({'trace':trace_meta,'profile':profile_meta},indent=2),encoding='utf-8')
if __name__=='__main__':main()
