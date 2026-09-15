"""Dataset-only input adapter for the fixed CARE author algorithm.

No labels, roots, metric outcomes or fault categories are used in this module.
"""
from pathlib import Path
import numpy as np
import pandas as pd

def install(adapter):
    def edges(resolved, inject_time):
        path=Path(resolved['trace_path'])
        columns=adapter._trace_columns()
        raw=pd.read_parquet(path,columns=columns) if path.suffix.lower()=='.parquet' else pd.read_csv(path,usecols=columns)
        for k in ('spanID','parentSpanID','traceID','serviceName'):raw[k]=raw[k].astype('string')
        for k in ('startTimeMillis','duration','statusCode'):raw[k]=pd.to_numeric(raw[k],errors='coerce')
        parents=raw.dropna(subset=['traceID','spanID']).drop_duplicates(['traceID','spanID'],keep='last')
        mapping={(str(r.traceID),str(r.spanID)):r.serviceName for r in parents.itertuples(index=False)}
        def source(trace,parent,target):
            # Author raw data include (service, service) root invocations.
            # A missing *nonempty* parent is not silently turned into a root.
            if pd.isna(parent) or str(parent).strip().lower() in {'','0','0000000000000000','none','nan'}:return target
            return mapping.get((str(trace),str(parent)))
        system=str(resolved['system'])
        raw['source']=[source(t,p,s) for t,p,s in zip(raw.traceID,raw.parentSpanID,raw.serviceName)]
        raw['target']=raw.serviceName
        for k in ('source','target'):raw[k]=raw[k].map(lambda x:adapter.normalize_service(system,x))
        raw['trace_id']=raw.traceID
        raw['time_sec']=raw.startTimeMillis/1000.
        raw['latency']=raw.duration # Jaeger duration in microseconds; no TraceRCA normalization.
        raw['http_status']=raw.statusCode.where(raw.statusCode.between(100,599))
        raw['_source_row']=np.arange(len(raw))
        keep=(raw.time_sec>=inject_time-300)&(raw.time_sec<inject_time+300)&raw.trace_id.notna()&raw.source.ne('')&raw.target.ne('')
        window=raw.loc[keep].sort_values(['trace_id','startTimeMillis','_source_row'],kind='mergesort')
        result=window[['trace_id','source','target','time_sec','latency','http_status']].reset_index(drop=True)
        meta={'raw_span_rows':len(raw),'raw_trace_ids':int(raw.traceID.nunique()),'window_edge_rows':len(result),'window_trace_ids':int(result.trace_id.nunique()),'window_start':inject_time-300,'window_end_exclusive':inject_time+300,'self_or_root_rows_retained':int(result.source.eq(result.target).sum()),'parent_mapping_key':'(traceID,spanID)','latency_unit':'microsecond','http_status':'actual HTTP values 100..599 only; absent or non-HTTP codes remain missing'}
        return result,meta

    def profiles(resolved):
        path=Path(resolved['profile_path']);times,frame=adapter._sorted_profile(pd.read_csv(path))
        series={};columns={}
        def get(name):return adapter._numeric_array(frame,name)
        if str(resolved['system']).lower()=='onlineboutique':
            bases=sorted({k.split('_container-',1)[0] for k in frame if '_container-' in k})
            for base in bases:
                name=adapter.normalize_service(str(resolved['system']),base);prefix=base+'_container-'
                mem=get(prefix+'memory-usage-bytes');limit=get(prefix+'spec-memory-limit-bytes')
                ratio=None
                if mem is not None and limit is not None:
                    ratio=np.full(len(times),np.nan);np.divide(mem,limit,out=ratio,where=limit>0)
                vals={'cpu_use':get(prefix+'cpu-usage-seconds-total'),'mem_use_amount':mem,'mem_use_percent':ratio,'file_write_rate':get(prefix+'fs-writes-bytes-total'),'file_read_rate':get(prefix+'fs-reads-bytes-total'),'net_send_rate':get(prefix+'network-transmit-bytes-total'),'net_receive_rate':get(prefix+'network-receive-bytes-total')}
                series[name]={k:(times,v) for k,v in vals.items() if v is not None}
                columns[name]={k:prefix+s for k,s in [('cpu_use','cpu-usage-seconds-total'),('mem_use_amount','memory-usage-bytes'),('file_write_rate','fs-writes-bytes-total'),('file_read_rate','fs-reads-bytes-total'),('net_send_rate','network-transmit-bytes-total'),('net_receive_rate','network-receive-bytes-total')] if k in series[name]}
        else:
            bases=sorted({k.rsplit('_',1)[0] for k in frame if k.endswith('_cpu') or k.endswith('_mem')})
            for name in bases:
                series[name]={k:(times,v) for k,v in [('cpu_use',get(name+'_cpu')),('mem_use_amount',get(name+'_mem'))] if v is not None}
                columns[name]={k:name+suffix for k,suffix in [('cpu_use','_cpu'),('mem_use_amount','_mem')] if k in series[name]}
        meta={'profile_path':str(path),'profile_rows':len(frame),'profile_kind':resolved.get('profile_kind'),'profile_time_min':float(times.min()),'profile_time_max':float(times.max()),'available_profile_features_by_service':{k:sorted(v) for k,v in series.items()},'column_mapping':columns,'transform':'native benchmark metric values; no second difference, no TraceRCA normalization; memory fraction=usage/limit','source_rate_note':'RE2 series bearing counter names are nonmonotone preprocessed observations; values are retained. Exact collector query is not supplied by benchmark. File input uses byte activity, not operation counts.'}
        return series,meta
    adapter.read_edge_spans=edges
    adapter.load_profile_series=profiles
    return adapter
