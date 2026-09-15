"""Direct fixed-author CARE replay, using only bundled scientific inputs."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json,hashlib,importlib.util,sys,ast,pickle,contextlib,io,csv,argparse,concurrent.futures
import pandas as pd
import numpy as np
from headless import _install_optional_stubs
from evaluate import expected_metrics,rank_metrics
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduced'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def modules():
    _install_optional_stubs();upstream=ROOT/'code/upstream';sys.path.insert(0,str(upstream))
    a=load('care_anomaly',upstream/'CARE/AnomalyDetection_alphacomputation.py')
    f=load('care_features',upstream/'CARE/run_selecting_features.py')
    s=load('care_spectrum',upstream/'CARE/weightedSpectrumAnalysis.py')
    a.DEBUG=False;f.DEBUG=False;f.tqdm=lambda x,*args,**kwargs:x
    source=upstream/'CARE/weightedSpectrumAnalysis.py';tree=ast.parse(source.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='process_file');keep=[];removed=0
    for n in node.body:
        drop=isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'at_colors','nt_colors'} for t in n.targets)
        drop=drop or (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='draw_communiti_based_graphs')
        if drop:removed+=1
        else:keep.append(n)
    assert removed==4
    node.body=keep;exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),s.__dict__)
    return a,f,s

def one(item):
    # No evaluation labels are opened in this function or its author calls.
    slug=hashlib.sha256((item['system']+'/'+item['case_id']).encode()).hexdigest()[:20]
    work=OUT/'work'/slug;work.mkdir(parents=True,exist_ok=True)
    normal=pd.read_parquet(ROOT/item['files']['normal']);fault=pd.read_parquet(ROOT/item['files']['fault'])
    normal['label']=0.;fault['label']=0.
    a,f,s=modules();features=work/'features.txt';chain=work/'observations.pkl'
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        f.selecting_feature_main(Path('input.pkl'),fault,str(features),normal,.1)
        scored=a.invo_anomaly_detection_main(fault,None,str(features),'',1.)
        scored=a.anomaly_severity_assignment(scored,str(features))
        with chain.open('wb') as handle:pickle.dump(scored,handle,protocol=pickle.HIGHEST_PROTOCOL)
        abnormal=scored.loc[scored['predict']==1,'trace_id'].nunique()
        if abnormal==0 or abnormal==scored.trace_id.nunique():
            # The author spectrum requires both graphs. Preserve an unranked
            # outcome; do not invent service scores or a root-label fallback.
            return {'case_id':item['case_id'],'system':item['system'],'candidate_scores':None,'ranking':[],'candidates':item['candidates'],'predict_rows':int((scored['predict']==1).sum()),'scored':False,'status':'no_abnormal_traces' if abnormal==0 else 'no_normal_traces','features_sha256':hashlib.sha256(features.read_bytes()).hexdigest()}
        returned=s.process_file(str(chain))
    ranked=returned[1];graph_scores={str(k):float(v[0]) for k,v in ranked.items()}
    candidates=item['candidates'];scores={k:graph_scores.get(k,0.) for k in candidates}
    assert all(np.isfinite(v) for v in scores.values())
    order=sorted(candidates,key=lambda k:(-scores[k],candidates.index(k)))
    return {'case_id':item['case_id'],'system':item['system'],'candidate_scores':scores,'ranking':order,'candidates':candidates,'predict_rows':int((scored['predict']==1).sum()),'scored':True,'status':'scored','features_sha256':hashlib.sha256(features.read_bytes()).hexdigest()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--limit',type=int);args=ap.parse_args()
    parameters=json.loads((ROOT/'parameters.json').read_text())
    for name,digest in parameters['author_files_sha256'].items():assert hashlib.sha256((ROOT/'code/upstream'/name).read_bytes()).hexdigest()==digest
    items=json.loads((ROOT/'data/input_manifest.json').read_text())
    if args.limit:items=items[:args.limit]
    OUT.mkdir(exist_ok=True);pred=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,row in enumerate(pool.map(one,items)):
            pred.append(row)
            if (i+1)%20==0:print(json.dumps({'scored_events':i+1}),flush=True)
    prelabel=''.join(json.dumps(r,sort_keys=True)+'\n' for r in pred)
    (OUT/'predictions_before_labels.jsonl').write_text(prelabel,encoding='utf-8')
    (OUT/'predictions_before_labels.sha256').write_text(hashlib.sha256(prelabel.encode()).hexdigest(),encoding='utf-8')
    # Evaluation begins only after every selected event has been scored.
    labels={(r['system'],r['case_id']):r['root_service'] for r in json.loads((ROOT/'data/evaluation_labels.json').read_text())}
    for row in pred:
        root=labels[row['system'],row['case_id']];row['root_service']=root
        row['metrics']=expected_metrics([row['candidate_scores'][k] for k in row['candidates']],row['candidates'],root) if row['scored'] else {k:0. for k in ('AC@1','AC@3','Avg@5','list_rank_mrr','AC@5')}
        row['ordinary_metrics']=rank_metrics(row['ranking'],root)
    (OUT/'care_case_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pred),encoding='utf-8')
    summary=[]
    for system in sorted({r['system'] for r in pred}):
        rows=[r for r in pred if r['system']==system]
        summary.append({'system':system,'n':len(rows),'ranked_events':sum(r['scored'] for r in rows),'unranked_events':sum(not r['scored'] for r in rows),'evaluation':'uniform_tie_expectation; unranked=0',**{k:float(np.mean([r['metrics'][k] for r in rows])) for k in ('AC@1','AC@3','Avg@5','list_rank_mrr','AC@5')}})
    with (OUT/'care_main_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(json.dumps({'complete':True,'scored_events':len(pred),'summary':summary}),flush=True)
if __name__=='__main__':main()
