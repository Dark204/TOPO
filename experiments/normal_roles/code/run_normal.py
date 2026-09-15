"""Replay role down-sampling from bundled complete span-role observations."""
from pathlib import Path
import gzip,json,csv,concurrent.futures
from collections import defaultdict
import normal
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduced'
def one(item):
    with gzip.open(ROOT/item['file'],'rt',encoding='utf-8') as f: data=json.load(f)
    source=data['source']; grouped=defaultdict(list)
    for group in data['ordered_role_units']:
        names=group['services'];pair=(group['service'],group['operation'])
        assert group['unit_count']==len(group['inbound'])==len(group['outbound'])
        # Input is already ordered by the original stable span identity.
        # The integer is an observation index, not a fabricated trace/span ID.
        grouped[pair]=[{'stable_id':[i],'inbound_services':[names[j] for j in incoming],'outbound_services':[names[j] for j in outgoing]} for i,(incoming,outgoing) in enumerate(zip(group['inbound'],group['outbound']))]
    identity=normal.evaluate_identity(source,data['operations'],data['kernel_order'])
    result=[]
    for retention in (1.,.5,.25):
        for seed in ((None,) if retention==1 else (17,29,43)):
            trace,details=normal.trace_from_roles(data['operations'],grouped,retention,seed,source['candidate_order'],data['kernel_order'])
            full=normal.evaluate_full(source,trace)
            for item in details['operations']:
                assert item['retained_observation_count']==int(np.floor(retention*item['raw_observation_count']))
                item.pop('h_so',None)  # No unused, uncomputed count in the delivery.
            result.append({'system':source['system'],'case_id':source['case_id'],'retention':retention,'seed':seed,'candidate_order':source['candidate_order'],'full':full,'identity':identity,'operations':details['operations'],'trace':{'x':trace['x'].tolist(),'W':trace['W'].astype(int).tolist(),'q_total':trace['q_total'],'support_count':trace['support_count']}})
    return result

def main():
    OUT.mkdir(exist_ok=True)
    items=json.loads((ROOT/'data/compact_manifest.json').read_text())
    assert len(items)==180
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(one,items)):
            rows.extend(result)
            if (i+1)%20==0: print(json.dumps({'events':i+1}),flush=True)
    with (OUT/'normal_role_case_results.jsonl').open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    metrics=('AC@1','AC@3','Avg@5','list_rank_mrr','AC@5')
    summaries=[]
    for system in ('OnlineBoutique','TrainTicket'):
        for retention in (1.,.5,.25):
            scoped=[r for r in rows if r['system']==system and r['retention']==retention]
            # Every incident has the same number of seeds in a retention group.
            assert len(scoped)==(90 if retention==1 else 270)
            summaries.append({'system':system,'retention':retention,'events':90,'conditions':len(scoped),**{k:float(np.mean([r['full']['metrics']['expected_tie_aware'][k] for r in scoped])) for k in metrics}})
    with (OUT/'normal_role_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    print(json.dumps({'events':180,'conditions':len(rows),'completed':True}),flush=True)
if __name__=='__main__':main()
