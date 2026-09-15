"""Compute method and ablation rankings from the released observations."""
from pathlib import Path
import argparse
import csv
import json
import time
from method import score_prepared, rank_scores
from evaluate import expected_metrics, rank_metrics, aggregate

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('AC@1','AC@3','Avg@5','MRR','AC@5')

def run(data, output):
    output.mkdir(parents=True, exist_ok=True)
    predictions=[]
    seen=set()
    for line in (data/'prepared_observations.jsonl').open(encoding='utf-8'):
        case=json.loads(line)
        key=(case['system'],str(case['window_minutes']),case['case_id'])
        if key in seen: raise ValueError(f'Duplicate observation: {key}')
        seen.add(key)
        start=time.perf_counter()
        computed=score_prepared(case)
        elapsed=time.perf_counter()-start
        for arm,scores in computed['arms'].items():
            predictions.append({'system':key[0],'window_minutes':key[1],'case_id':key[2],
                                'arm':arm,'candidates':case['candidates'],
                                'scores':scores.tolist(),'ranking':rank_scores(scores,case['candidates']),
                                'six_arm_scoring_seconds':elapsed})
    # Evaluation labels are opened only after all score vectors have been computed.
    labels=json.loads((data/'evaluation_labels.json').read_text(encoding='utf-8'))
    roots={(r['system'],str(r['window_minutes']),r['case_id']):r['root_service'] for r in labels}
    if len(roots)!=len(labels) or set(roots)!=seen:
        raise ValueError('Labels must match the complete observation set one-to-one')
    evaluated=[]
    with (output/'case_results.jsonl').open('w',encoding='utf-8') as handle:
        for row in predictions:
            root=roots[row['system'],row['window_minutes'],row['case_id']]
            value=expected_metrics(row['scores'],row['candidates'],root)
            value['MRR']=value.pop('list_rank_mrr')
            panel='generalization' if row['case_id'].startswith('case_') else 'main'
            result={**row,'panel':panel,'root_service':root,'metrics':value,
                    'ordered_list_metrics':rank_metrics(row['ranking'],root)}
            handle.write(json.dumps(result,separators=(',',':'))+'\n')
            evaluated.append({**result,**value})
    for filename,keys in [('summary.csv',('panel','system','window_minutes','arm')),
                          ('pooled_summary.csv',('panel','window_minutes','arm'))]:
        summaries=aggregate(evaluated,keys)
        with (output/filename).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=[*keys,'n',*METRICS])
            writer.writeheader();writer.writerows(summaries)
    print(json.dumps({'observation_records':len(seen),'arm_records':len(predictions),'output':str(output)}))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/main')
    args=parser.parse_args()
    run(args.data,args.output)

if __name__=='__main__':main()
