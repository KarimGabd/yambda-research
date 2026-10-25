"""Compare past-only candidate generators on the fixed Yambda validation slice.
Run: python src/stage3_candidates.py
Inputs: data/processed/events_300.parquet
Outputs: results/stage3_candidates.csv, results/stage3_selected.json
No test labels are read or evaluated.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed' / 'events_300.parquet'
OUT = ROOT / 'results'
DAY = 86400
END = 26000000
T = END - 21 * DAY
H = 7 * DAY
POPULAR_WINDOW = 30 * DAY
RECENT_WINDOW = 60 * DAY
TOP_K = 10
POP_SIZES = (500, 1000, 2000, 5000)
REC_SIZES = (100, 200, 400)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(DATA, columns=['uid','item_id','timestamp','event_type','played_ratio_pct'])
    d['positive'] = d.event_type.eq('listen') & d.played_ratio_pct.gt(50)
    # Never access any events later than the validation label window.
    assert T + H < END - 7 * DAY
    past = d.loc[d.timestamp < T]
    future = d.loc[(d.timestamp >= T) & (d.timestamp < T + H) & d.positive,
                   ['uid','item_id']].drop_duplicates()
    truth = {int(uid): set(group.item_id.astype(int)) for uid, group in future.groupby('uid')}
    assert truth
    history_pos = past.loc[past.positive]
    recent_global = history_pos.loc[history_pos.timestamp >= T - POPULAR_WINDOW]
    counts = recent_global.item_id.value_counts().rename_axis('item_id').reset_index(name='n')
    popular = counts.sort_values(['n','item_id'],ascending=[False,True]).item_id.astype(int).tolist()
    recent = history_pos.loc[history_pos.timestamp >= T - RECENT_WINDOW,
                             ['uid','item_id','timestamp']].sort_values(
                                ['uid','timestamp','item_id'], ascending=[True,False,True]
                             ).drop_duplicates(['uid','item_id'])
    recent_map = {int(uid): group.item_id.astype(int).tolist() for uid,group in recent.groupby('uid')}
    catalog = set(past.item_id.astype(int).unique())
    total = sum(map(len, truth.values()))
    new = sum(len(items-catalog) for items in truth.values())
    print(f'Validation: {len(truth)} users; {total} distinct relevant pairs; '
          f'{new/total:.2%} pairs reference items absent from pre-cutoff catalog')

    rows=[]
    for pop_n in POP_SIZES:
        pop_set = set(popular[:pop_n])
        for recent_n in REC_SIZES:
            hit=0; recalls=[]; sizes=[]
            for uid, relevant in truth.items():
                c = pop_set | set(recent_map.get(uid,[])[:recent_n])
                found = len(relevant & c)
                hit += found
                recalls.append(found/len(relevant))
                sizes.append(len(c))
            rows.append(dict(global_top=pop_n,user_recent=recent_n,
                             candidate_recall_micro=hit/total,
                             candidate_recall_macro=float(np.mean(recalls)),
                             mean_candidate_count=float(np.mean(sizes)),
                             median_candidate_count=float(np.median(sizes)),
                             users_with_labels=len(truth)))
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT / 'stage3_candidates.csv',index=False)
    print('\n=== Candidate comparison on VALIDATION ONLY ===')
    print(frame.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    # Explicit, reproducible budget: maximize macro coverage among <=2000 mean candidates/user.
    eligible=frame.loc[frame.mean_candidate_count<=2000].sort_values(
        ['candidate_recall_macro','mean_candidate_count'],ascending=[False,True])
    if eligible.empty:
        raise RuntimeError('Candidate budget leaves no eligible configurations')
    choice=eligible.iloc[0].to_dict()
    config={
        'global_top':int(choice['global_top']), 'user_recent':int(choice['user_recent']),
        'global_window_days':30,'user_window_days':60,
        'mean_candidate_count':float(choice['mean_candidate_count']),
        'validation_candidate_recall_macro':float(choice['candidate_recall_macro']),
        'validation_candidate_recall_micro':float(choice['candidate_recall_micro']),
        'candidate_budget_mean_per_user':2000,
        'selection_rule':'highest validation macro candidate recall under mean 2000 candidates/user',
        'test_labels_accessed':False,
        'warning':'Only validation labels used for candidate-selection; freeze before final test.'
    }
    (OUT / 'stage3_selected.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('\nSelected candidate specification:')
    print(json.dumps(config,indent=2))
    print('\nSaved:',OUT / 'stage3_candidates.csv', 'and', OUT / 'stage3_selected.json')

if __name__=='__main__':
    main()
