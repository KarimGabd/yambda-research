"""Yambda research — leakage-safe handcrafted-feature CatBoost baseline.

Place in yambda-research/src/stage4_manual_catboost.py.
Inputs:
    data/processed/events_300.parquet
    data/artist_item_mapping.parquet
    results/stage3_selected.json
Outputs:
    results/stage4_manual_validation.json
    results/stage4_manual_feature_importance.csv
    results/stage4_manual_validation_scores.parquet
    results/stage4_manual_model.cbm
    results/stage4_manual_run_config.json

Uses ONLY training/validation labels. Events after validation label window are
explicitly removed before any features are constructed. Test is untouched.
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / 'data' / 'processed' / 'events_300.parquet'
UIDS_PATH = ROOT / 'data' / 'processed' / 'selected_uids_300.txt'
ARTIST_PATH = ROOT / 'data' / 'artist_item_mapping.parquet'
CONFIG_PATH = ROOT / 'results' / 'stage3_selected.json'
OUT = ROOT / 'results'
DAY = 86_400
END = 26_000_000
HORIZON = 7 * DAY
TRAIN_TIMES = [END - n * DAY for n in (77, 63, 49, 35)]
VAL_TIME = END - 21 * DAY
TOP_K = 10
SEED = 42


def prepare_events():
    df = pd.read_parquet(EVENTS_PATH, columns=[
        'uid', 'item_id', 'timestamp', 'event_type', 'played_ratio_pct', 'is_organic'
    ], filters=[('timestamp', '<', VAL_TIME + HORIZON)])
    df['positive'] = df.event_type.eq('listen') & df.played_ratio_pct.gt(50)
    assert df.timestamp.max() < VAL_TIME + HORIZON
    assert len(df) > 0
    users = np.sort(np.loadtxt(UIDS_PATH, dtype='uint32', ndmin=1))
    assert len(users) == len(np.unique(users)) == 300
    assert set(df.uid.unique()).issubset(set(users))
    df['uid'] = df.uid.astype('uint32')
    df['item_id'] = df.item_id.astype('uint32')
    return df, users


def load_artist_mapping(df):
    # Metadata are static; sample only item/artist edges needed in this experiment.
    allowed = pd.Index(df.item_id.unique())
    mapping = pd.read_parquet(ARTIST_PATH, columns=['item_id', 'artist_id'])
    mapping = mapping.loc[mapping.item_id.isin(allowed), ['item_id', 'artist_id']]
    mapping = mapping.drop_duplicates(['item_id', 'artist_id']).copy()
    mapping['item_id'] = mapping.item_id.astype('uint32')
    mapping['artist_id'] = mapping.artist_id.astype('uint32')
    print('Artist mapping edges in selected sample:', len(mapping), flush=True)
    return mapping


def get_candidates(past, users, t, spec):
    """EXACT same ranking/tie breaks and union as stage3_candidates.py."""
    gp = int(spec['global_top'])
    ur = int(spec['user_recent'])
    gw = int(spec['global_window_days']) * DAY
    uw = int(spec['user_window_days']) * DAY
    pos = past.loc[past.positive]
    popular_counts = pos.loc[pos.timestamp >= t - gw, 'item_id'].value_counts().rename_axis('item_id').reset_index(name='n')
    popular = popular_counts.sort_values(['n', 'item_id'], ascending=[False, True]).item_id.astype(int).tolist()[:gp]
    pop_rank = {item: i + 1 for i, item in enumerate(popular)}
    recent = pos.loc[pos.timestamp >= t - uw, ['uid', 'item_id', 'timestamp']]
    recent = recent.sort_values(['uid', 'timestamp', 'item_id'], ascending=[True, False, True]).drop_duplicates(['uid', 'item_id'])
    recent_map = {int(uid): g.item_id.astype(int).tolist()[:ur] for uid, g in recent.groupby('uid')}

    chunks = []
    for uid in users:
        uid = int(uid)
        own = recent_map.get(uid, [])
        own_rank = {item: j + 1 for j, item in enumerate(own)}
        choices = sorted(set(popular).union(own))
        if not choices:
            continue
        part = pd.DataFrame({'uid': np.uint32(uid), 'item_id': np.asarray(choices, dtype='uint32')})
        part['global_rank'] = np.asarray([pop_rank.get(i, 0) for i in choices], dtype='float32')
        part['recent_rank'] = np.asarray([own_rank.get(i, 0) for i in choices], dtype='float32')
        part['is_global_candidate'] = (part.global_rank > 0).astype('float32')
        part['is_recent_candidate'] = (part.recent_rank > 0).astype('float32')
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def add_count(frame, base, keys, name):
    """Past-only group statistic. Missing groups mean zero historical events."""
    group = base.groupby(keys, sort=False).size().rename(name).reset_index()
    frame = frame.merge(group, on=keys, how='left', sort=False, validate='many_to_one')
    frame[name] = frame[name].fillna(0).astype('float32')
    return frame


def add_last_age(frame, base, keys, t, name):
    group = base.groupby(keys, sort=False)['timestamp'].max().rename('last_ts').reset_index()
    frame = frame.merge(group, on=keys, how='left', sort=False, validate='many_to_one')
    frame[name] = ((t - frame.last_ts) / DAY).fillna(365).clip(lower=0, upper=365).astype('float32')
    return frame.drop(columns='last_ts')


def add_artist_features(frame, pos, artists, t):
    """Max number of recent positive listens to any artist linked to item.

    Uses all listed artists instead of arbitrarily choosing a primary artist.
    """
    c_artists = frame[['uid', 'item_id']].reset_index(names='row_id').merge(
        artists, on='item_id', how='left', sort=False
    )
    pos60 = pos.loc[pos.timestamp >= t - 60 * DAY, ['uid', 'item_id', 'timestamp']]
    history_artists = pos60.merge(artists, on='item_id', how='inner', sort=False)
    for days in (30, 60):
        name = f'user_artist_pos_{days}d_max'
        hist = history_artists.loc[history_artists.timestamp >= t - days * DAY]
        group = hist.groupby(['uid', 'artist_id'], sort=False).size().rename(name).reset_index()
        joined = c_artists[['row_id', 'uid', 'artist_id']].merge(
            group, on=['uid', 'artist_id'], how='left', sort=False, validate='many_to_one'
        )
        values = joined.groupby('row_id', sort=False)[name].max()
        frame[name] = values.reindex(np.arange(len(frame))).fillna(0).to_numpy(dtype=np.float32)
        del group, joined
    del c_artists, history_artists
    return frame


def snapshot_features(events, users, artists, t, spec, name):
    begin = time.perf_counter()
    past = events.loc[events.timestamp < t]
    assert len(past) and past.timestamp.max() < t
    c = get_candidates(past, users, t, spec)
    pos = past.loc[past.positive]

    # User, item, user-item past-only interaction statistics.
    for days in (7, 30, 60):
        window = pos.loc[pos.timestamp >= t - days * DAY]
        for keys, prefix in ((['uid'], 'user'), (['item_id'], 'item'), (['uid', 'item_id'], 'pair')):
            c = add_count(c, window, keys, f'{prefix}_pos_{days}d')

    for keys, prefix in ((['uid'], 'user'), (['item_id'], 'item'), (['uid', 'item_id'], 'pair')):
        c = add_last_age(c, pos, keys, t, f'{prefix}_days_since_last_positive')

    listening = past.loc[past.event_type.eq('listen') & (past.timestamp >= t - 30 * DAY)]
    likes = past.loc[past.event_type.eq('like') & (past.timestamp >= t - 60 * DAY)]
    for keys, prefix in ((['uid'], 'user'), (['item_id'], 'item'), (['uid', 'item_id'], 'pair')):
        c = add_count(c, listening, keys, f'{prefix}_all_listens_30d')
        c = add_count(c, likes, keys, f'{prefix}_likes_60d')

    pos30 = pos.loc[pos.timestamp >= t - 30 * DAY]
    unique = pos30.groupby('uid', sort=False)['item_id'].nunique().rename('user_unique_pos_items_30d').reset_index()
    c = c.merge(unique, on='uid', how='left', sort=False, validate='many_to_one')
    c['user_unique_pos_items_30d'] = c.user_unique_pos_items_30d.fillna(0).astype('float32')
    distinct_users = pos30.groupby('item_id', sort=False)['uid'].nunique().rename('item_unique_positive_users_30d').reset_index()
    c = c.merge(distinct_users, on='item_id', how='left', sort=False, validate='many_to_one')
    c['item_unique_positive_users_30d'] = c.item_unique_positive_users_30d.fillna(0).astype('float32')
    c['pair_to_user_share_30d'] = c.pair_pos_30d / (1 + c.user_pos_30d)
    c['pair_to_item_share_30d'] = c.pair_pos_30d / (1 + c.item_pos_30d)

    c = add_artist_features(c, pos, artists, t)
    # Label is created only AFTER all features, from [t,t+7d); no future leakage.
    label_pairs = events.loc[
        (events.timestamp >= t) & (events.timestamp < t + HORIZON) & events.positive,
        ['uid', 'item_id']
    ].drop_duplicates()
    label_pairs['label'] = np.uint8(1)
    c = c.merge(label_pairs, on=['uid', 'item_id'], how='left', sort=False, validate='many_to_one')
    c['label'] = c.label.fillna(0).astype('uint8')
    c['snapshot'] = name
    print(f'{name}: candidates={len(c):,}, positives in candidates={int(c.label.sum()):,}, '
          f'feature time={time.perf_counter()-begin:.1f}s', flush=True)
    return c, label_pairs


def evaluation(frame, labels, score_column):
    # The denominator contains ALL distinct future positives, including items
    # absent from the candidate set. Users with no future positives are omitted
    # from ranking metric macro averages (as in earlier stages).
    truth = {int(u): set(g.item_id.astype(int)) for u, g in labels.groupby('uid')}
    assert truth, 'No positive future listen labels in validation'
    top = frame.sort_values(['uid', score_column, 'item_id'], ascending=[True, False, True], kind='stable')
    top = top.groupby('uid', sort=False).head(TOP_K)
    ranked = {int(u): g.item_id.astype(int).tolist() for u, g in top.groupby('uid')}
    cand = {int(u): set(g.item_id.astype(int)) for u, g in frame.groupby('uid')}
    recalls = []
    ndcgs = []
    candidate_recalls = []
    total_hits = 0
    total_pos = 0
    for u, rel in truth.items():
        top10 = ranked.get(u, [])
        hit = len(set(top10) & rel)
        recalls.append(hit / len(rel))
        dcg = sum(1.0/math.log2(j+2) for j, it in enumerate(top10) if it in rel)
        ideal = sum(1.0/math.log2(j+2) for j in range(min(TOP_K, len(rel))))
        ndcgs.append(dcg / ideal if ideal else 0.)
        n_cand = len(cand.get(u, set()) & rel)
        candidate_recalls.append(n_cand / len(rel))
        total_hits += n_cand
        total_pos += len(rel)
    return {
        'evaluated_users_with_positive': len(truth),
        'distinct_future_positive_pairs': total_pos,
        'candidate_recall_macro': float(np.mean(candidate_recalls)),
        'candidate_recall_micro': float(total_hits/total_pos),
        'recall_at_10_macro': float(np.mean(recalls)),
        'ndcg_at_10_macro': float(np.mean(ndcgs)),
        'mean_candidates_per_evaluated_user': float(np.mean([len(cand.get(u, set())) for u in truth])),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    assert (spec['global_top'], spec['user_recent'], spec['global_window_days'], spec['user_window_days']) == (1000,400,30,60), (
        'Unexpected candidate specification. Check stage3_selected.json before training.'
    )
    events, users = prepare_events()
    artists = load_artist_mapping(events)
    print('Frozen candidate spec:', {k:spec[k] for k in ('global_top','user_recent','global_window_days','user_window_days')})
    print('Test labels: NOT LOADED', flush=True)

    features_start = time.perf_counter()
    train = []
    for j, t in enumerate(TRAIN_TIMES, start=1):
        snap, _ = snapshot_features(events, users, artists, t, spec, f'train_{j}')
        train.append(snap)
    val, val_truth = snapshot_features(events, users, artists, VAL_TIME, spec, 'validation')
    feature_generation_sec = round(time.perf_counter() - features_start, 2)
    del artists, events
    gc.collect()

    # Check candidate generation against the user's frozen stage3 result.
    coverage = evaluation(val, val_truth, 'global_rank')
    assert abs(coverage['candidate_recall_macro'] - spec['validation_candidate_recall_macro']) < 1e-8, (
        'Candidates differ from stage3; stop rather than silently changing the experiment', coverage
    )
    assert abs(coverage['candidate_recall_micro'] - spec['validation_candidate_recall_micro']) < 1e-8
    print('Stage3 candidate coverage reproduced exactly.', flush=True)

    all_train = pd.concat(train, ignore_index=True)
    # Freeze exact candidate pairs so subsequent automatic feature experiments
    # can assert both train and validation examples are identical.
    all_train[['snapshot', 'uid', 'item_id', 'label']].to_parquet(
        OUT / 'stage4_train_pairs.parquet', index=False)
    val[['snapshot', 'uid', 'item_id', 'label']].to_parquet(
        OUT / 'stage4_validation_pairs.parquet', index=False)
    del train
    gc.collect()
    non_features = {'uid', 'item_id', 'snapshot', 'label'}
    features = [col for col in val.columns if col not in non_features]
    assert features and all(col in all_train.columns for col in features)
    x_train = all_train[features].astype('float32')
    y_train = all_train.label.astype('uint8')
    x_val = val[features].astype('float32')
    y_val = val.label.astype('uint8')
    print(f'Train rows={len(all_train):,}, positives={int(y_train.sum()):,}; '
          f'validation rows={len(val):,}, positives={int(y_val.sum()):,}; '
          f'{len(features)} features', flush=True)

    model = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        loss_function='Logloss',
        eval_metric='Logloss',
        l2_leaf_reg=4,
        random_seed=SEED,
        thread_count=4,
        allow_writing_files=False,
        verbose=50,
    )
    start = time.perf_counter()
    model.fit(x_train, y_train, eval_set=(x_val,y_val), early_stopping_rounds=50, use_best_model=True)
    training_sec = round(time.perf_counter()-start, 2)
    scores = model.predict_proba(x_val)[:, 1]
    val['score'] = scores
    results = evaluation(val, val_truth, 'score')
    val['popularity_score'] = np.where(val.global_rank > 0, -val.global_rank, -1e9)
    popularity = evaluation(val, val_truth, 'popularity_score')
    results['popularity_recall_at_10_macro'] = popularity['recall_at_10_macro']
    results['popularity_ndcg_at_10_macro'] = popularity['ndcg_at_10_macro']
    results.update({
        'validation_logloss_on_candidates': float(log_loss(y_val, scores, labels=[0,1])),
        'validation_roc_auc_on_candidates': float(roc_auc_score(y_val,scores)),
        'train_candidate_rows': int(len(all_train)),
        'train_positive_candidate_rows': int(y_train.sum()),
        'validation_candidate_rows': int(len(val)),
        'validation_positive_candidate_rows': int(y_val.sum()),
        'feature_count': len(features),
        'best_iteration': int(model.best_iteration_),
        'training_seconds': training_sec,
        'feature_generation_seconds': feature_generation_sec,
        'evaluation_note': 'LogLoss/AUC only on generated candidates; ranking denominators include ALL future positives, with repeats deduplicated per user-item. Test is untouched.',
    })
    model.save_model(str(OUT / 'stage4_manual_model.cbm'))
    val[['uid','item_id','label','score']].to_parquet(OUT / 'stage4_manual_validation_scores.parquet',index=False)
    importance = pd.DataFrame({'feature':features,'importance':model.feature_importances_}).sort_values('importance',ascending=False)
    importance.to_csv(OUT / 'stage4_manual_feature_importance.csv',index=False)
    (OUT / 'stage4_manual_validation.json').write_text(json.dumps(results, indent=2, ensure_ascii=False),encoding='utf-8')
    (OUT / 'stage4_manual_run_config.json').write_text(json.dumps({
        'train_cutoffs':TRAIN_TIMES, 'validation_cutoff':VAL_TIME,
        'test_labels_accessed':False, 'candidate_spec_file':'results/stage3_selected.json',
        'features':features,
        'catboost':{'iterations':400,'depth':6,'learning_rate':0.05,'l2_leaf_reg':4,
                    'random_seed':SEED,'thread_count':4,'early_stopping_rounds':50},
        'ranking_metric':'unconditional macro recall@10 and ndcg@10 on users with >=1 future positive',
    },indent=2),encoding='utf-8')
    print('\n=== Manual CatBoost VALIDATION ===')
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print('\n=== Top 15 feature importances ===')
    print(importance.head(15).to_string(index=False))
    print('\nSaved model and reproducibility artifacts to:',OUT)
    print('TEST UNTOUCHED.')

if __name__ == '__main__':
    main()
