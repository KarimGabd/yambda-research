"""Stage 5: train-only automatic temporal feature selection on frozen Yambda candidate pairs.

Copy this file to yambda-research/src/stage5_auto_features.py, next to
stage4_manual_catboost.py, then run `python src/stage5_auto_features.py`.

Inputs from earlier stages (unchanged):
  data/processed/events_300.parquet
  data/processed/selected_uids_300.txt
  data/artist_item_mapping.parquet
  results/stage3_selected.json
  results/stage4_train_pairs.parquet
  results/stage4_validation_pairs.parquet
  results/stage4_manual_validation.json
  results/stage4_manual_run_config.json

Produces results/stage5_* artifacts.
NEVER accesses test labels. No external calls or downloads.
"""
from __future__ import annotations

import gc
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

import stage4_manual_catboost as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
DAY = base.DAY
SEED = base.SEED
WINDOWS = (1, 3, 7, 14, 30, 60)
KINDS = ("positive", "listen", "like")
RELATIONS = {
    "user": ["uid"],
    "item": ["item_id"],
    "pair": ["uid", "item_id"],
}
# Candidate source/rank are observable at prediction time for *both* models.
COMMON = ["global_rank", "recent_rank", "is_global_candidate", "is_recent_candidate"]
NUMBER_SELECTED = 24   # 24 automatically selected + 4 frozen candidate features = 28


def spec_catalog():
    """Declarative grammar: every combination is generated, not hand-picked."""
    specs = []
    for kind, days, relation in itertools.product(KINDS, WINDOWS, RELATIONS):
        specs.append(dict(
            name=f"auto_{relation}_{kind}_{days}d_count",
            kind=kind, days=days, relation=relation, aggregation="count",
        ))
    for days in WINDOWS:
        specs.append(dict(name=f"auto_user_positive_{days}d_nunique_items",
                          kind="positive", days=days, relation="user", aggregation="nunique_items"))
        specs.append(dict(name=f"auto_item_positive_{days}d_nunique_users",
                          kind="positive", days=days, relation="item", aggregation="nunique_users"))
        specs.append(dict(name=f"auto_user_artist_positive_{days}d_max_count",
                          kind="positive", days=days, relation="user_artist", aggregation="max_count"))
    assert len(specs) == 72  # 3*6*3 + 6*3
    assert len({s['name'] for s in specs}) == len(specs)
    return specs


def generate_aggregates(frame, events, artists, cutoff, specs):
    """All features are based ONLY on event.timestamp < cutoff.

    `frame` holds frozen candidate pairs and already computed manual features.
    We never use the target (label) or any post-cutoff event in auto features.
    """
    start = time.perf_counter()
    past = events.loc[events.timestamp < cutoff]
    assert len(past) and int(past.timestamp.max()) < cutoff
    out = frame
    kind_masks = {
        'positive': past.positive,
        'listen': past.event_type.eq('listen'),
        'like': past.event_type.eq('like'),
    }
    # Cache window-specific event tables once; share across relationships.
    event_windows = {}
    for kind, days in itertools.product(KINDS, WINDOWS):
        mask = kind_masks[kind] & past.timestamp.ge(cutoff - days * DAY)
        event_windows[(kind, days)] = past.loc[mask, ['uid', 'item_id', 'timestamp']]

    for s in specs:
        if s['relation'] == 'user_artist':
            continue
        kind, days, relation, agg = (s[k] for k in ('kind', 'days', 'relation', 'aggregation'))
        source = event_windows[(kind, days)]
        if agg == 'count':
            keys = RELATIONS[relation]
            counts = source.groupby(keys, sort=False).size().rename(s['name']).reset_index()
        elif agg == 'nunique_items':
            keys = ['uid']
            counts = source.groupby(keys, sort=False)['item_id'].nunique().rename(s['name']).reset_index()
        elif agg == 'nunique_users':
            keys = ['item_id']
            counts = source.groupby(keys, sort=False)['uid'].nunique().rename(s['name']).reset_index()
        else:
            raise AssertionError('Unexpected aggregation: ' + agg)
        prev = len(out)
        out = out.merge(counts, on=keys, how='left', sort=False, validate='many_to_one')
        assert len(out) == prev, 'Auto feature computation changed number of candidate pairs'
        out[s['name']] = out[s['name']].fillna(0).astype('float32')
        del counts

    # Indirect user -> artist -> item relationships. All artists of a track can count.
    artist_specs = [s for s in specs if s['relation'] == 'user_artist']
    if artist_specs:
        c_artists = out[['uid', 'item_id']].reset_index(names='row_id').merge(
            artists, on='item_id', how='inner', sort=False
        )
        # All event timestamps in event_windows come from strictly before cutoff.
        pos_artists = event_windows[('positive', max(WINDOWS))].merge(
            artists, on='item_id', how='inner', sort=False
        )
        for s in artist_specs:
            src = pos_artists.loc[pos_artists.timestamp >= cutoff - s['days'] * DAY]
            counts = src.groupby(['uid', 'artist_id'], sort=False).size().rename(s['name']).reset_index()
            joined = c_artists[['row_id', 'uid', 'artist_id']].merge(
                counts, on=['uid', 'artist_id'], how='left', sort=False, validate='many_to_one'
            )
            maxima = joined.groupby('row_id', sort=False)[s['name']].max()
            out[s['name']] = maxima.reindex(np.arange(len(out))).fillna(0).to_numpy(dtype='float32')
            del src, counts, joined, maxima
        del c_artists, pos_artists

    names = [s['name'] for s in specs]
    assert not out[names].isna().any().any(), 'Missing automatic feature values'
    assert np.isfinite(out[names].to_numpy(dtype='float32')).all(), 'Non-finite auto features'
    return out, time.perf_counter() - start


def duplicate_manual_equivalent(auto_name):
    """Map automatically generated exact equivalents to a manual feature, if any.

    Used ONLY for avoiding duplicate columns in Manual+Auto, not for auto selection.
    """
    import re
    m = re.fullmatch(r'auto_(user|item|pair)_(positive|listen|like)_(\d+)d_count', auto_name)
    if m:
        relation, kind, days = m.groups()
        if kind == 'positive' and int(days) in (7, 30, 60):
            return f'{relation}_pos_{days}d'
        if kind == 'listen' and int(days) == 30:
            return f'{relation}_all_listens_30d'
        if kind == 'like' and int(days) == 60:
            return f'{relation}_likes_60d'
    m = re.fullmatch(r'auto_(user|item)_positive_(\d+)d_nunique_(items|users)', auto_name)
    if m:
        rel, days, count_type = m.groups()
        if int(days) == 30 and rel == 'user' and count_type == 'items':
            return 'user_unique_pos_items_30d'
        if int(days) == 30 and rel == 'item' and count_type == 'users':
            return 'item_unique_positive_users_30d'
    m = re.fullmatch(r'auto_user_artist_positive_(\d+)d_max_count', auto_name)
    if m and int(m.group(1)) in (30, 60):
        return f'user_artist_pos_{m.group(1)}d_max'
    return None


def verify_frozen_pairs(train, val):
    """Exact row-order, labels and candidate identity must equal stage 4."""
    keys = ['snapshot', 'uid', 'item_id', 'label']
    for name, observed in (('train', train), ('validation', val)):
        saved_path = OUT / f'stage4_{name}_pairs.parquet'
        if not saved_path.exists():
            raise FileNotFoundError(f'Missing {saved_path}; run stage4_manual_catboost.py first')
        saved = pd.read_parquet(saved_path, columns=keys)
        actual = observed[keys].reset_index(drop=True)
        assert len(saved) == len(actual), f'{name}: candidate row count changed'
        for col in keys:
            a = saved[col].to_numpy()
            b = actual[col].to_numpy()
            assert np.array_equal(a, b), f'{name}: candidate or label mismatch in {col}'
        print(f'Frozen {name} candidate pairs and labels VERIFIED: {len(saved):,}', flush=True)
        del saved


def get_model(iterations=400, depth=6, verbose=False):
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        loss_function='Logloss',
        eval_metric='Logloss',
        l2_leaf_reg=4,
        random_seed=SEED,
        thread_count=4,
        allow_writing_files=False,
        verbose=50 if verbose else False,
    )


def evaluate_model(name, features, train, val, truth):
    assert len(features) == len(set(features))
    x_train = train[features].astype('float32')
    x_val = val[features].astype('float32')
    y_train = train.label.to_numpy(dtype='uint8')
    y_val = val.label.to_numpy(dtype='uint8')
    model = get_model(verbose=True)
    start = time.perf_counter()
    model.fit(x_train, y_train,
              eval_set=(x_val, y_val), use_best_model=True,
              early_stopping_rounds=50)
    seconds = time.perf_counter() - start
    scores = model.predict_proba(x_val)[:, 1]
    scored = val[['uid', 'item_id', 'label']].copy()
    scored['score'] = scores
    metrics = base.evaluation(scored, truth, 'score')
    metrics.update({
        'validation_logloss_on_candidates': float(log_loss(y_val, scores, labels=[0, 1])),
        'validation_roc_auc_on_candidates': float(roc_auc_score(y_val, scores)),
        'train_candidate_rows': int(len(train)),
        'validation_candidate_rows': int(len(val)),
        'feature_count': len(features),
        'best_iteration': int(model.best_iteration_),
        'training_seconds': round(seconds, 2),
        'test_labels_accessed': False,
    })
    model.save_model(str(OUT / f'stage5_{name}_model.cbm'))
    imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    imp.sort_values('importance', ascending=False).to_csv(
        OUT / f'stage5_{name}_feature_importance.csv', index=False
    )
    scored = scored.rename(columns={'score': f'{name}_score'})
    del x_train, x_val, model
    gc.collect()
    return metrics, scored


def self_test():
    """Small in-memory leakage test, independent of local Parquet files."""
    cutoff = base.VAL_TIME
    rows = [
        (1, 11, cutoff - 1 * DAY, 'listen', 90),
        (1, 12, cutoff - 9 * DAY, 'listen', 70),
        (1, 11, cutoff - 25 * DAY, 'like', 0),
        (2, 12, cutoff - 45 * DAY, 'listen', 100),
        (2, 13, cutoff - 15 * DAY, 'listen', 92),
        (2, 13, cutoff + 1 * DAY, 'listen', 90),
    ]
    e = pd.DataFrame(rows, columns=['uid','item_id','timestamp','event_type','played_ratio_pct'])
    e['positive'] = e.event_type.eq('listen') & e.played_ratio_pct.gt(50)
    artists = pd.DataFrame({'item_id':[11,12,13], 'artist_id':[101,102,101]})
    s = spec_catalog()
    c = pd.DataFrame({'uid':[1,1,2,2], 'item_id':[11,13,12,13]})
    a, _ = generate_aggregates(c.copy(), e, artists, cutoff, s)
    b, _ = generate_aggregates(c.copy(), e.loc[e.timestamp < cutoff].copy(), artists, cutoff, s)
    names = [x['name'] for x in s]
    pd.testing.assert_frame_equal(a[names], b[names])
    assert a.loc[0, 'auto_pair_positive_1d_count'] == 1
    assert a.loc[3, 'auto_pair_positive_1d_count'] == 0
    assert a.loc[0, 'auto_user_artist_positive_60d_max_count'] == 1
    print('SELF-TEST PASSED: exact 72 auto features, strictly past-only; future events do not alter them.')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    self_test()
    needs = [
        'stage4_manual_validation.json', 'stage4_manual_run_config.json',
        'stage4_train_pairs.parquet', 'stage4_validation_pairs.parquet',
    ]
    for filename in needs:
        if not (OUT / filename).exists():
            raise FileNotFoundError(f'Missing results/{filename}; run Stage 4 first')
    spec = json.loads((OUT/'stage3_selected.json').read_text(encoding='utf-8'))
    config = json.loads((OUT/'stage4_manual_run_config.json').read_text(encoding='utf-8'))
    manual_metrics = json.loads((OUT/'stage4_manual_validation.json').read_text(encoding='utf-8'))
    assert [int(spec['global_top']), int(spec['user_recent']),
            int(spec['global_window_days']), int(spec['user_window_days'])] == [1000,400,30,60]
    assert config['train_cutoffs'] == base.TRAIN_TIMES
    assert config['validation_cutoff'] == base.VAL_TIME
    manual_features = config['features']
    assert len(manual_features) == 28
    assert all(f in manual_features for f in COMMON)
    specs = spec_catalog()
    auto_names = [x['name'] for x in specs]
    (OUT / 'stage5_generated_feature_specs.json').write_text(
        json.dumps(specs, indent=2), encoding='utf-8'
    )

    events, users = base.prepare_events()  # already filtered to < validation cutoff+7 days
    artists = base.load_artist_mapping(events)
    start_manual, manual_sec, auto_sec = time.perf_counter(), 0., 0.
    train_parts = []
    val_truth = None
    val = None
    for name, t in [(f'train_{i}', time_) for i, time_ in enumerate(base.TRAIN_TIMES, start=1)] + [
        ('validation', base.VAL_TIME)
    ]:
        tick = time.perf_counter()
        frame, truth = base.snapshot_features(events, users, artists, t, spec, name)
        manual_sec += time.perf_counter() - tick
        frame, sec = generate_aggregates(frame, events, artists, t, specs)
        auto_sec += sec
        print(f'{name}: generated {len(auto_names)} automatic features in {sec:.1f}s', flush=True)
        if name == 'validation':
            val = frame
            val_truth = truth
        else:
            train_parts.append(frame)
    del events, artists
    gc.collect()
    train = pd.concat(train_parts, ignore_index=True)
    del train_parts
    gc.collect()
    print(f'Total manual feature calculation: {manual_sec:.1f}s; '
          f'additional auto feature calculation: {auto_sec:.1f}s')
    assert val is not None and val_truth is not None
    assert len(train) == manual_metrics['train_candidate_rows']
    assert len(val) == manual_metrics['validation_candidate_rows']
    assert int(train.label.sum()) == manual_metrics['train_positive_candidate_rows']
    assert int(val.label.sum()) == manual_metrics['validation_positive_candidate_rows']
    verify_frozen_pairs(train, val)
    assert all(f in train for f in auto_names + manual_features)

    # Train-only selector: importance learned without ANY validation labels.
    selector_features = COMMON + auto_names
    selector = get_model(iterations=200, depth=5)
    select_start = time.perf_counter()
    selector.fit(train[selector_features].astype('float32'), train.label.to_numpy(dtype='uint8'))
    select_seconds = time.perf_counter() - select_start
    importance = pd.DataFrame({
        'feature': selector_features,
        'importance': selector.feature_importances_,
    }).sort_values(['importance', 'feature'], ascending=[False, True])
    importance.to_csv(OUT / 'stage5_selector_importances.csv', index=False)
    choices = importance.loc[importance.feature.isin(auto_names), 'feature'].head(NUMBER_SELECTED).tolist()
    assert len(choices) == NUMBER_SELECTED
    del selector
    gc.collect()
    auto_features = COMMON + choices
    novel_features = [name for name in choices if duplicate_manual_equivalent(name) not in manual_features]
    combined_features = manual_features + novel_features
    print(f'Train-only selection: picked {len(choices)} of {len(auto_names)} generated features; '
          f'Auto model uses {len(auto_features)} features; '
          f'Combined uses {len(combined_features)} features ({len(novel_features)} new).')
    print('Selected automatic features:', choices)
    print('Features skipped from combined due to exact manual equivalent:',
          len(choices) - len(novel_features))

    (OUT / 'stage5_selected_features.json').write_text(json.dumps({
        'selector': 'CatBoost feature importance on pooled TRAIN only',
        'selector_iterations': 200, 'selector_depth': 5,
        'selector_seconds': round(select_seconds, 2),
        'selection_validation_labels_accessed': False,
        'common_candidate_features': COMMON,
        'generated_feature_pool_count': len(auto_names),
        'number_selected_generated': len(choices),
        'auto_generated_features_selected': choices,
        'combined_extra_generated_features': novel_features,
        'manual_feature_count': len(manual_features),
        'auto_model_feature_count': len(auto_features),
        'combined_model_feature_count': len(combined_features),
    }, indent=2), encoding='utf-8')

    print('\n=== Automatic CatBoost ===', flush=True)
    auto_metrics, auto_scores = evaluate_model('auto', auto_features, train, val, val_truth)
    print('\n=== Manual + Auto CatBoost ===', flush=True)
    combined_metrics, combined_scores = evaluate_model('combined', combined_features, train, val, val_truth)
    # Scores are preserved by row index and same user-item candidate order.
    assert np.array_equal(auto_scores[['uid','item_id']].to_numpy(),
                          combined_scores[['uid','item_id']].to_numpy())
    scores_out = auto_scores.copy()
    scores_out['combined_score'] = combined_scores.combined_score.to_numpy()
    scores_out.to_parquet(OUT / 'stage5_validation_scores.parquet', index=False)

    comparison = pd.DataFrame([
        {'model':'Manual (stage4)',
         'ndcg_at_10_macro':manual_metrics['ndcg_at_10_macro'],
         'recall_at_10_macro':manual_metrics['recall_at_10_macro'],
         'logloss_on_candidates':manual_metrics['validation_logloss_on_candidates'],
         'roc_auc_on_candidates':manual_metrics['validation_roc_auc_on_candidates'],
         'feature_count':manual_metrics['feature_count'],
         'training_seconds':manual_metrics['training_seconds']},
        *[{'model':name,
           'ndcg_at_10_macro':m['ndcg_at_10_macro'],
           'recall_at_10_macro':m['recall_at_10_macro'],
           'logloss_on_candidates':m['validation_logloss_on_candidates'],
           'roc_auc_on_candidates':m['validation_roc_auc_on_candidates'],
           'feature_count':m['feature_count'],
           'training_seconds':m['training_seconds']}
          for name, m in [('Auto (stage5)', auto_metrics), ('Combined (stage5)', combined_metrics)]]
    ])
    comparison.to_csv(OUT / 'stage5_comparison.csv', index=False)
    (OUT / 'stage5_validation.json').write_text(json.dumps({
        'manual':manual_metrics,
        'auto':auto_metrics,
        'combined':combined_metrics,
        'selection_seconds':round(select_seconds, 2),
        'manual_feature_generation_seconds_this_run':round(manual_sec, 2),
        'automatic_feature_generation_seconds':round(auto_sec, 2),
        'candidate_pairs_verified_identical_to_stage4':True,
        'validation_only':True,
        'test_labels_accessed':False,
    }, indent=2), encoding='utf-8')
    print('\n=== STAGE 5 VALIDATION COMPARISON (test is untouched) ===')
    print(comparison.to_string(index=False, float_format=lambda x:f'{x:.5f}'))
    print('\nSaved stage5 model files, feature catalog, selected features, comparison, and scores to', OUT)
    print('TEST UNTOUCHED.')


if __name__ == '__main__':
    main()
