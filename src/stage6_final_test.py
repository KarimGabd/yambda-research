"""Stage 6: ONE final Yambda test evaluation with frozen Stage-4/Stage-5 models.

Put this file at yambda-research/src/stage6_final_test.py. Run once AFTER
Stage 4 and Stage 5 have been finalized, and do not retune using test results.

Outputs in results/:
  stage6_test_comparison.csv
  stage6_test_metrics.json
  stage6_test_scores.parquet
  stage6_per_user_metrics.csv

No training or feature selection happens here. The 7-day test labels are
accessed for the FIRST TIME by this script, only for final scoring.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

import stage4_manual_catboost as base
import stage5_auto_features as auto

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
TEST_TIME = base.END - base.HORIZON
METRICS_FILE = OUT / 'stage6_test_metrics.json'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_setup():
    """Check *all* files / feature choices before we expose test labels."""
    model_files = {
        'manual': OUT / 'stage4_manual_model.cbm',
        'auto': OUT / 'stage5_auto_model.cbm',
        'combined': OUT / 'stage5_combined_model.cbm',
    }
    required = [
        base.EVENTS_PATH, base.UIDS_PATH, base.ARTIST_PATH,
        OUT / 'stage3_selected.json',
        OUT / 'stage4_manual_run_config.json',
        OUT / 'stage5_selected_features.json',
        OUT / 'stage5_validation.json',
        *model_files.values(),
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError('Missing frozen input: ' + str(path))
    for output in (
        METRICS_FILE,
        OUT / 'stage6_test_comparison.csv',
        OUT / 'stage6_test_scores.parquet',
        OUT / 'stage6_per_user_metrics.csv',
    ):
        if output.exists():
            raise FileExistsError(
                f'{output.name} already exists. Do not rerun/reselect on test.'
            )

    spec = json.loads((OUT / 'stage3_selected.json').read_text(encoding='utf-8'))
    assert [int(spec[k]) for k in (
        'global_top', 'user_recent', 'global_window_days', 'user_window_days'
    )] == [1000, 400, 30, 60]

    previous = json.loads((OUT / 'stage4_manual_run_config.json').read_text(encoding='utf-8'))
    selected = json.loads((OUT / 'stage5_selected_features.json').read_text(encoding='utf-8'))
    val_report = json.loads((OUT / 'stage5_validation.json').read_text(encoding='utf-8'))
    assert val_report['test_labels_accessed'] is False
    assert previous['validation_cutoff'] == base.VAL_TIME
    assert previous['train_cutoffs'] == base.TRAIN_TIMES
    assert base.VAL_TIME + base.HORIZON < TEST_TIME, 'Validation/test windows overlap'
    assert len(auto.spec_catalog()) == 72

    manual_features = previous['features']
    selection = selected['auto_generated_features_selected']
    novel = selected['combined_extra_generated_features']
    assert len(manual_features) == 28
    assert len(selection) == 24 and len(set(selection)) == 24
    assert len(novel) == 12 and len(set(novel)) == 12
    assert set(novel).issubset(set(selection))
    assert selected['selection_validation_labels_accessed'] is False

    features = {
        'manual': manual_features,
        'auto': auto.COMMON + selection,
        'combined': manual_features + novel,
    }
    assert [len(features[k]) for k in ('manual', 'auto', 'combined')] == [28, 28, 40]
    assert all(len(v) == len(set(v)) for v in features.values())

    models = {}
    for name, path in model_files.items():
        m = CatBoostClassifier()
        m.load_model(str(path))
        assert list(m.feature_names_) == features[name], (
            f'{name}: saved model feature names/order do not match frozen feature list'
        )
        models[name] = m

    audit_files = [
        OUT / 'stage3_selected.json',
        OUT / 'stage4_manual_run_config.json',
        OUT / 'stage5_selected_features.json',
        *model_files.values(),
    ]
    hashes = {p.name: sha256(p) for p in audit_files}
    return spec, features, models, hashes


def load_complete_events():
    """Unlike Stage 4, read the LAST 7 days so we can evaluate true test labels."""
    columns = ['uid', 'item_id', 'timestamp', 'event_type', 'played_ratio_pct', 'is_organic']
    events = pd.read_parquet(base.EVENTS_PATH, columns=columns)
    if len(events) == 0:
        raise ValueError('Empty events sample')
    assert events.timestamp.min() >= 0
    assert events.timestamp.max() < base.END
    events['positive'] = events.event_type.eq('listen') & events.played_ratio_pct.gt(50)
    events['uid'] = events.uid.astype('uint32')
    events['item_id'] = events.item_id.astype('uint32')
    users = np.sort(np.loadtxt(base.UIDS_PATH, dtype='uint32', ndmin=1))
    assert len(users) == len(np.unique(users)) == 300
    assert set(events.uid.unique()).issubset(set(users))
    assert int((events.timestamp >= TEST_TIME).sum()) > 0, 'No test-period events!'
    return events, users


def per_user_metrics(scored, labels, score_column, name):
    """Exactly the same user-level DCG/Recall definitions as base.evaluation."""
    truth = {int(u): set(g.item_id.astype(int)) for u, g in labels.groupby('uid')}
    top = scored.sort_values(
        ['uid', score_column, 'item_id'],
        ascending=[True, False, True], kind='stable'
    ).groupby('uid', sort=False).head(base.TOP_K)
    ranked = {int(u): g.item_id.astype(int).tolist() for u, g in top.groupby('uid')}
    rows = []
    for user, relevant in truth.items():
        choices = ranked.get(user, [])
        hits = len(set(choices) & relevant)
        dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(choices) if item in relevant)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(base.TOP_K, len(relevant))))
        rows.append({
            'uid': user, 'model': name, 'positive_items': len(relevant),
            'recall_at_10': hits / len(relevant), 'ndcg_at_10': dcg / idcg,
        })
    return pd.DataFrame(rows)


def paired_bootstrap(per_user, a, b, n_boot=2000):
    """95% paired user bootstrap CI for NDCG(b) - NDCG(a). Descriptive only."""
    wide = per_user.pivot(index='uid', columns='model', values='ndcg_at_10')
    diff = (wide[b] - wide[a]).dropna().to_numpy(dtype='float64')
    rng = np.random.default_rng(base.SEED)
    samples = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    estimates = diff[samples].mean(axis=1)
    return {
        'model_b_minus_a': f'{b} - {a}',
        'mean_paired_difference': float(diff.mean()),
        'bootstrap_95pct_low': float(np.quantile(estimates, 0.025)),
        'bootstrap_95pct_high': float(np.quantile(estimates, 0.975)),
        'resampled_users': len(diff),
        'bootstrap_resamples': n_boot,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec, features, models, hashes = frozen_setup()
    print('FROZEN models and feature selections verified BEFORE test labels are read.')
    print('Test cutoff:', TEST_TIME, '| Horizon seconds:', base.HORIZON)
    print('No model fitting or tuning will be performed.\n')

    events, users = load_complete_events()
    artists = base.load_artist_mapping(events)
    t0 = time.perf_counter()
    frame, truth = base.snapshot_features(events, users, artists, TEST_TIME, spec, 'test')
    manual_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    frame, additional_auto_time = auto.generate_aggregates(
        frame, events, artists, TEST_TIME, auto.spec_catalog()
    )
    auto_time = time.perf_counter() - t0
    print(f'Feature generation: manual={manual_time:.2f}s, auto-additional={auto_time:.2f}s')
    assert all(f in frame.columns for group in features.values() for f in group)
    assert not frame[['uid', 'item_id']].duplicated().any()
    assert frame['snapshot'].eq('test').all()
    expected_truth = events.loc[
        events.timestamp.ge(TEST_TIME) & events.timestamp.lt(base.END) & events.positive,
        ['uid', 'item_id']
    ].drop_duplicates()
    assert len(expected_truth) == len(truth)
    positives_in_candidates = int(frame['label'].sum())
    assert positives_in_candidates == len(
        frame.loc[frame.label.eq(1), ['uid', 'item_id']].drop_duplicates()
    )
    print('Test label pairs:', len(truth), 'of which in candidates:', positives_in_candidates)
    past_catalog = pd.Index(events.loc[events.timestamp < TEST_TIME, 'item_id'].unique())
    unseen_fraction = float((~truth.item_id.isin(past_catalog)).mean())

    metrics = {}
    per_user = []
    scored = frame[['uid', 'item_id', 'label']].copy()
    for name in ('manual', 'auto', 'combined'):
        t0 = time.perf_counter()
        x = frame[features[name]].astype('float32')
        assert np.isfinite(x.to_numpy()).all(), f'{name}: non-finite features'
        y = frame.label.to_numpy(dtype='uint8')
        probability = models[name].predict_proba(x)[:, 1]
        inference_seconds = time.perf_counter() - t0
        scores = frame[['uid', 'item_id']].copy()
        scores['score'] = probability
        result = base.evaluation(scores, truth, 'score')
        result.update({
            'test_logloss_on_candidates': float(log_loss(y, probability, labels=[0, 1])),
            'test_roc_auc_on_candidates': float(roc_auc_score(y, probability)),
            'feature_count': len(features[name]),
            'inference_seconds': round(inference_seconds, 2),
        })
        metrics[name] = result
        user_df = per_user_metrics(scores, truth, 'score', name)
        assert abs(user_df.ndcg_at_10.mean() - result['ndcg_at_10_macro']) < 1e-12
        assert abs(user_df.recall_at_10.mean() - result['recall_at_10_macro']) < 1e-12
        per_user.append(user_df)
        scored[f'{name}_score'] = probability.astype('float32')
        print(f'{name}: NDCG@10={result["ndcg_at_10_macro"]:.5f} | '
              f'Recall@10={result["recall_at_10_macro"]:.5f} | '
              f'LogLoss={result["test_logloss_on_candidates"]:.5f}')

    scored['popularity_score'] = np.where(frame.global_rank > 0, -frame.global_rank, -1e9).astype('float32')
    metrics['popularity'] = base.evaluation(scored, truth, 'popularity_score')
    for other in ('auto', 'combined'):
        for k in ('candidate_recall_micro', 'candidate_recall_macro'):
            assert abs(metrics[other][k] - metrics['manual'][k]) < 1e-12

    user_metrics = pd.concat(per_user, ignore_index=True)
    ci = {
        'auto_minus_manual': paired_bootstrap(user_metrics, 'manual', 'auto'),
        'combined_minus_manual': paired_bootstrap(user_metrics, 'manual', 'combined'),
    }
    comparison = pd.DataFrame([
        dict(model=name,
             ndcg_at_10_macro=m['ndcg_at_10_macro'],
             recall_at_10_macro=m['recall_at_10_macro'],
             candidate_recall_macro=m['candidate_recall_macro'],
             test_logloss_on_candidates=m.get('test_logloss_on_candidates'),
             test_roc_auc_on_candidates=m.get('test_roc_auc_on_candidates'),
             feature_count=m.get('feature_count'),
             inference_seconds=m.get('inference_seconds'))
        for name, m in metrics.items()
    ])
    summary = {
        'stage': 'final_test_once',
        'test_cutoff': TEST_TIME,
        'label_window_end_exclusive': base.END,
        'test_window_days': 7,
        'candidate_spec': {k: spec[k] for k in ('global_top', 'user_recent', 'global_window_days', 'user_window_days')},
        'test_new_observed_catalog_item_fraction': unseen_fraction,
        'test_candidate_rows': len(frame),
        'test_positive_pairs': len(truth),
        'test_positive_candidate_rows': positives_in_candidates,
        'metrics': metrics,
        'paired_user_bootstrap_ndcg_differences': ci,
        'time_seconds': {'manual_features': round(manual_time, 2), 'auto_additional': round(auto_time, 2)},
        'frozen_inputs_sha256': hashes,
        'versions': {'python': platform.python_version(), 'pandas': pd.__version__,
                     'numpy': np.__version__, 'catboost': catboost.__version__},
        'method_note': (
            'Saved Stage 4/5 models evaluated without retraining on the SAME test candidates. '
            'Ranking metrics over users with at least one future positive, denominators include '
            'all future positives. AUC/LogLoss conditional on candidates. Paired bootstrap '
            'over evaluated users is descriptive, not a population generalization guarantee.'
        ),
    }
    # Write the summary JSON LAST: its existence is our guard against repeat evaluation.
    scored.to_parquet(OUT / 'stage6_test_scores.parquet', index=False)
    user_metrics.to_csv(OUT / 'stage6_per_user_metrics.csv', index=False)
    comparison.to_csv(OUT / 'stage6_test_comparison.csv', index=False)
    METRICS_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print('\n=== STAGE 6 FINAL TEST (NO RETUNING) ===')
    print(comparison.to_string(index=False, float_format=lambda n: f'{n:.5f}'))
    print('\n=== PAIRED NDCG@10 USER BOOTSTRAP ===')
    print(json.dumps(ci, indent=2))
    print('\nSaved test results to:', OUT)
    print('Test is now used. DO NOT change feature selection or hyperparameters based on it.')


if __name__ == '__main__':
    main()
