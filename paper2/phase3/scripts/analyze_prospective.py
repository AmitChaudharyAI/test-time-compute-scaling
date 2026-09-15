#!/usr/bin/env python3
"""Frozen Phase 3 replay and preregistered paired analysis; no inference/tuning."""
import csv
import hashlib
import json
import math
import fcntl
import sys
from collections import Counter
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'scripts'))
import generate_fresh_panel as gen


def csvout(name, rows):
    with (P / 'results' / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def prefix(rows):
    valid = [r for r in rows if r['vote_key'] is not None]
    counts = Counter(r['vote_key'] for r in valid)
    ranked = sorted(counts.values(), reverse=True)
    top = ranked[0] if ranked else 0
    tied = {k for k, v in counts.items() if v == top}
    winner = next((r for r in valid if r['vote_key'] in tied), None)
    return dict(samples=len(rows), tokens=sum(r['output_tokens'] for r in rows),
                winner_sample=winner['sample_index'] if winner else None,
                answer=winner['raw_extracted_answer'] if winner else None,
                key=winner['vote_key'] if winner else None,
                margin=(top-(ranked[1] if len(ranked)>1 else 0))/len(valid) if valid else None,
                invalid=len(rows)-len(valid), tie=len(tied)>1, valid=len(valid))


def adaptive(rows, q, decisions=None):
    # Each stopping decision is made before evaluation/future prefixes exist.
    for t in range(2, 17):
        x = prefix(rows[:t])
        stop = t == 16 or (x['margin'] is not None and x['margin'] >= .50)
        if decisions is not None:
            decisions.append(dict(question_index=q, checkpoint=t, valid_votes=x['valid'],
                                  margin=x['margin'], stop=stop, forced=t == 16,
                                  cumulative_output_tokens=x['tokens']))
        if stop:
            return x


def ci(a):
    return np.percentile(a, [2.5, 97.5]).tolist()


def exact_p(b, c):
    n = b+c
    return min(1., 2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.


def main():
    gen.verify_freeze()
    initial_raw_hash = hashlib.sha256(gen.RAW_PATH.read_bytes()).hexdigest()
    completion = json.loads((P/'recovery_completion_audit.json').read_text())
    assert completion['status'] == 'PASS' and initial_raw_hash == completion['raw_sha256']
    assert initial_raw_hash == (P/'raw_generations.sha256').read_text().split()[0]
    keys = gen.load_completed()  # Original strict extraction/scoring/seed validation.
    expected = {(q,s) for q in range(500) for s in range(1,17)}
    if keys != expected:
        raise RuntimeError(f'Incomplete panel: {len(keys)} pairs; missing {len(expected-keys)}')
    rows = [json.loads(line) for line in gen.RAW_PATH.read_text().splitlines()]
    assert len(rows) == 8000
    dataset = gen.load_dataset(gen.DATASET_ID, revision=gen.DATASET_REVISION, split='test', cache_dir=str(P/'cache/datasets'))
    panel = {q: sorted([r for r in rows if r['question_index']==q], key=lambda r:r['sample_index']) for q in range(500)}
    for q, rr in panel.items():
        for r in rr:
            assert r['question']==dataset[q]['problem'] and r['reference_answer']==dataset[q]['answer']
            assert r['normalized_answer']==gen.normalize_answer(r['raw_extracted_answer'])
            assert r['batch_size'] in range(1,65) and r['batch_wall_latency_seconds'] >= 0
            assert abs(r['latency_seconds']-r['batch_wall_latency_seconds']/r['batch_size']) < 1.1e-6
    ordered_seeds=[r['seed'] for q in range(500) for r in panel[q]]
    seed_hash=hashlib.sha256(json.dumps(ordered_seeds,separators=(',',':')).encode()).hexdigest()
    audit=json.loads((P/'seed_audit.json').read_text())
    assert seed_hash==audit['ordered_phase3_seed_list_sha256']
    assert min(ordered_seeds)==audit['phase3']['minimum'] and max(ordered_seeds)==audit['phase3']['maximum']
    (P/'results').mkdir(exist_ok=True); (P/'figures').mkdir(exist_ok=True)
    integrity=dict(status='PASS',records=len(rows),unique_pairs=len(keys),missing_pairs=0,duplicate_pairs=0,
                   questions=500,samples_per_question=16,all_seeds_match_seed_audit=True,ordered_seed_sha256=seed_hash,
                   historical_seed_overlap=0,raw_sha256=hashlib.sha256(gen.RAW_PATH.read_bytes()).hexdigest(),
                   identical_text_excess_records=sum(v-1 for v in Counter(r['raw_generation'] for r in rows).values()),
                   full_experimental_output_tokens=sum(r['output_tokens'] for r in rows),
                   invalid_extractions=sum(r['vote_key'] is None for r in rows))
    (P/'results/generation_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    decisions=[]
    # Replay receives no references, correctness labels, or future rows.
    voting_panel = {q: [{k:r[k] for k in ('sample_index','vote_key','raw_extracted_answer','output_tokens')} for r in panel[q]] for q in range(500)}
    methods={'adaptive':[adaptive(voting_panel[q],q,decisions) for q in range(500)]}
    for n in (1,2,4,8,16):
        methods[f'fixed_n{n}']=[prefix(voting_panel[q][:n]) for q in range(500)]
    # Score only after each method has frozen its selected original answer.
    for xx in methods.values():
        for q,x in enumerate(xx):
            x['correct'] = bool(x['winner_sample'] is not None and gen.math_equal(x['answer'],dataset[q]['answer']))
    csvout('checkpoint_decisions.csv',decisions)
    csvout('question_results.csv',[dict(question_index=q,method=m,**x) for m,xx in methods.items() for q,x in enumerate(xx)])
    rng=np.random.default_rng(40260915)
    indices=rng.integers(0,500,size=(10000,500))
    boot={m:np.array([x['correct'] for x in xx],dtype=float)[indices].mean(axis=1) for m,xx in methods.items()}
    n16_tokens=sum(x['tokens'] for x in methods['fixed_n16']); n8_tokens=sum(x['tokens'] for x in methods['fixed_n8'])
    metrics=[]
    for m,xx in methods.items():
        tok=sum(x['tokens'] for x in xx); sam=sum(x['samples'] for x in xx)
        lo,hi=ci(boot[m])
        metrics.append(dict(method=m,scope='prospective primary' if m=='adaptive' else 'fixed-budget baseline',
                            correct_count=sum(x['correct'] for x in xx),accuracy=sum(x['correct'] for x in xx)/500,
                            accuracy_ci_low=lo,accuracy_ci_high=hi,total_samples=sam,mean_samples=sam/500,
                            median_samples=float(np.median([x['samples'] for x in xx])),total_output_tokens=tok,mean_output_tokens=tok/500,
                            sample_reduction_vs_n16_pct=100*(1-sam/8000),token_reduction_vs_n16_pct=100*(1-tok/n16_tokens),
                            token_reduction_vs_n8_pct=100*(1-tok/n8_tokens),invalid_extractions=sum(x['invalid'] for x in xx),
                            ties=sum(x['tie'] for x in xx),all_invalid_questions=sum(x['valid']==0 for x in xx)))
    csvout('method_metrics.csv',metrics)
    comparisons=[]
    aa=methods['adaptive']
    for n in (16,8,4):
        bb=methods[f'fixed_n{n}']; b=sum(a['correct'] and not b['correct'] for a,b in zip(aa,bb)); c=sum(b['correct'] and not a['correct'] for a,b in zip(aa,bb))
        lo,hi=ci(100*(boot['adaptive']-boot[f'fixed_n{n}']))
        comparisons.append(dict(comparison=f'adaptive_minus_fixed_n{n}',scope='primary confirmatory' if n==16 else 'secondary exploratory unadjusted',
                                accuracy_difference_pp=100*(b-c)/500,ci_low_pp=lo,ci_high_pp=hi,
                                both_correct=sum(a['correct'] and b['correct'] for a,b in zip(aa,bb)),adaptive_only=b,fixed_only=c,
                                both_wrong=sum(not a['correct'] and not b['correct'] for a,b in zip(aa,bb)),
                                exact_two_sided_mcnemar_p=exact_p(b,c),alpha=.05))
    csvout('paired_comparisons.csv',comparisons)
    csvout('generation_descriptives.csv',[dict(sample_index=t,records=500,invalid_extractions=sum(panel[q][t-1]['vote_key'] is None for q in range(500)),token_limit_records=sum(panel[q][t-1]['output_tokens']==1024 for q in range(500)),mean_output_tokens=sum(panel[q][t-1]['output_tokens'] for q in range(500))/500) for t in range(1,17)])
    stops=[]
    for t in range(2,17):
        xx=[x for x in aa if x['samples']==t]
        stops.append(dict(stopping_sample=t,count=len(xx),fraction=len(xx)/500,
                          conditional_accuracy=sum(x['correct'] for x in xx)/len(xx) if xx else None,
                          mean_output_tokens=np.mean([x['tokens'] for x in xx]).item() if xx else None,
                          mean_margin=np.mean([x['margin'] for x in xx if x['margin'] is not None]).item() if any(x['margin'] is not None for x in xx) else None))
    csvout('stopping_distribution.csv',stops)
    errors=[]
    for q,(a,b) in enumerate(zip(aa,methods['fixed_n16'])):
        trajectory=[prefix(panel[q][:t]) for t in range(1,17)]
        errors.append(dict(question_index=q,paired_category=('both_correct' if a['correct'] and b['correct'] else 'adaptive_only' if a['correct'] else 'fixed_only' if b['correct'] else 'both_wrong'),
                           premature_stop_paired_recovery=not a['correct'] and a['samples']<16 and b['correct'],
                           stable_wrong_consensus_descriptive=not a['correct'] and a['key'] is not None and all(x['key']==a['key'] for x in trajectory[a['samples']-1:]),
                           invalid_influenced_descriptive=a['invalid']>0,
                           unstable_trajectory_descriptive=len({x['key'] for x in trajectory if x['key'] is not None})>1,
                           stopping_sample=a['samples']))
    csvout('error_categories.csv',errors)
    historical=[json.loads(line) for line in (gen.ROOT/'results/raw/generations.jsonl').read_text().splitlines()]
    oldpanel={q:sorted([r for r in historical if r['question_index']==q],key=lambda r:r['sample_index']) for q in range(500)}
    assert len(historical)==8000
    assert {(r['question_index'],r['sample_index']) for r in historical}==expected
    assert all(r['seed']==20260902+r['question_index']*16+r['sample_index']-1 for r in historical)
    oldvotes={q:[{k:r[k] for k in ('sample_index','vote_key','raw_extracted_answer','output_tokens')} for r in oldpanel[q]] for q in range(500)}
    old=[adaptive(oldvotes[q],q) for q in range(500)]
    old16=[prefix(oldvotes[q]) for q in range(500)]
    for xx in (old,old16):
        for q,x in enumerate(xx):
            x['correct']=bool(x['winner_sample'] is not None and gen.math_equal(x['answer'],oldpanel[q][0]['reference_answer']))
    offline=dict(scope='secondary descriptive; separate historical panel',accuracy=sum(x['correct'] for x in old)/500,
                 fixed_n16_accuracy=sum(x['correct'] for x in old16)/500,mean_samples=sum(x['samples'] for x in old)/500,
                 token_reduction_vs_n16_pct=100*(1-sum(x['tokens'] for x in old)/sum(x['tokens'] for x in old16)),
                 stopping_counts={str(t):sum(x['samples']==t for x in old) for t in range(2,17)})
    attempts=[json.loads(x) for x in (P/'generation_attempts.jsonl').read_text().splitlines()]
    failures=[json.loads(x) for x in (P/'failed_attempts.jsonl').read_text().splitlines()] if (P/'failed_attempts.jsonl').exists() else []
    latency=dict(started_attempts=len(attempts),logged_failures=failures,
                 completed_batch_wall_seconds=sum(r['batch_wall_latency_seconds']/r['batch_size'] for r in rows),
                 allocated_row_latency_seconds=sum(r['latency_seconds'] for r in rows),
                 interpretation='Amortized batch metadata; not isolated latency or online adaptive speedup')
    result=dict(latency_and_failures=latency,integrity=integrity,analysis_seed=40260915,bootstrap_resamples=10000,metrics=metrics,
                paired_comparisons=comparisons,stopping_distribution=stops,historical_separate_panel=offline,
                descriptive_shortfall_bands=[dict(band_pp=v,observed_difference_within_band=comparisons[0]['accuracy_difference_pp']>=-v,
                    token_reduction_pct=metrics[0]['token_reduction_vs_n16_pct'],interpretation='descriptive, not non-inferiority') for v in (.5,1.,2.)],
                maximum_budget_fraction=stops[-1]['fraction'],error_summary={k:sum(x[k] for x in errors) for k in ('premature_stop_paired_recovery','stable_wrong_consensus_descriptive','invalid_influenced_descriptive','unstable_trajectory_descriptive')})
    (P/'results/analysis_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    from publish_prospective import publish, analysis_audit
    publish(result, methods, rows, errors)
    analysis_audit(result, methods, panel, decisions, initial_raw_hash)
    print(json.dumps({'metrics':metrics,'primary':comparisons[0],'integrity':'PASS'},indent=2))


if __name__=='__main__':
    with (P/'generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
