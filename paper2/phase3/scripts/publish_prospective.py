"""Reporting and independent checks; no parameter selection or inference."""
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import numpy as np
import generate_fresh_panel as gen
P=gen.PHASE3
R=P/'results'


def writecsv(name, rows):
 with (R/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def table(rows, columns):
 return '| '+' | '.join(label for key,label,fmt in columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'+'\n'.join('| '+' | '.join('—' if row[key] is None else format(row[key],fmt) if fmt else str(row[key]) for key,label,fmt in columns)+' |' for row in rows)


def publish(r,methods,rows,errors):
 metrics=r['metrics']; a=metrics[0]; n16=metrics[-1]
 for x in metrics:
  x['invalid_extraction_rate']=x['invalid_extractions']/x['total_samples']
  x['tie_rate']=x['ties']/500
  x['all_invalid_rate']=x['all_invalid_questions']/500
 writecsv('prospective_summary.csv',metrics)
 writecsv('method_metrics.csv',metrics)
 writecsv('fixed_budget_baselines.csv',metrics[1:])
 for c in r['paired_comparisons']:
  n=int(c['comparison'].split('fixed_n')[1]); b=next(x for x in metrics if x['method']==f'fixed_n{n}')
  c.update(adaptive_accuracy=a['accuracy'],fixed_accuracy=b['accuracy'],
   discordant_pairs=c['adaptive_only']+c['fixed_only'],test='exact two-sided binomial McNemar; null p=0.5',
   binomial_statistic_adaptive_only=c['adaptive_only'],mean_sample_reduction=b['mean_samples']-a['mean_samples'],
   total_sample_reduction=b['total_samples']-a['total_samples'],sample_reduction_pct=100*(1-a['total_samples']/b['total_samples']),
   mean_output_token_reduction=b['mean_output_tokens']-a['mean_output_tokens'],
   total_output_token_reduction=b['total_output_tokens']-a['total_output_tokens'],token_reduction_pct=100*(1-a['total_output_tokens']/b['total_output_tokens']))
 writecsv('paired_comparison.csv',r['paired_comparisons'])
 writecsv('paired_comparisons.csv',r['paired_comparisons'])
 for s in r['stopping_distribution']:
  xx=[x for x in methods['adaptive'] if x['samples']==s['stopping_sample']]
  s.update(percentage=100*s['fraction'],correct_count=sum(x['correct'] for x in xx),total_output_tokens=sum(x['tokens'] for x in xx),
   invalid_extractions=sum(x['invalid'] for x in xx),tie_count=sum(x['tie'] for x in xx))
 writecsv('stopping_distribution.csv',r['stopping_distribution'])
 old=r['historical_separate_panel']
 writecsv('historical_vs_fresh.csv',[
  dict(panel='historical offline',scope='secondary descriptive separate panel',adaptive_accuracy=old['accuracy'],fixed_n16_accuracy=old['fixed_n16_accuracy'],mean_samples=old['mean_samples'],token_reduction_vs_n16_pct=old['token_reduction_vs_n16_pct']),
  dict(panel='fresh prospective',scope='prospective panel',adaptive_accuracy=a['accuracy'],fixed_n16_accuracy=n16['accuracy'],mean_samples=a['mean_samples'],token_reduction_vs_n16_pct=a['token_reduction_vs_n16_pct'])])
 writecsv('historical_vs_fresh_stopping.csv',[dict(stopping_sample=t,historical_count=old['stopping_counts'][str(t)],fresh_count=r['stopping_distribution'][t-2]['count']) for t in range(2,17)])
 writecsv('descriptive_shortfall_bands.csv',r['descriptive_shortfall_bands'])
 writecsv('paired_error_categories.csv',[dict(category=k,count=sum(e['paired_category']==k for e in errors),percentage=100*sum(e['paired_category']==k for e in errors)/500) for k in ('both_correct','adaptive_only','fixed_only','both_wrong')])
 manifest=json.loads((P/'pre_inference_manifest.json').read_text())
 recovery=json.loads((P/'recovery_audit.json').read_text())
 metadata=dict(completed_at=json.loads((P/'generation_state.json').read_text())['updated_at'],
  panel_path=str(gen.RAW_PATH.relative_to(gen.ROOT)),panel_sha256=r['integrity']['raw_sha256'],record_count=8000,unique_pairs=8000,malformed_records=0,duplicates=0,missing_pairs=0,
  model=gen.MODEL_ID,model_revision=gen.MODEL_REVISION,dataset=gen.DATASET_ID,dataset_revision=gen.DATASET_REVISION,
  question_count=500,samples_per_question=16,generation_seed_base=gen.BASE_SEED,ordered_seed_sha256=r['integrity']['ordered_seed_sha256'],
  historical_seed_overlap=0,decoding=gen.GENERATION,runtime_audit_sha256=manifest['sha256']['paper2/phase3/runtime_audit.json'],
  frozen_policy_sha256=manifest['sha256']['paper2/phase3/FROZEN_POLICY.md'],experiment_freeze_sha256=manifest['sha256']['paper2/phase3/EXPERIMENT_FREEZE.md'],
  full_panel_output_tokens=n16['total_output_tokens'],invalid_extractions=r['integrity']['invalid_extractions'],invalid_extraction_rate=r['integrity']['invalid_extractions']/8000,
  identical_text_excess_records=r['integrity']['identical_text_excess_records'],recovery_original_records=recovery['record_count'],recovery_added_records=8000-recovery['record_count'],
  interruption='Temporary instance interruption due to credit exhaustion, as reported by user; hard interruption has no Python exception record.',
  attempts_and_latency=r['latency_and_failures'],analysis_seed=40260915,bootstrap_resamples=10000,
  compute_interpretation='Adaptive cost is counterfactual prefix samples/tokens on the prospectively generated full panel; no online wall-clock saving measured.',
  scoring_environment_unchanged=True,figures_environment='Isolated /tmp/phase3-plot-env; used only for rendering saved results; not used for voting, scoring, bootstrap or policy.')
 (R/'final_generation_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
 c=r['paired_comparisons'][0]
 columns=[('method','Method',''),('correct_count','Correct / 500','d'),('accuracy','Accuracy','.2%'),('mean_samples','Mean samples','.3f'),('median_samples','Median samples','.1f'),('mean_output_tokens','Mean output tokens','.2f'),('total_output_tokens','Total output tokens',',d')]
 stopcols=[('stopping_sample','STOP sample','d'),('count','Questions','d'),('percentage','Percent','.1f'),('conditional_accuracy','Conditional accuracy','.2%'),('mean_output_tokens','Mean output tokens','.2f'),('total_output_tokens','Total tokens',',d'),('mean_margin','Mean margin','.3f')]
 paircols=[('comparison','Comparison',''),('scope','Scope',''),('accuracy_difference_pp','Difference (pp)','.2f'),('ci_low_pp','95% CI low (pp)','.2f'),('ci_high_pp','95% CI high (pp)','.2f'),('adaptive_only','Adaptive only','d'),('fixed_only','Fixed only','d'),('exact_two_sided_mcnemar_p','Exact p-value','.6g')]
 stats=f'''# Preregistered prospective statistical analysis

All 500 questions and all methods use the same completed generation panel. No exclusions or outcome-based termination occurred. The sole primary comparison is adaptive minus fixed N=16, with two-sided alpha=0.05.

## Exact paired primary test

Both correct: {c['both_correct']}; adaptive correct / N16 wrong: {c['adaptive_only']}; adaptive wrong / N16 correct: {c['fixed_only']}; both wrong: {c['both_wrong']}.

The preregistered exact two-sided binomial McNemar test conditions on {c['discordant_pairs']} discordant questions. Its binomial statistic is the adaptive-only count, k={c['adaptive_only']} of n={c['discordant_pairs']}, under null p=0.5. Exact p={c['exact_two_sided_mcnemar_p']:.10g}. No asymptotic chi-square statistic or continuity correction is used. If there were zero discordant pairs, p would be 1.

Adaptive minus N16 accuracy: **{c['accuracy_difference_pp']:.2f} percentage points**, paired percentile 95% bootstrap interval **[{c['ci_low_pp']:.2f}, {c['ci_high_pp']:.2f}] pp**.

## Frozen bootstrap and secondary tests

NumPy default_rng(40260915), 10,000 resamples of 500 question indices with replacement. Identical indices are reused across every method; confidence bounds are the 2.5th and 97.5th percentiles. Individual method accuracy intervals are in prospective_summary.csv. Paired difference intervals and the prespecified secondary N=8 and N=4 exact tests are below. Secondary p-values are exploratory and unadjusted; no familywise confirmatory claim is made.

{table(r['paired_comparisons'],paircols)}

There is no formal equivalence or non-inferiority test. Non-significance and overlapping intervals cannot demonstrate equivalence. The preregistered 0.5, 1.0 and 2.0 pp observed-shortfall bands are descriptive only and are not acceptance thresholds or policy-selection rules. Conditional stopping accuracy, error trajectories and historical-versus-fresh comparisons are secondary descriptive analyses. No new statistical tests were introduced.
'''
 (R/'statistical_analysis.md').write_text(stats)
 conclusion=('The frozen policy has lower observed accuracy than N=16' if c['accuracy_difference_pp']<0 else 'The frozen policy has higher observed accuracy than N=16' if c['accuracy_difference_pp']>0 else 'The frozen policy has the same observed accuracy as N=16')
 if c['exact_two_sided_mcnemar_p']<.05: conclusion+='; the preregistered primary exact paired test rejects equal discordant outcome probabilities at alpha=0.05.'
 else: conclusion+='; the preregistered primary exact paired test does not reject equal discordant outcome probabilities at alpha=0.05. This does not establish equivalence.'
 report=f'''# Phase 3 prospective results

The frozen **margin_min2_0.50** policy achieved **{a['accuracy']:.2%} ({a['correct_count']}/500)**, versus **{n16['accuracy']:.2%} ({n16['correct_count']}/500)** for fixed N=16. The paired difference was **{c['accuracy_difference_pp']:.2f} pp**, with preregistered paired 95% bootstrap CI **[{c['ci_low_pp']:.2f}, {c['ci_high_pp']:.2f}] pp** and exact two-sided McNemar **p={c['exact_two_sided_mcnemar_p']:.6g}**. {conclusion}

## Principal results from the same prospective panel

{table(metrics,columns)}

Individual accuracy confidence intervals, samples/tokens, invalid counts/rates and ties are in [prospective_summary.csv](prospective_summary.csv); fixed baselines are in [fixed_budget_baselines.csv](fixed_budget_baselines.csv).

Adaptive versus N16 saves **{c['mean_sample_reduction']:.3f} samples/question**, **{c['total_sample_reduction']:,} total samples ({c['sample_reduction_pct']:.2f}%)**, and **{c['mean_output_token_reduction']:,.2f} output tokens/question**, **{c['total_output_token_reduction']:,} total output tokens ({c['token_reduction_pct']:.2f}%)**. Adaptive token reduction versus N8 is **{a['token_reduction_vs_n8_pct']:.2f}%**. The observed accuracy–compute trade-off across every fixed budget and adaptive is plotted with paired-bootstrap individual accuracy intervals; connecting fixed-budget points is a visual guide, not interpolation or optimization.

## Paired comparison

{table(r['paired_comparisons'],paircols)}

Primary paired counts: both correct {c['both_correct']}; adaptive-only {c['adaptive_only']}; N16-only {c['fixed_only']}; both wrong {c['both_wrong']}. Full statistical definitions are in [statistical_analysis.md](statistical_analysis.md). Secondary N8/N4 tests are exploratory and unadjusted.

## Stopping behavior

{table(r['stopping_distribution'],stopcols)}

All eligible checkpoints 2 through 15 use V>0 and margin >=0.50; sample 16 forces termination. The maximum-budget fraction is **{r['maximum_budget_fraction']:.2%}**. Empty stopping groups have undefined conditional accuracy/cost/margin, shown as —. Per-question results and every visited checkpoint are saved for audit.

## Invalid extraction and ties

Full-panel invalid extraction rate: **{r['integrity']['invalid_extractions']}/8,000 ({r['integrity']['invalid_extractions']/8000:.2%})**. Adaptive-prefix invalid extraction rate: **{a['invalid_extractions']}/{a['total_samples']} ({a['invalid_extraction_rate']:.2%})**. Invalids consume samples/tokens and cast no vote; no replacement samples or repairs are made. Invalid-rate denominators are generated samples within each method's prefix.

Adaptive terminal top-tie rate: **{a['ties']}/500 ({a['tie_rate']:.2%})**. Fixed N16 terminal top-tie rate: **{n16['ties']}/500 ({n16['tie_rate']:.2%})**. A tie means at least two keys share the positive maximum vote count at the selected prefix; all-invalid is reported separately. Earliest valid sample among tied winners resolves the output. Method-specific ties and all-invalid counts are in the summary table CSV.

Identical response text excess records: **{r['integrity']['identical_text_excess_records']}**, defined as sum(count−1) across identical raw-text groups in the full panel. Distinct seeded trajectories with identical text remain in the analysis; duplicate pair records are forbidden.

## Preregistered secondary/descriptive analyses

Paired errors are reported first in paired_error_categories.csv. Operational premature-stop paired recoveries (adaptive wrong, STOP<16, N16 correct): **{r['error_summary']['premature_stop_paired_recovery']}**; this is not proof of causation. Stable wrong consensus: **{r['error_summary']['stable_wrong_consensus_descriptive']}** (wrong adaptive plurality persists from STOP through sample 16); invalid-influenced prefixes: **{r['error_summary']['invalid_influenced_descriptive']}** (at least one invalid before STOP); unstable trajectories: **{r['error_summary']['unstable_trajectory_descriptive']}** (more than one non-null plurality over samples 1..16). These overlapping descriptive categories never change predictions.

Historical offline adaptive accuracy **{old['accuracy']:.2%}**, fixed N16 accuracy **{old['fixed_n16_accuracy']:.2%}**, mean samples **{old['mean_samples']:.3f}**, token reduction versus its own N16 **{old['token_reduction_vs_n16_pct']:.2f}%**. Fresh adaptive accuracy **{a['accuracy']:.2%}**, mean samples **{a['mean_samples']:.3f}**, token reduction **{a['token_reduction_vs_n16_pct']:.2f}%**. Historical and fresh panels are reported separately, with stopping distributions in historical_vs_fresh_stopping.csv; no pooling or cross-panel inferential tests.

Observed shortfall-band descriptions (not non-inferiority tests): {', '.join(str(x['band_pp'])+' pp: '+str(x['observed_difference_within_band']) for x in r['descriptive_shortfall_bands'])}. Every band has the same observed {a['token_reduction_vs_n16_pct']:.2f}% token reduction; these bands do not select a new policy. No equivalence, optimality, or non-inferiority claim is made.

## Generation, compute and latency accounting

The full prospective experiment cost is **8,000 generations and {n16['total_output_tokens']:,} output tokens**. Adaptive's **{a['total_samples']:,} samples and {a['total_output_tokens']:,} tokens** are counterfactual stopping-prefix cost on the prospectively collected trajectories, not actual full-panel spending or measured online wall-clock savings. Latency is batch-wall-time/batch-size amortized metadata. The RTX 4090 and historical RTX 5070 Ti difference forbids cross-hardware latency speedup claims.

Started generation attempts: **{r['latency_and_failures']['started_attempts']}**; logged Python failures: **{len(r['latency_and_failures']['logged_failures'])}**. The user-reported credit interruption is preserved in recovery history; a hard interruption need not produce a Python exception record. Completed-batch wall time summed by allocation: **{r['latency_and_failures']['completed_batch_wall_seconds']:.2f} s**; interrupted attempt work is not included in this completed-batch latency total. Metadata is in final_generation_metadata.json.

## Integrity and recommendation

GENERATION INTEGRITY: PASS. FROZEN POLICY INTEGRITY: PASS. PROSPECTIVE ANALYSIS: PASS. These statuses are validated by analysis_integrity.json, including an independent vote/STOP replay, cost reconciliation, bootstrap/test checks, preserved raw hash and frozen artifact hashes. No model inference, policy tuning, source-scoring edits or manual answer repair occurred during evaluation.

**Recommendation B:** present the frozen policy as an observed accuracy–compute trade-off with its paired uncertainty and error counts; retain the frozen rule and its results. This is an editorial recommendation, not a data-selected threshold or redesigned policy. No online experiment was conducted.
'''
 (R/'PROSPECTIVE_RESULTS.md').write_text(report)
 # Root entry points preserve relative links by redirecting to the canonical report.
 (P/'PROSPECTIVE_RESULTS.md').write_text('# Phase 3 prospective results\n\nSee the complete [prospective report](results/PROSPECTIVE_RESULTS.md), [statistical analysis](results/statistical_analysis.md), and [analysis integrity audit](results/analysis_integrity.json).\n')
 (R/'analysis_summary.json').write_text(json.dumps(r,indent=2)+'\n')


def analysis_audit(r,methods,panel,decisions,raw_hash):
 """Independent integer tally replay, rather than calling analysis prefix/adaptive."""
 checks={}
 gen.verify_freeze()
 assert hashlib.sha256(gen.RAW_PATH.read_bytes()).hexdigest()==raw_hash
 checks['frozen_manifest_and_raw_hash_unchanged']='PASS'
 byq={q:[] for q in range(500)}
 for d in decisions: byq[d['question_index']].append(d)
 for q,rr in panel.items():
  counts=Counter(); token_sum=0; invalid=0; stop=None; expected_decisions=[]
  prefixes={}
  for t,row in enumerate(rr,1):
   token_sum+=row['output_tokens']
   if row['vote_key'] is None: invalid+=1
   else: counts[row['vote_key']]+=1
   winner_keys=[k for k in counts if counts[k]==max(counts.values())] if counts else []
   selected=next((x for x in rr[:t] if x['vote_key'] in winner_keys),None)
   sorted_counts=sorted(counts.values(),reverse=True)
   numerator=sorted_counts[0]-(sorted_counts[1] if len(sorted_counts)>1 else 0) if counts else 0
   valid=t-invalid; margin=numerator/valid if valid else None
   x=dict(samples=t,tokens=token_sum,invalid=invalid,valid=valid,tie=len(winner_keys)>1,
    winner_sample=selected['sample_index'] if selected else None,answer=selected['raw_extracted_answer'] if selected else None,
    key=selected['vote_key'] if selected else None,margin=margin,
    correct=bool(selected and gen.math_equal(selected['raw_extracted_answer'],rr[0]['reference_answer'])))
   prefixes[t]=x
   if t>=2 and stop is None:
    # Exact integer inequality avoids sharing the production float comparison.
    should_stop=t==16 or (valid>0 and 2*numerator>=valid)
    expected_decisions.append(dict(question_index=q,checkpoint=t,valid_votes=valid,margin=margin,stop=should_stop,forced=t==16,cumulative_output_tokens=token_sum))
    if should_stop: stop=x
  assert byq[q]==expected_decisions
  assert methods['adaptive'][q]==stop
  for n in (1,2,4,8,16): assert methods[f'fixed_n{n}'][q]==prefixes[n]
 checks['independent_prefix_only_stopping_plurality_tie_scoring_replay']='PASS'
 assert sum(x['count'] for x in r['stopping_distribution'])==500
 assert sum(x['correct_count'] for x in r['stopping_distribution'])==r['metrics'][0]['correct_count']
 assert sum(x['total_output_tokens'] for x in r['stopping_distribution'])==r['metrics'][0]['total_output_tokens']
 for m in r['metrics']:
  xx=methods[m['method']]
  assert m['correct_count']==sum(x['correct'] for x in xx)
  assert m['total_output_tokens']==sum(x['tokens'] for x in xx)
  assert m['total_samples']==sum(x['samples'] for x in xx)
  assert m['accuracy']==m['correct_count']/500
  assert m['mean_samples']==m['total_samples']/500
  assert m['mean_output_tokens']==m['total_output_tokens']/500
 checks['500_questions_all_methods_stopping_counts_cost_reconciliation']='PASS'
 indices=np.random.default_rng(40260915).integers(0,500,size=(10000,500))
 boots={m:np.asarray([x['correct'] for x in xx],dtype=float)[indices].mean(1) for m,xx in methods.items()}
 for m in r['metrics']:
  assert np.allclose(np.percentile(boots[m['method']],[2.5,97.5]),[m['accuracy_ci_low'],m['accuracy_ci_high']],rtol=0,atol=1e-12)
 for c in r['paired_comparisons']:
  n=int(c['comparison'].split('fixed_n')[1]); aa=methods['adaptive']; bb=methods[f'fixed_n{n}']
  b=sum(x['correct'] and not y['correct'] for x,y in zip(aa,bb)); f=sum(y['correct'] and not x['correct'] for x,y in zip(aa,bb))
  assert b==c['adaptive_only'] and f==c['fixed_only']
  total=b+f
  exact=float(min(Fraction(1),2*sum((Fraction(math.comb(total,k),2**total) for k in range(min(b,f)+1)),Fraction(0)))) if total else 1.
  assert abs(exact-c['exact_two_sided_mcnemar_p'])<1e-14
  assert np.allclose(np.percentile(100*(boots['adaptive']-boots[f'fixed_n{n}']),[2.5,97.5]),[c['ci_low_pp'],c['ci_high_pp']],rtol=0,atol=1e-12)
  assert abs(c['accuracy_difference_pp']-100*(b-f)/500)<1e-12
 checks['preregistered_bootstrap_seed_pairing_and_exact_binomial_test']='PASS'
 for name in ('prospective_summary.csv','fixed_budget_baselines.csv','paired_comparison.csv','stopping_distribution.csv'):
  with (R/name).open() as f: saved=list(csv.DictReader(f))
  source=r['metrics'] if name=='prospective_summary.csv' else r['metrics'][1:] if name=='fixed_budget_baselines.csv' else r['paired_comparisons'] if name=='paired_comparison.csv' else r['stopping_distribution']
  assert len(saved)==len(source)
  for sr,actual in zip(saved,source):
   assert all(sr[k]==('' if v is None else str(v)) for k,v in actual.items())
 checks['reported_csvs_match_validated_results']='PASS'
 # Plotting subprocess consumes only saved aggregate results, not raw outcomes.
 subprocess.run(['/tmp/phase3-plot-env/bin/python',str(P/'scripts/plot_prospective.py')],check=True)
 gen.verify_freeze()
 assert hashlib.sha256(gen.RAW_PATH.read_bytes()).hexdigest()==raw_hash
 for stem in ('accuracy_vs_mean_samples','accuracy_vs_output_tokens','stopping_distribution','adaptive_vs_fixed_budget_tradeoff'):
  for suffix in ('pdf','svg','png'):
   assert (P/'figures'/f'{stem}.{suffix}').stat().st_size>1000
 checks['publication_figures_and_post_render_frozen_environment']='PASS'
 files=[x for x in R.iterdir() if x.is_file() and x.name!='analysis_integrity.json']+[x for x in (P/'figures').iterdir() if x.is_file()]+[P/'scripts/analyze_prospective.py',P/'scripts/publish_prospective.py',P/'scripts/plot_prospective.py']
 audit=dict(time=datetime.now(timezone.utc).isoformat(),generation_integrity='PASS',frozen_policy_integrity='PASS',prospective_analysis='PASS',checks=checks,
  raw_sha256=raw_hash,frozen_policy_sha256=hashlib.sha256((P/'FROZEN_POLICY.md').read_bytes()).hexdigest(),experiment_freeze_sha256=hashlib.sha256((P/'EXPERIMENT_FREEZE.md').read_bytes()).hexdigest(),
  policy_id='margin_min2_0.50',policy_modified=False,policy_tuned=False,prospective_questions=500,unique_generation_pairs=8000,
  analysis_python=sys.executable,bootstrap_seed=40260915,bootstrap_resamples=10000,
  output_and_analysis_source_sha256={str(x.relative_to(P)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(files)})
 (R/'analysis_integrity.json').write_text(json.dumps(audit,indent=2)+'\n')
 print('GENERATION INTEGRITY: PASS\nFROZEN POLICY INTEGRITY: PASS\nPROSPECTIVE ANALYSIS: PASS')
