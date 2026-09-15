"""Read-only panel validation, with a separate recovery audit artifact."""
import json, hashlib, sys, subprocess, importlib.metadata, fcntl
from datetime import datetime, timezone
from collections import Counter
import generate_fresh_panel as g
import torch
p=g.PHASE3
with (p/'generation.lock').open('a') as lock:
 fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
 g.verify_freeze()
 runtime=json.loads((p/'runtime_audit.json').read_text())
 assert sys.executable==runtime['executable']
 assert torch.backends.cudnn.version()==runtime['cudnn']
 assert torch.cuda.get_device_properties(0).total_memory==runtime['gpu_vram_bytes']
 smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
 assert smi==runtime['nvidia_smi'], (smi,runtime['nvidia_smi'])
 subprocess.run([sys.executable,'-m','pip','check'],check=True)
 data=g.RAW_PATH.read_bytes(); lines=data.splitlines(keepends=True)
 rows=[]; errors=[]
 for n,line in enumerate(lines,1):
  try:
   assert line.endswith(b'\n') and line.strip(), 'blank or unterminated'
   row=json.loads(line); g.validate_record(row)
   assert row['normalized_answer']==g.normalize_answer(row['raw_extracted_answer'])
   assert 1<=row['batch_size']<=64
   assert abs(row['latency_seconds']-row['batch_wall_latency_seconds']/row['batch_size'])<0.000002
   rows.append(row)
  except Exception as e: errors.append({'line':n,'error':repr(e)})
 keys=[(r['question_index'],r['sample_index']) for r in rows]
 duplicates=len(keys)-len(set(keys))
 assert not errors and not duplicates, (errors,duplicates)
 ds=g.load_dataset(g.DATASET_ID,revision=g.DATASET_REVISION,split='test',cache_dir=str(p/'cache/datasets'))
 assert len(ds)==500
 for r in rows:
  d=ds[r['question_index']]; assert r['question']==d['problem'] and r['reference_answer']==d['answer']
 completed=g.load_completed(); expected={(q,s) for q in range(500) for s in range(1,17)}
 missing=sorted(expected-completed)
 attempts=[json.loads(x) for x in (p/'generation_attempts.jsonl').read_text().splitlines()]
 for a in attempts:
  assert a['seeds']==[g.derived_seed(q,s) for q,s in a['pairs']]
 audit={'time':datetime.now(timezone.utc).isoformat(),'status':'PASS','record_count':len(lines),'unique_pairs':len(completed),'duplicates':duplicates,'malformed_records':len(errors),'seed_consistency':'PASS','frozen_artifacts':'PASS','environment':'PASS','dataset_prompt_scoring_cost_validation':'PASS','raw_bytes':len(data),'raw_sha256':hashlib.sha256(data).hexdigest(),'missing_count':len(missing),'missing_pairs':missing,'missing_seeds':[g.derived_seed(q,s) for q,s in missing],'attempts_with_missing_pairs':sum(any(tuple(pair) not in completed for pair in a['pairs']) for a in attempts)}
 name='recovery_audit.json' if len(completed)<8000 else 'recovery_completion_audit.json'
 (p/name).write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps({k:v for k,v in audit.items() if k not in ('missing_pairs','missing_seeds')},indent=2))
 if len(completed)==8000:
  before=json.loads((p/'recovery_audit.json').read_text())
  assert hashlib.sha256(data[:before['raw_bytes']]).hexdigest()==before['raw_sha256'], 'Original records changed'
  assert completed==expected
  print('Original persisted bytes unchanged; exactly 500 x 16 pairs: PASS')
