"""Read-only snapshot of durable generation progress; no outcome inspection."""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
P=Path(__file__).resolve().parents[1]
state=json.loads((P/'generation_state.json').read_text())
b=(P/'raw_generations.jsonl').read_bytes() if (P/'raw_generations.jsonl').exists() else b''
lines=b.splitlines(keepends=True)
tail=bool(lines and not lines[-1].endswith(b'\n'))
if tail: lines=lines[:-1]
rows=[json.loads(x) for x in lines]
keys=[(r['question_index'],r['sample_index']) for r in rows]
assert len(keys)==len(set(keys))
assert all(0<=q<500 and 1<=s<=16 for q,s in keys)
assert all(r['seed']==r['generation_seed']==30260915+r['question_index']*16+r['sample_index']-1 for r in rows)
counts=Counter(q for q,s in keys)
elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(state['started_at'])).total_seconds()
rate=len(rows)/elapsed if elapsed else 0
print(json.dumps(dict(completed_generations=len(rows),target=8000,completed_questions=sum(v==16 for v in counts.values()),elapsed_seconds=elapsed,generations_per_second=rate,estimated_remaining_seconds=(8000-len(rows))/rate if rate else None,integrity='PASS for complete snapshot rows',in_progress_tail=tail,bytes=len(b),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader'],text=True).strip()),indent=2))
