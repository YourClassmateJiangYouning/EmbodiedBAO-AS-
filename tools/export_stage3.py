"""Export the final 100 Stage 3 episodes as portable CSV/JSONL plus summaries."""
from __future__ import annotations
import csv, glob, hashlib, json, os, re, shutil, tarfile
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'exports', 'stage3_final100_20261101')
os.makedirs(OUT, exist_ok=True)

# Collect final episode JSONs (exclude the per-step sidecars).
files = []
for f in glob.glob(os.path.join(ROOT, 'results', '**', 'stage3-*', 'episode_*.json'), recursive=True):
    if re.search(r'/episode_\d+\.json$', f):
        files.append(f)
by_pair = {}
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    pid = d.get('stage3_pair_id')
    if not pid:
        continue
    if pid not in by_pair or os.path.getmtime(f) > os.path.getmtime(by_pair[pid][0]):
        by_pair[pid] = (f, d)
episodes = [v[1] for _, v in sorted(by_pair.items(), key=lambda kv: kv[0])]
print('episode json files', len(files), 'unique pairs', len(episodes))

rows = []
for d in episodes:
    steps = d.get('steps') or []
    invalid = [s for s in steps if s.get('action') == 'invalid']
    timeouts = [s for s in invalid if any(k in (s.get('reasoning') or '').lower() for k in ('timed out', 'timeout', 'urlopen error'))]
    rows.append({
        'pair_id': d.get('stage3_pair_id'),
        'model': d.get('stage3_source_model') or d.get('model_name'),
        'robot': d.get('stage3_robot'),
        'robot_name': d.get('stage3_robot_name'),
        'scene': d.get('stage3_scene'),
        'a_s_requested': d.get('stage3_a_s_requested'),
        'a_s_built': d.get('a_s_ratio'),
        'channel_width_m': d.get('channel_width'),
        'shoulder_width_m': d.get('stage3_shoulder_width'),
        'passed': d.get('passed'),
        'total_steps': d.get('total_steps'),
        'invalid_response_count': d.get('invalid_response_count'),
        'timeout_invalid_count': len(timeouts),
        'end_reason': d.get('end_reason'),
        'final_position_x': d.get('final_position_x'),
        'final_position_z': d.get('final_position_z'),
        'final_torso_rotation_deg': d.get('final_torso_rotation'),
        'wall_collision_count': d.get('wall_collision_count'),
        'turn_blocked_count': d.get('turn_blocked_count'),
        'max_rotation_deg': d.get('max_rotation_deg'),
        'total_llm_time_ms': d.get('total_llm_time_ms'),
        'source_scene': d.get('stage3_source_scene'),
        'source_level': d.get('stage3_source_level'),
        'source_a_s': d.get('stage3_source_a_s'),
        'source_episode_id': d.get('stage3_source_episode_id'),
        'source_marker_slot': d.get('stage3_source_marker_slot'),
        'source_model': d.get('stage3_source_model'),
        'result_json': os.path.relpath(by_pair[d.get('stage3_pair_id')][0], ROOT),
    })

# Episodes CSV.
with open(os.path.join(OUT, 'stage3_episodes.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# Full JSONL (includes steps).
with open(os.path.join(OUT, 'stage3_episodes.jsonl'), 'w') as f:
    for d in episodes:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

# Long per-step CSV.
step_fields = ['pair_id','model','robot','scene','a_s_requested','step','action','feedback','step_success','collision','position_x','position_z','torso_rotation_deg','llm_response_time_ms','confidence','reasoning','scene_description']
step_rows = []
for d in episodes:
    for s in d.get('steps') or []:
        step_rows.append({
            'pair_id': d.get('stage3_pair_id'),
            'model': d.get('stage3_source_model') or d.get('model_name'),
            'robot': d.get('stage3_robot'),
            'scene': d.get('stage3_scene'),
            'a_s_requested': d.get('stage3_a_s_requested'),
            'step': s.get('step'),
            'action': s.get('action'),
            'feedback': s.get('feedback'),
            'step_success': s.get('step_success'),
            'collision': s.get('collision'),
            'position_x': s.get('position_x'),
            'position_z': s.get('position_z'),
            'torso_rotation_deg': s.get('torso_rotation'),
            'llm_response_time_ms': s.get('llm_response_time_ms'),
            'confidence': s.get('confidence'),
            'reasoning': s.get('reasoning'),
            'scene_description': s.get('scene_description'),
        })
with open(os.path.join(OUT, 'stage3_steps.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=step_fields); w.writeheader(); w.writerows(step_rows)

# Failure list.
failures = [r for r in rows if not r['passed']]
with open(os.path.join(OUT, 'stage3_failures.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(failures)

# Summary markdown.
L = []
def line(x=''): L.append(x)
line('# Stage 3 final 100 — export summary')
line()
line(f'- Episodes: {len(rows)}')
line(f'- Passed: {sum(bool(r["passed"]) for r in rows)}')
line(f'- Failed: {len(failures)}')
line(f'- Total step rows: {len(step_rows)}')
line(f'- Timeout-invalid steps across all episodes: {sum(int(r["timeout_invalid_count"]) for r in rows)}')
line(f'- Episodes with at least one timeout: {sum(int(r["timeout_invalid_count"]) > 0 for r in rows)}')
line()
line('## By model')
line()
line('| Model | n | passed | failed | timeout-affected |')
line('|---|---:|---:|---:|---:|')
for m in sorted({r['model'] for r in rows}):
    g=[r for r in rows if r['model']==m]
    line(f'| {m} | {len(g)} | {sum(bool(r["passed"]) for r in g)} | {sum(not bool(r["passed"]) for r in g)} | {sum(int(r["timeout_invalid_count"])>0 for r in g)} |')
line()
line('## By robot')
line()
line('| Robot | n | passed | failed | timeout-affected |')
line('|---|---:|---:|---:|---:|')
for m in ('G1','R1'):
    g=[r for r in rows if r['robot']==m]
    line(f'| {m} | {len(g)} | {sum(bool(r["passed"]) for r in g)} | {sum(not bool(r["passed"]) for r in g)} | {sum(int(r["timeout_invalid_count"])>0 for r in g)} |')
line()
line('## By A/S')
line()
line('| A/S | n | passed | failed |')
line('|---:|---:|---:|---:|')
for a in sorted({float(r['a_s_requested']) for r in rows}):
    g=[r for r in rows if abs(float(r['a_s_requested'])-a)<1e-9]
    line(f'| {a:.2f} | {len(g)} | {sum(bool(r["passed"]) for r in g)} | {sum(not bool(r["passed"]) for r in g)} |')
line()
line('## Failed episodes')
line()
line('| Pair | Model | Robot | Scene | A/S | Steps | Invalid | Timeout invalid | End reason |')
line('|---|---|---|---|---:|---:|---:|---:|---|')
for r in failures:
    line(f'| {r["pair_id"]} | {r["model"]} | {r["robot"]} | {r["scene"]} | {float(r["a_s_requested"]):.1f} | {r["total_steps"]} | {r["invalid_response_count"]} | {r["timeout_invalid_count"]} | {r["end_reason"]} |')
open(os.path.join(OUT, 'stage3_summary.md'), 'w').write('\n'.join(L)+'\n')

# Copy the input plan and prior audit for provenance.
for src, dst in [
    ('reports/stage3_final100_plan.csv','stage3_input_plan.csv'),
    ('reports/stage3_timeout_audit.csv','stage3_timeout_audit_original.csv'),
]:
    p=os.path.join(ROOT,src)
    if os.path.exists(p): shutil.copy2(p, os.path.join(OUT,dst))

# Manifest with hashes and sizes.
manifest = {'export': OUT, 'episodes': len(rows), 'passed': sum(bool(r['passed']) for r in rows), 'failed': len(failures), 'step_rows': len(step_rows), 'files': {}}
for name in sorted(os.listdir(OUT)):
    p=os.path.join(OUT,name)
    if os.path.isfile(p):
        h=hashlib.sha256()
        with open(p,'rb') as fh:
            for b in iter(lambda: fh.read(1024*1024), b''): h.update(b)
        manifest['files'][name]={'bytes':os.path.getsize(p),'sha256':h.hexdigest()}
open(os.path.join(OUT,'manifest.json'),'w').write(json.dumps(manifest,indent=2))

# Compress the folder for transfer.
tar_path=OUT+'.tar.gz'
with tarfile.open(tar_path,'w:gz') as t:
    t.add(OUT, arcname=os.path.basename(OUT))
print('wrote',OUT)
print('archive',tar_path, os.path.getsize(tar_path))
print('passed',manifest['passed'],'failed',manifest['failed'])
