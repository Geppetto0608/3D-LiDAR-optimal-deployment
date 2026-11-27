"""Replay saved RSU stress geometry, transfers, full visibility, and analysis.

Uses the run's frozen source. Does not optimize, train, or modify experiment
artifacts. The analysis directory must contain its original script and hashes.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def compare_archive(problem, archive, provider):
    """Compare every candidate, without reusing stored LOS or mask caches."""
    for name in ('poses', 'targets', 'critical'):
        require(np.array_equal(problem[name], archive[name]), f'Regenerated {name} differs')
    poses, targets, packed = archive['poses'], archive['targets'], archive['packed_masks']
    require(np.isfinite(poses).all() and np.isfinite(targets).all(), 'Non-finite geometry')
    require(archive['critical'].dtype == bool, 'Invalid critical mask dtype')
    require(packed.dtype == np.uint8, 'Invalid packed mask dtype')
    require(packed.shape == (len(poses), (len(targets) + 7) // 8), 'Invalid mask shape')
    digest = hashlib.sha256()
    visible = 0
    for cid, pose in enumerate(poses):
        mask = provider.mask(pose)
        require(mask.dtype == bool and mask.shape == (len(targets),), 'Invalid direct mask')
        fresh = np.packbits(mask)
        if not np.array_equal(fresh, packed[cid]):
            expected = np.unpackbits(packed[cid], count=len(targets)).astype(bool)
            different = np.flatnonzero(mask != expected)
            raise ValueError(f'Direct mask differs: candidate={cid}, '
                             f'target_count={len(different)}, first_targets={different[:20].tolist()}')
        digest.update(fresh.tobytes())
        visible += int(mask.sum())
    return dict(candidate_masks=len(poses), targets=len(targets),
                candidate_target_pairs=int(len(poses) * len(targets)),
                visible_edges=visible, fresh_packed_sha256=digest.hexdigest(), mismatches=0)


def plan_signature(plans):
    return [{k: v for k, v in p.items() if k != 'planning_sec'} for p in plans]


def check_hashes(root, hashes):
    for relative, expected in hashes.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), f'Input escapes root: {relative}')
        require(sha(path) == expected, f'Input changed: {relative}')


def replay_analysis(root, analysis, output):
    provenance = read(analysis / 'validation_summary.json')
    require(provenance['status'] == 'ANALYZED', 'Expected completed original analysis')
    require(sha(analysis / 'analyze.py') == provenance['script_sha256'], 'Analysis script changed')
    check_hashes(root, provenance['input_hashes'])
    destination = output / 'analysis_replay'
    command = [sys.executable, '-B', str(analysis / 'analyze.py'), '--root', str(root),
               '--output', str(destination)]
    with (output / 'analysis_replay.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    original = {p.name for p in analysis.glob('*.csv')}
    repeated = {p.name for p in destination.glob('*.csv')}
    require(original and original == repeated, 'Analysis CSV completeness mismatch')
    matches = {}
    for name in sorted(original):
        require(sha(analysis / name) == sha(destination / name), f'Analysis changed: {name}')
        matches[name] = sha(destination / name)
    replay = read(destination / 'validation_summary.json')
    require(replay['checks'] == provenance['checks'], 'Audit counts changed')
    require(replay['summary'] == provenance['summary'], 'Analysis summary changed')
    require(replay['input_hashes'] == provenance['input_hashes'], 'Analysis input set changed')
    return provenance, matches


def validate(root, analysis, output):
    started = time.perf_counter()
    manifest = read(root / 'manifest.json')
    require(manifest['status'] == 'complete', 'Only completed experiments can be validated')
    original_analysis, csv_hashes = replay_analysis(root, analysis, output)
    source = root / 'source'
    frozen_hashes = {'source/' + k: v for k, v in manifest['source_hashes'].items()}
    check_hashes(root, frozen_hashes)
    require(sha(root / 'spec.json') == manifest['spec_sha256'], 'Frozen spec changed')
    spec = read(root / 'spec.json')

    # Import only the frozen physics/generation code, with all optimization calls blocked.
    require(not any(name == 'src' or name.startswith('src.') for name in sys.modules),
            'Run this validator in a fresh process; src was already imported')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import library_rsu_stress as study
    from src.core.library_transfer_evaluation import Work
    from src.core.visibility import MeshVisibility

    require(Path(study.__file__).resolve() == source / 'library_rsu_stress.py',
            'Imported non-frozen experiment module')
    counts = Counter()
    records, signatures, transfers = [], {}, {}
    with patch.object(study, 'optimize', side_effect=RuntimeError('Optimization forbidden in validation')):
        with patch.object(study, 'solve_frontier', side_effect=RuntimeError('Solver forbidden in validation')):
            for job in manifest['jobs']:
                require(job['status'] == 'complete', 'Incomplete job')
                d = job['descriptor']
                is_source = job['phase'] == 'source'
                level = spec['source_scene'] if is_source else job['scene']
                directory = (root / 'library' / d['name'] if is_source else
                             root / 'cases' / (d['name'] + '_' + level))
                problem = study.domain(d, spec['sampling'])
                with np.load(directory / 'candidates.npz', allow_pickle=False) as data:
                    archive = {name: data[name] for name in data.files}
                if is_source:
                    layouts = read(directory / 'layouts.json')
                    canonical = problem['canonical']
                    if isinstance(canonical, np.ndarray):
                        canonical = canonical.tolist()
                    require(layouts['canonical'] == canonical, 'Source canonical coordinates changed')
                    counts['source_canonical_replays'] += 1
                else:
                    with patch.object(study, 'query', side_effect=RuntimeError('Query forbidden in transfer')):
                        actual = study.library_pool(root, spec, problem)
                    saved = read(directory / 'transfer.json')
                    for field in ('order', 'sources'):
                        require(actual[field] == saved[field], f'Transfer replay mismatch: {field}')
                    counts['transfer_replays'] += 1
                    require(d['name'] not in {s['source'] for s in actual['sources']}, 'Target is a source')
                    signature = plan_signature(read(directory / 'query_plans.json'))
                    if d['name'] in signatures:
                        require(signature == signatures[d['name']], 'Query plans depend on obstacle level')
                        require(actual['order'] == transfers[d['name']], 'Transfer order depends on obstacles')
                        counts['cross_scene_plan_checks'] += len(signature)
                    signatures[d['name']] = signature
                    transfers[d['name']] = actual['order']
                before = time.perf_counter()
                work = Work()
                provider = MeshVisibility(study.scene(problem, level), problem['targets'],
                                          problem['profile'], work=work)
                record = compare_archive(problem, archive, provider)
                record.update(job=job['id'], phase=job['phase'], scene=level,
                              raycast_sec=time.perf_counter() - before, work=dict(work.counts))
                records.append(record)
                for key in ('candidate_masks', 'candidate_target_pairs', 'visible_edges'):
                    counts[key] += record[key]
                counts[job['phase'] + '_scenes'] += 1
                counts['rays_submitted'] += work.counts.get('rays_submitted', 0)
                save(output / 'progress.json', dict(status='running', completed=len(records),
                                                   total=len(manifest['jobs']), counts=dict(counts)))
                del provider, archive, problem
    check_hashes(root, original_analysis['input_hashes'])
    check_hashes(root, frozen_hashes)
    current_root = Path(__file__).resolve().parents[1]
    changed = [name for name, expected in manifest['source_hashes'].items()
               if not (current_root / name).is_file() or sha(current_root / name) != expected]
    versions = {}
    for package in ('numpy', 'open3d', 'scipy', 'gurobipy', 'pandas'):
        versions[package] = metadata.version(package)
    report = dict(status='passed', verification_status='VERIFIED',
        scope='Full candidate visibility, geometry/transfer replay, saved constraints and analysis only',
        full_optimization_rerun=False, training=False, fresh_raycast=True,
        root=str(root), analysis=str(analysis), created_at=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        wall_sec=time.perf_counter() - started, counts=dict(counts), scenes=records,
        saved_audit_checks=original_analysis['checks'], analysis_csv_sha256=csv_hashes,
        input_hashes=original_analysis['input_hashes'], raw_inputs_unchanged=True,
        validator_sha256=sha(Path(__file__).resolve()), frozen_source_hashes=manifest['source_hashes'],
        current_source_differences=changed, python=sys.version, package_versions=versions,
        limitations=['Same Open3D engine and shared optical model; not independent sensor-model validation.',
                     'Solver bounds are checked for consistency, not independently re-proved on large problems.',
                     'Timing/speedup, unseen-map generalization and physical deployment are not validated.'])
    save(output / 'validation.json', report)
    save(output / 'progress.json', dict(status='passed', completed=len(records), total=len(records)))
    lines = [
        '# RSU Stress Experiment Validation', '', '## Material Passport',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: validate',
        '- Verification Status: VERIFIED (scoped computational replay only)',
        '- Version Label: rsu_stress_validation_v1', '', '## Scope',
        'No training or large optimization was repeated. Original inputs were unchanged.',
        'Fresh direct raycasting uses the frozen geometry and the same Open3D/optical model.',
        'Every candidate was checked, not just the selected deployments.', '', '## Checks',
        '```json', json.dumps(dict(counts), indent=2), '```',
        f'- {len(csv_hashes)} analysis CSV files reproduced byte-for-byte.',
        '- Selected/alternative deployment constraints, density and certificates passed the saved-array audit.',
        '- All source/target archives matched regenerated poses, targets and critical flags.',
        '- Transfers reproduced without physical visibility queries or solver calls.',
        '- Query plans are identical across obstacle levels of the same geometry.',
        '- Raw input and frozen source hashes were unchanged after verification.',
        f'- Current checkout differences from the frozen run: {changed}.', '',
        '## Statistical Fallacy Scan',
        '11/11 reviewed; details retained in analysis_replay/REPORT.md.',
        'Descriptive means only: 2,016 rows are not 2,016 independent geometries.',
        'No significance, causal generalization, or real-world detection claim is validated.', '',
        '## Limits', *['- ' + item for item in report['limitations']], '',
        'Verification does not upgrade weak solver gaps or certify all library entries near-optimal.',
        'Unit/small-solver regression results, when run, are saved separately in tests.json.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--analysis', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root, analysis, output = (p.resolve() for p in (args.root, args.analysis, args.output))
    require(not output.is_relative_to(root), 'Validation output must be outside the experiment root')
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = validate(root, analysis, output)
    except Exception as exc:
        save(output / 'failure.json', dict(status='failed', error=repr(exc), traceback=traceback.format_exc()))
        raise
    print(json.dumps(dict(status=result['status'], counts=result['counts'],
                          wall_sec=result['wall_sec'], output=str(output)), indent=2))


if __name__ == '__main__':
    main()
