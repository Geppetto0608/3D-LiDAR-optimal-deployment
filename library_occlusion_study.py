"""Frozen paired clear/occluded development study; no execution without --execute."""
from argparse import ArgumentParser, SUPPRESS
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

import library_transfer_study as target_run
from prototype_offline import digest, save
from src.core.library_control_evaluation import plan_matched_controls
from src.core.library_installation import SCHEMA as INSTALLATION_SCHEMA, allowed_candidates, with_installation
from src.core.scene_occlusion import SCHEMA as OCCLUSION_SCHEMA, with_occlusion


BASE = Path(__file__).resolve().parent
SCHEMA = 'library_paired_occlusion_study_v1'
SWEEP_SCHEMA = 'library_paired_occlusion_sweep_v1'
SITING_SCHEMA = 'library_occlusion_siting_sweep_v1'
SWEEP_SCHEMAS = (SWEEP_SCHEMA, SITING_SCHEMA)


def expand_design(design):
    """Expand declared geometry axes without visibility, selection or file writes."""
    if not isinstance(design, dict) or set(design) != {'track_a', 'track_b'}:
        raise ValueError('Expected track_a and track_b geometry designs')
    a, b = design['track_a'], design['track_b']
    if (not isinstance(a, dict) or set(a) != {'road_widths_m', 'junction_centers_m', 'scenario_id_start'}
            or not isinstance(b, dict) or set(b) != {'lengths_m', 'road_widths_m', 'turn_degrees',
                                                   'bend_length_m', 'mesh_step_m', 'modules'}):
        raise ValueError('Unknown or missing geometry design fields')

    def axis(values, positive=True):
        if (not isinstance(values, list) or not 1 <= len(values) <= 64
                or any(type(v) not in (int, float) or not np.isfinite(v)
                       or (positive and v <= 0) for v in values)
                or len(set(values)) != len(values)):
            raise ValueError('Geometry axes require distinct finite numeric values')

    axis(a['road_widths_m'])
    axis(b['lengths_m'])
    axis(b['road_widths_m'])
    axis(b['turn_degrees'], positive=False)
    axis([b['bend_length_m']])
    axis([b['mesh_step_m']])
    centers = a['junction_centers_m']
    if (not isinstance(centers, list) or not 1 <= len(centers) <= 64
            or any(not isinstance(c, list) or len(c) != 2
                   or any(type(v) not in (int, float) or not np.isfinite(v) for v in c) for c in centers)
            or len({tuple(c) for c in centers}) != len(centers)):
        raise ValueError('Expected distinct finite junction centers')
    if (type(a['scenario_id_start']) is not int or a['scenario_id_start'] < 400000
            or type(b['modules']) is not int or b['modules'] < 1):
        raise ValueError('Synthetic scenario start >= 400000 and positive integer module count required')
    count = len(a['road_widths_m']) * len(centers)
    count += len(b['lengths_m']) * len(b['road_widths_m']) * len(b['turn_degrees'])
    if count > 64:
        raise ValueError('Geometry sweep exceeds 64-map guard')
    targets = []
    for index, (width, (x, y)) in enumerate(product(a['road_widths_m'], centers)):
        target = dict(id=a['scenario_id_start'] + index, parent_scenario_id=1,
            topology='T_JUNCTION', road_width=width, num_lanes=2,
            geometry=dict(junction_x_m=x, junction_y_m=y), provenance='synthetic_junction_offset')
        targets.append(dict(name=f'a_{index:03d}', track='A', target=target))
    for index, (length, width, angle) in enumerate(product(b['lengths_m'], b['road_widths_m'], b['turn_degrees'])):
        target = dict(geometry=dict(length_m=length, road_width_m=width, turn_deg=angle,
            bend_length_m=b['bend_length_m'], mesh_step_m=b['mesh_step_m']), modules=b['modules'])
        targets.append(dict(name=f'b_{index:03d}', track='B', target=target))
    return targets


def screen_spec(problem, recipe):
    if (not isinstance(recipe, dict) or set(recipe) != {
            'thickness_m', 'height_m', 'road_width_fraction', 'a_branch_station_m',
            'b_station_fractions'}):
        raise ValueError('Unknown or missing screen recipe fields')
    for key in ('thickness_m', 'height_m', 'road_width_fraction', 'a_branch_station_m'):
        v = recipe[key]
        if type(v) not in (int, float) or not np.isfinite(v) or v <= 0:
            raise ValueError('Screen recipe requires positive finite numbers')
    fractions = recipe['b_station_fractions']
    if (recipe['road_width_fraction'] >= 1 or not isinstance(fractions, list)
            or len(fractions) != 2 or any(type(v) not in (int, float) or not np.isfinite(v)
            or not 0 < v < 1 for v in fractions) or fractions != sorted(set(fractions))):
        raise ValueError('Expected two increasing interior station fractions and partial road span')
    geometry = problem['geometry']
    if problem['context']['track'] == 'A':
        center = [geometry.junction_x_m, geometry.junction_y_m]
        centers = [[center[0] + sign * recipe['a_branch_station_m'], center[1]] for sign in (-1, 1)]
        headings, width = [0., 0.], geometry.scenario.road_width
    else:
        centers, angles = geometry.centerline(np.asarray(fractions) * geometry.length_m)
        headings, width = np.rad2deg(angles).tolist(), geometry.road_width_m
    boxes = [dict(center_m=[float(xy[0]), float(xy[1]), recipe['height_m'] / 2],
                  size_m=[recipe['thickness_m'], width * recipe['road_width_fraction'], recipe['height_m']],
                  yaw_deg=heading) for xy, heading in zip(centers, headings)]
    return dict(schema=OCCLUSION_SCHEMA, boxes=boxes)


def installation_spec(problem, recipe):
    """One geometry-defined no-installation band, unrelated to coverage outcomes."""
    if (not isinstance(recipe, dict) or set(recipe) != {
            'band_length_m', 'a_east_station_m', 'b_station_fraction'}):
        raise ValueError('Unknown or missing installation recipe fields')
    if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in recipe.values()):
        raise ValueError('Installation recipe requires positive finite numbers')
    if recipe['b_station_fraction'] >= 1:
        raise ValueError('Installation station fraction must be interior')
    geometry = problem['geometry']
    half = recipe['band_length_m'] / 2
    if problem['context']['track'] == 'A':
        x, y = geometry.junction_x_m + recipe['a_east_station_m'], geometry.junction_y_m
        width = geometry.scenario.road_width / 2 + geometry.sidewalk_width
        if x - half <= geometry.junction_x_m + width or x + half > geometry.map_len:
            raise ValueError('Installation band must fit the straight east approach')
    else:
        station = recipe['b_station_fraction'] * geometry.length_m
        straight_end = (geometry.length_m - geometry.bend_length_m) / 2
        if station - half < 0 or station + half > straight_end:
            raise ValueError('Installation band must fit the first straight approach')
        centers, _ = geometry.centerline(np.array([station]))
        x, y = centers[0]
        width = geometry.road_width_m / 2 + geometry.sidewalk_width_m
    return dict(schema=INSTALLATION_SCHEMA,
                forbidden_regions=[[float(x - half), float(y - width), float(x + half), float(y + width)]])


def prepare_suite(spec):
    sweep = isinstance(spec, dict) and spec.get('schema') in SWEEP_SCHEMAS
    siting = isinstance(spec, dict) and spec.get('schema') == SITING_SCHEMA
    targets_key = 'design' if sweep else 'targets'
    fields = {'schema', 'sources', targets_key, 'k_values', 'control_seeds', 'enumeration_limit', 'screens'}
    if siting:
        fields.add('installation_exclusion')
    if (not isinstance(spec, dict) or set(spec) != fields or spec['schema'] not in (SCHEMA, *SWEEP_SCHEMAS)
            or not isinstance(spec['sources'], dict) or set(spec['sources']) != {'A', 'B'}):
        raise ValueError('Invalid paired study schema')
    targets = expand_design(spec['design']) if sweep else spec['targets']
    if not isinstance(targets, list) or not 1 <= len(targets) <= (64 if sweep else 16):
        raise ValueError('Paired study exceeds target-count guard')
    sources = {track: (BASE / path).resolve() for track, path in spec['sources'].items()}
    cases, used = [], set()
    for row in targets:
        if (not isinstance(row, dict) or set(row) != {'name', 'track', 'target' if sweep else 'target_json'}
                or row['track'] not in sources or not isinstance(row['name'], str)
                or not row['name'] or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in row['name'])
                or row['name'] in used):
            raise ValueError('Target names must be distinct safe lowercase identifiers')
        used.add(row['name'])
        if sweep:
            target = row['target']
            encoded = json.dumps(target, sort_keys=True, allow_nan=False).encode('utf-8')
            path, target_hash = None, hashlib.sha256(encoded).hexdigest()
        else:
            path = (BASE / row['target_json']).resolve()
            target = json.loads(path.read_text())
            target_hash = digest(path)
        problem = target_run.prepare(row['track'], sources[row['track']], target, spec['k_values'])
        occluded = screen_spec(problem, spec['screens'])
        installations = [('open', None)] if not siting else [
            ('open', dict(schema=INSTALLATION_SCHEMA, forbidden_regions=[])),
            ('restricted', installation_spec(problem, spec['installation_exclusion']))]
        for level, installation in installations:
            eligible = problem if installation is None else with_installation(problem, installation)
            pair = f'{row["name"]}_{level}' if siting else row['name']
            for condition, occlusion in [('clear', dict(schema=OCCLUSION_SCHEMA, boxes=[])), ('occluded', occluded)]:
                changed = with_occlusion(eligible, occlusion)
                plan = plan_matched_controls(changed, spec['control_seeds'], spec['enumeration_limit'])
                case = dict(name=f'{pair}_{condition}', pair=pair, condition=condition,
                    track=row['track'], target=target, target_path=str(path) if path else None,
                    target_sha256=target_hash, occlusion=occlusion, plan=plan)
                if siting:
                    case.update(base_map=row['name'], installation_level=level, installation=installation)
                cases.append(case)
    return sources, cases


def verify_pairs(root, cases):
    """Integrity checks only; no method ranking or outcome-driven condition changes."""
    checked = []
    for pair in dict.fromkeys(case['pair'] for case in cases):
        rows = {c['condition']: c for c in cases if c['pair'] == pair}
        if set(rows) != {'clear', 'occluded'}:
            raise ValueError('Each pair requires clear and occluded runs')
        paths = {key: root / 'runs' / row['name'] for key, row in rows.items()}
        reports = {key: json.loads((path / 'report.json').read_text()) for key, path in paths.items()}
        for report in reports.values():
            if not report['validation']['passed']:
                raise ValueError('Target validation failed')
        with np.load(paths['clear'] / 'target.npz', allow_pickle=False) as clear, \
                np.load(paths['occluded'] / 'target.npz', allow_pickle=False) as blocked:
            for field in ('poses', 'targets', 'pool_ids'):
                if not np.array_equal(clear[field], blocked[field]):
                    raise ValueError(f'Occlusion changed the paired {field}')
            if clear['masks'].shape != blocked['masks'].shape or np.any(blocked['masks'] & ~clear['masks']):
                raise ValueError('Additive opaque boxes created new visibility')
            removed = int(np.count_nonzero(clear['masks'] & ~blocked['masks']))
            changed = int(np.count_nonzero(np.any(clear['masks'] != blocked['masks'], axis=1)))
        plans = [{arm['arm_id']: (arm['query_plan']['selected_ids'], arm['physical_query_ids'])
                  for arm in report['arms']} for report in reports.values()]
        if plans[0] != plans[1]:
            raise ValueError('Paired physical query plans differ')
        checked.append(dict(pair=pair, domains_and_query_plans_identical=True,
            additive_visibility_check=True, removed_visibility_edges=removed,
            affected_candidates=changed, effective_occlusion=removed > 0))
    return dict(passed=True, pairs=checked,
        semantics='Edges are candidate-target visibility entries, not ray counts. Zero-effect cases are retained.')


def verify_installation_pairs(root, cases):
    """A unary exclusion must change only the candidate domain, not visibility."""
    checked = []
    for base_map, condition in dict.fromkeys((c['base_map'], c['condition']) for c in cases):
        members = [c for c in cases if (c['base_map'], c['condition']) == (base_map, condition)]
        rows = {c['installation_level']: c for c in members}
        if len(members) != 2 or set(rows) != {'open', 'restricted'}:
            raise ValueError('Each installation pair requires open and restricted runs')
        paths = {key: root / 'runs' / row['name'] for key, row in rows.items()}
        reports = {key: json.loads((path / 'report.json').read_text()) for key, path in paths.items()}
        if any(not r['validation'].get('installation_passed') or not r['validation']['passed']
               for r in reports.values()):
            raise ValueError('Installation validation missing or failed')
        with np.load(paths['open'] / 'target.npz', allow_pickle=False) as full, \
                np.load(paths['restricted'] / 'target.npz', allow_pickle=False) as limited:
            allowed = np.flatnonzero(allowed_candidates(full['poses'], rows['restricted']['installation']))
            if (not np.array_equal(full['original_candidate_ids'], np.arange(len(full['poses'])))
                    or not np.array_equal(limited['original_candidate_ids'], allowed)):
                raise ValueError('Installation original candidate mapping mismatch')
            if not np.array_equal(full['targets'], limited['targets']):
                raise ValueError('Installation changed target order')
            for field in ('poses', 'masks'):
                if not np.array_equal(full[field][allowed], limited[field]):
                    raise ValueError(f'Installation changed retained {field}')
            expected_pool = np.flatnonzero(np.isin(allowed, full['pool_ids']))
            if not np.array_equal(limited['pool_ids'], expected_pool):
                raise ValueError('Installation changed surviving library pool')
            for level, archive in [('open', full), ('restricted', limited)]:
                report = reports[level]
                meta = report['plan']['context']['installation']
                if (meta['spec'] != rows[level]['installation']
                        or meta['eligible_to_original_ids'] != archive['original_candidate_ids'].tolist()
                        or report['plan']['query_budget'] != len(archive['pool_ids'])):
                    raise ValueError('Saved installation plan disagrees with archive')
            before, after = len(full['pool_ids']), len(limited['pool_ids'])
            removed = len(full['poses']) - len(limited['poses'])
        with np.load(paths['open'] / 'mesh.npz', allow_pickle=False) as full, \
                np.load(paths['restricted'] / 'mesh.npz', allow_pickle=False) as limited:
            if any(not np.array_equal(full[field], limited[field]) for field in ('vertices', 'triangles')):
                raise ValueError('Installation changed scene mesh')
        checked.append(dict(base_map=base_map, condition=condition, excluded_candidates=removed,
                            query_budget_before=before, query_budget_after=after,
                            empty_library_pool=after == 0, retained_visibility_identical=True))
    return dict(passed=True, pairs=checked,
                semantics='Query budgets match within each case, not necessarily across installation levels.')


def _worker(root):
    root = Path(root).resolve()
    if BASE != root / 'source':
        raise ValueError('Batch worker requires frozen source')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['schema'] not in (SCHEMA, *SWEEP_SCHEMAS) or manifest['status'] != 'prepared':
        raise ValueError('Batch requires a newly prepared manifest')
    started = time.perf_counter()
    try:
        target_run._verify(BASE, manifest['source_hashes'])
        target_run._verify(root, manifest['input_hashes'])
        for track, hashes in manifest['library_hashes'].items():
            target_run._verify(root / 'inputs/libraries' / track, hashes)
        spec = json.loads((root / 'inputs/spec.json').read_text())
        cases = json.loads((root / 'inputs/cases.json').read_text())
        manifest['status'] = 'running'
        save(root / 'manifest.json', manifest)
        for case in cases:
            if manifest['schema'] in SWEEP_SCHEMAS and shutil.disk_usage(root).free < 2 * 1024**3:
                raise RuntimeError('Sweep requires at least 2 GiB free disk before each case')
            manifest['current_case'] = case['name']
            save(root / 'manifest.json', manifest)
            print(f'START {case["name"]}', flush=True)
            output = target_run.execute(case['track'], root / 'inputs/libraries' / case['track'],
                case['target'], spec['k_values'], root / 'runs' / case['name'],
                control_seeds=spec['control_seeds'], enumeration_limit=spec['enumeration_limit'],
                occlusion=case['occlusion'], installation=case.get('installation'), command=manifest['command'])
            report = json.loads((output / 'report.json').read_text())
            for key, expected in case['plan'].items():
                if key not in ('context', 'execute') and report['plan'][key] != expected:
                    raise ValueError(f'Frozen plan changed: {key}')
            if report['plan']['context']['scene_occlusion'] != case['occlusion']:
                raise ValueError('Frozen occlusion input changed')
            if ('installation' in case
                    and report['plan']['context']['installation']['spec'] != case['installation']):
                raise ValueError('Frozen installation input changed')
            manifest['completed'].append(case['name'])
            save(root / 'manifest.json', manifest)
            print(f'COMPLETE {case["name"]}', flush=True)
        save(root / 'pair_validation.json', verify_pairs(root, cases))
        if manifest['schema'] == SITING_SCHEMA:
            save(root / 'installation_validation.json', verify_installation_pairs(root, cases))
            manifest['installation_validation_sha256'] = digest(root / 'installation_validation.json')
        target_run._verify(BASE, manifest['source_hashes'])
        target_run._verify(root, manifest['input_hashes'])
        for track, hashes in manifest['library_hashes'].items():
            target_run._verify(root / 'inputs/libraries' / track, hashes)
        manifest.update(status='complete', validation_sha256=digest(root / 'pair_validation.json'))
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest.update(finished_at=datetime.now(timezone.utc).isoformat(),
                        batch_worker_wall_sec=time.perf_counter() - started)
        save(root / 'manifest.json', manifest)


def execute_suite(spec, output, command):
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Output must be new')
    sources, cases = prepare_suite(spec)
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if spec['schema'] in SWEEP_SCHEMAS and shutil.disk_usage(ancestor).free < 2 * 1024**3:
        raise RuntimeError('Sweep requires at least 2 GiB free disk')
    if any(source in output.resolve().parents for source in sources.values()):
        raise ValueError('Output cannot be inside a source library')
    libraries = {track: target_run._file_hashes(source) for track, source in sources.items()}
    names = [*target_run.source_names(), 'library_occlusion_study.py',
             'LIBRARY_QUERY_CONTROLS.md', 'LIBRARY_OCCLUSION_PROTOCOL.md']
    if spec['schema'] == SITING_SCHEMA:
        names.append('LIBRARY_SITING_PROTOCOL.md')
    contents = {name: (BASE / name).read_bytes() for name in names}
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    for track, source in sources.items():
        shutil.copytree(source, output / 'inputs/libraries' / track)
        target_run._verify(output / 'inputs/libraries' / track, libraries[track])
    save(output / 'inputs/spec.json', spec)
    save(output / 'inputs/cases.json', cases)
    for name, content in contents.items():
        path = output / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    timeout = len(cases) * 600 + 120
    if spec['schema'] in SWEEP_SCHEMAS:
        timeout = min(timeout, 7200)
    manifest = dict(schema=spec['schema'], status='prepared', created_at=datetime.now(timezone.utc).isoformat(),
        command=command, python=sys.version, original_libraries={k: str(v) for k, v in sources.items()},
        library_hashes=libraries, expected_case_count=len(cases), completed=[],
        expected_comparison_rows=len(cases) * (4 + len(spec['control_seeds'])) * len(spec['k_values']),
        source_hashes={name: digest(output / 'source' / name) for name in names},
        input_hashes={name: digest(output / name) for name in ('inputs/spec.json', 'inputs/cases.json')},
        case_timeout_sec=600, batch_timeout_sec=timeout,
        execution_environment={name: os.environ.get(name) for name in
            ('CUDA_VISIBLE_DEVICES', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    save(output / 'manifest.json', manifest)
    try:
        with (output / 'launcher.log').open('x') as log:
            with subprocess.Popen([sys.executable, '-B', str(output / 'source/library_occlusion_study.py'),
                '--worker-root', str(output), '--execute'], stdout=log, stderr=subprocess.STDOUT,
                cwd=output / 'source', start_new_session=True) as worker:
                try:
                    returncode = worker.wait(timeout=manifest['batch_timeout_sec'])
                except BaseException:
                    # A batch deadline must also stop its active nested target worker.
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
                    raise
        manifest = json.loads((output / 'manifest.json').read_text())
        if returncode or manifest['status'] != 'complete':
            raise RuntimeError(f'Batch failed; see {output / "launcher.log"}')
        for track, source in sources.items():
            if target_run._file_hashes(source) != libraries[track]:
                raise ValueError('Original source library changed')
        manifest['original_sources_unchanged'] = True
    except BaseException as exc:
        manifest = json.loads((output / 'manifest.json').read_text())
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        save(output / 'manifest.json', manifest)
    return output


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--spec', default='configs/library_occlusion_20261122.json')
    parser.add_argument('--output', default='results/library_occlusion_20261122')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-root', help=SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_root:
        if not args.execute:
            raise ValueError('Worker requires --execute')
        return _worker(args.worker_root)
    spec = json.loads(Path(args.spec).read_text())
    if not args.execute:
        _, cases = prepare_suite(spec)
        plan = dict(execute=False, schema=spec['schema'], expected_case_count=len(cases),
            expected_comparison_rows=sum(c['plan']['expected_arm_count'] * len(c['plan']['k_values']) for c in cases),
            cases=[dict(name=c['name'],
            candidate_count=c['plan']['candidate_count'], target_count=c['plan']['target_count'],
            full_subsets=c['plan']['subset_counts']['full_reference'],
            query_budget=c['plan']['query_budget'], arms=c['plan']['expected_arm_count']) for c in cases])
        print(json.dumps(plan, indent=2))
        return plan
    root = execute_suite(spec, args.output,
        [sys.executable, str(BASE / 'library_occlusion_study.py'), *(sys.argv[1:] if argv is None else argv)])
    print(f'Complete: {root}')
    return root


if __name__ == '__main__':
    main()
