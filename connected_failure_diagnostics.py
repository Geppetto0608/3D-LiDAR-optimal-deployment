"""Read-only hindsight analysis of saved candidate omissions and LS barriers."""
from argparse import ArgumentParser
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from src.methods.connected_diagnostics import (TOL, best_assignment, omission_witness,
    pair_table, search_diagnostic)
from src.roadside.corridor_requirements import CorridorRequirements


BASE = Path(__file__).resolve().parent
METHODS = ('independent_transfer', 'top_joint', 'diverse_joint', 'top_joint_ls',
           'diverse_joint_ls', 'reference_one_per_module', 'reference_global')
CODE = ['connected_failure_diagnostics.py', 'src/methods/connected_diagnostics.py',
        'src/roadside/corridor_requirements.py'] + [str(p.relative_to(BASE)) for p in sorted((BASE/'src/core').glob('*.py'))]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


class Inputs:
    def __init__(self, root):
        self.root, self.hashes = Path(root), {}

    def file(self, relative):
        path = self.root / relative
        key, value = str(path.relative_to(self.root)), digest(path)
        if key in self.hashes and self.hashes[key] != value:
            raise ValueError(f'Original study input changed during analysis: {key}')
        self.hashes[key] = value
        return path

    def json(self, relative):
        return json.loads(self.file(relative).read_text())

    def unchanged(self):
        if any(digest(self.root/p) != h for p, h in self.hashes.items()):
            raise ValueError('Original study inputs changed during the analysis')


def checked_manifest(inputs):
    manifest = inputs.json('manifest.json')
    validation = inputs.json('independent_validation.json')
    if (manifest['status'] != 'complete' or validation['manifest_sha256'] != inputs.hashes['manifest.json']
            or not validation['all_reference_optima_match']):
        raise ValueError('A completed independently audited study is required')
    for relative, expected in manifest['input_hashes'].items():
        if digest(inputs.file(relative)) != expected:
            raise ValueError(f'Frozen input hash mismatch: {relative}')
    for relative, expected in manifest['source_hashes'].items():
        if digest(inputs.file('source/'+relative)) != expected:
            raise ValueError(f'Frozen source hash mismatch: {relative}')
    return manifest


def verify_assignment(table, ids, expected, req, poses, masks):
    if ids is None:
        if expected is not None:
            raise ValueError('Missing assignment with a reported objective')
        return
    assessment = req.evaluate(poses[ids], masks[ids].sum(axis=0), sensor_masks=masks[ids])
    if not assessment.feasible or abs(assessment.objective_score-expected) > 1e-7 or not table.feasible[tuple(ids)]:
        raise ValueError('Counterfactual witness disagrees with the shared evaluator')


def candidate_response(cid, other, masks, costs, weights):
    unique = int(np.count_nonzero(masks[cid] & ~masks[other]))
    overlap = int(np.count_nonzero(masks[cid] & masks[other]))
    coefficient = weights.get('coverage_weight', 1)+weights.get('importance_weight', 0)
    return dict(standalone_covered_count=int(masks[cid].sum()), unique_count=unique,
        overlap_count=overlap, marginal_j=float(coefficient*unique-weights.get('overlap_penalty', 0)*overlap
            - weights.get('sensor_count_penalty', 0)-weights.get('installation_cost_weight', 0)*costs[cid]))


def omitted_rows(witness, pool, source, source_scores, canonical, masks, table, weights, identity):
    if witness.get('witness_ids') is None:
        return []
    source_count = len(source)
    source_pool = set(i % source_count for i in pool)
    output = []
    for cid in witness['missing_ids']:
        sid = cid % source_count
        other = next(i for i in witness['witness_ids'] if i != cid)
        site = np.flatnonzero(np.all(source[:, :2] == source[sid, :2], axis=1))
        kept = [i for i in site if int(i) in source_pool]
        mode_kept = any(np.array_equal(source[i, [2, 3]], source[sid, [2, 3]]) for i in kept)
        source_order = sorted(site.tolist(), key=lambda i: (-source_scores[i], i))
        target_site = site + (cid//source_count)*source_count
        valid_site = [int(i) for i in target_site if table.feasible[i, other]]
        target_rank = 1+sum(table.scores[i, other] > table.scores[cid, other]+TOL for i in valid_site)
        kept_target = [int(i+(cid//source_count)*source_count) for i in kept]
        eligible_kept = [i for i in kept_target if table.feasible[i, other]]
        context_shortfall = (float(table.scores[cid, other]-max(table.scores[i, other] for i in eligible_kept))
                             if eligible_kept else None)
        reason = ('omitted_pitch_in_retained_height_yaw_mode' if mode_kept else 'omitted_height_yaw_mode')
        if identity['pool'] == 'top':
            reason = 'below_source_score_cutoff'
        output.append(dict(**identity, candidate_id=cid, source_candidate_id=sid,
            station_m=float(canonical[cid, 0]), side='left' if canonical[cid, 1] > 0 else 'right',
            height_m=float(canonical[cid, 2]), relative_yaw_deg=float(canonical[cid, 3]),
            pitch_deg=float(canonical[cid, 4]), reason=reason,
            source_site_rank=1+sum(source_scores[i] > source_scores[sid]+TOL for i in site),
            source_site_id_tiebroken_rank=source_order.index(sid)+1,
            target_site_rank_in_optimal_partner_context=int(target_rank),
            optimal_partner_id=other, kept_same_site_context_shortfall_j=context_shortfall,
            **candidate_response(cid, other, masks, table.costs, weights)))
    return output


def response_difference(first, second, masks, table, interface, weights):
    a, b = masks[first].sum(axis=0), masks[second].sum(axis=0)
    delta_covered = int(np.count_nonzero(b)-np.count_nonzero(a))
    delta_double = int(np.count_nonzero(b == 2)-np.count_nonzero(a == 2))
    delta_cost = float(table.costs[second].sum()-table.costs[first].sum())
    changes = dict(covered_count_delta=delta_covered, double_count_delta=delta_double,
        installation_cost_delta=delta_cost,
        coverage_term=(weights.get('coverage_weight', 1)+weights.get('importance_weight', 0))*delta_covered,
        overlap_term=-weights.get('overlap_penalty', 0)*delta_double,
        cost_term=-weights.get('installation_cost_weight', 0)*delta_cost)
    changes['total_j_delta'] = sum(changes[k] for k in ('coverage_term', 'overlap_term', 'cost_term'))
    for label, use in [('interface', interface), ('outside_interface', ~interface)]:
        changes[label] = dict(newly_covered=int(np.count_nonzero((a == 0) & (b > 0) & use)),
            lost_coverage=int(np.count_nonzero((a > 0) & (b == 0) & use)),
            double_count_delta=int(np.count_nonzero((b == 2) & use)-np.count_nonzero((a == 2) & use)))
    return changes


def analyze_case(inputs, job, base, source, source_scores, source_pools):
    relative = Path(job['directory'])
    cm = inputs.json(relative/'manifest.json')
    if digest(inputs.file(relative/'geometry.npz')) != cm['geometry_sha256']:
        raise ValueError('Original geometry hash mismatch')
    profile = inputs.json(relative/'corridor_profile.json')
    if digest(inputs.file(relative/'corridor_profile.json')) != cm['profile_sha256']:
        raise ValueError('Original profile hash mismatch')
    spec = inputs.json(relative/'requirements.json')
    geometry = inputs.json(relative/'geometry.json')
    with np.load(inputs.file(relative/'geometry.npz'), allow_pickle=False) as data:
        poses, canonical, modules = data['poses'], data['canonical_candidates'], data['module_ids']
        targets, coordinates, allowed = data['targets'], data['target_coordinates'], data['allowed_candidates']
        pools = {kind: data[kind+'_pool'].tolist() for kind in ('top', 'diverse')}
    for kind, values in pools.items():
        expected = source_pools[kind]+[i+len(source) for i in source_pools[kind]]
        if values != expected:
            raise ValueError('Transferred pool IDs disagree with frozen source selection')
    with np.load(inputs.file(relative/'reference_masks.npz'), allow_pickle=False) as data:
        masks = data['masks']
    req = CorridorRequirements(spec, targets, coordinates, canonical, poses, profile,
        length_m=geometry['length_m'], interface_half_length_m=base['interface_half_length_m'])
    if not np.array_equal(allowed, req.allowed):
        raise ValueError('Original eligibility mask mismatch')
    table = pair_table(poses, masks, spec, allowed, req.interface_mask, profile['interface'])
    all_ids = np.arange(len(poses))
    global_best = best_assignment(table, all_ids, all_ids)
    results = {method: inputs.json(relative/method/'result.json') for method in METHODS}
    for method, result in results.items():
        ids = result['selected_ids']
        if method.startswith('reference_') and result['reference_masks_sha256'] != digest(inputs.file(relative/'reference_masks.npz')):
            raise ValueError('Original reference mask hash mismatch')
        if ids is None:
            continue
        archive = inputs.file(relative/method/'deployment.npz')
        if digest(archive) != result['deployment_sha256']:
            raise ValueError('Original deployment hash mismatch')
        with np.load(archive, allow_pickle=False) as data:
            if not all(np.array_equal(data[k], v) for k, v in [('candidate_ids', ids), ('poses', poses[ids]),
                    ('canonical', canonical[ids]), ('sensor_masks', masks[ids]), ('density', masks[ids].sum(axis=0))]):
                raise ValueError('Original deployment arrays disagree')
        if (abs(table.scores[tuple(ids)]-result['assessment']['objective_score']) > 1e-7
                or bool(table.feasible[tuple(ids)]) != result['assessment']['feasible']):
            raise ValueError('Pair table disagrees with original saved deployment')
    reported_global = results['reference_global']['assessment']
    if (global_best['score'] is None) != (reported_global is None) or (reported_global is not None
            and abs(global_best['score']-reported_global['objective_score']) > 1e-7):
        raise ValueError('New full enumeration disagrees with the audited global optimum')
    detail = dict(job=job, global_optimum=global_best, methods={})
    rows, omissions = [], []
    identity = dict(profile=job['profile'], target=job.get('target_name', str(job['angle_deg'])),
        angle_deg=job['angle_deg'], road_width_m=geometry['road_width_m'])
    for kind in ('top', 'diverse'):
        row = dict(**identity, pool=kind, status='complete', raw_gap_j=None, ls_gap_j=None,
            quota_loss_j=None, pool_omission_loss_j=None, min_target_candidate_additions=None,
            distinct_missing_source_ids_in_witness=None, optimal_pair_count=None,
            fixed_side_loss_j=None, anchor_loss_j=None, disconnected_loss_j=None,
            local_search_loss_j=None, one_agent_relocation_gain_j=None, minimum_objective_dip=None)
        omitted = omission_witness(table, pools[kind], modules)
        raw, refined = results[kind+'_joint'], results[kind+'_joint_ls']
        if omitted['restricted']['score'] is not None:
            if raw['assessment'] is None or abs(omitted['restricted']['score']-raw['assessment']['objective_score']) > 1e-7:
                raise ValueError('New pool enumeration disagrees with the audited restricted optimum')
        elif raw['assessment'] is not None:
            raise ValueError('Reported feasible pool but full enumeration finds none')
        method = dict(omission=omitted)
        if global_best['score'] is None or raw['assessment'] is None or refined['assessment'] is None:
            row['status'] = 'not_diagnosed_no_feasible_reference_or_initialization'
        else:
            initial, final = refined['initialization']['selected_ids'], refined['selected_ids']
            if initial != raw['selected_ids']:
                raise ValueError('LS initialization differs from the independently selected pool solution')
            search = search_diagnostic(table, canonical, initial, final, base['local_station_radius_m'])
            values = [global_best['score'], search['fixed_side_optimum']['score'], search['anchor_optimum']['score'],
                      search['feasible_component_optimum']['score'], refined['assessment']['objective_score']]
            if any(a < b-1e-7 for a, b in zip(values, values[1:])):
                raise ValueError('Nested domain objective ordering failed')
            losses = [max(0., a-b) for a, b in zip(values, values[1:])]
            for key in ('fixed_side_optimum', 'anchor_optimum', 'feasible_component_optimum', 'one_agent_relocation_optimum'):
                value = search[key]
                verify_assignment(table, value['selected_ids'], value['score'], req, poses, masks)
            for pair, value in zip(search['feasible_component_optimum']['witness_path'],
                                   search['feasible_component_optimum']['witness_path_scores']):
                verify_assignment(table, pair, value, req, poses, masks)
            if (refined['stopping_reason'] == 'no_improving_one_coordinate_move'
                    and search['one_coordinate_best_j'] > values[-1]+1e-7):
                raise ValueError('Saved LS stopping reason is false')
            rejected = Counter()
            for pair in search['rejected_coordinate_moves']:
                a = req.evaluate(poses[pair], masks[pair].sum(axis=0), sensor_masks=masks[pair])
                if a.feasible:
                    raise ValueError('Pair table incorrectly rejected a coordinate move')
                rejected.update(a.violations)
            search['rejected_coordinate_reasons'] = dict(rejected)
            witness_ids = omitted['witness_ids']
            verify_assignment(table, witness_ids, omitted['reference']['score'], req, poses, masks)
            difference = response_difference(final, witness_ids, masks, table, req.interface_mask, spec.get('weights', {}))
            if abs(difference['total_j_delta']-(omitted['reference']['score']-values[-1])) > 1e-7:
                raise ValueError('Coverage/overlap/cost response decomposition failed')
            row.update(raw_gap_j=values[0]-raw['assessment']['objective_score'], ls_gap_j=values[0]-values[-1],
                quota_loss_j=values[0]-omitted['reference']['score'],
                pool_omission_loss_j=omitted['reference']['score']-raw['assessment']['objective_score'],
                min_target_candidate_additions=omitted['minimum_target_candidate_additions'],
                distinct_missing_source_ids_in_witness=len({i % len(source) for i in omitted['missing_ids']}),
                optimal_pair_count=omitted['optimal_pair_count'],
                **dict(zip(('fixed_side_loss_j', 'anchor_loss_j', 'disconnected_loss_j', 'local_search_loss_j'), losses)),
                one_agent_relocation_gain_j=search['one_agent_relocation_optimum']['score']-values[-1],
                minimum_objective_dip=search['feasible_component_optimum']['minimum_objective_dip'])
            if abs(sum(losses)-row['ls_gap_j']) > 1e-7:
                raise ValueError('Loss decomposition does not sum to the full objective shortfall')
            omissions.extend(omitted_rows(omitted, pools[kind], source, source_scores, canonical,
                masks, table, spec.get('weights', {}), {**identity, 'pool': kind}))
            method.update(search=search, ls_to_quota_optimum_response=difference)
        rows.append(row)
        detail['methods'][kind] = method
    return detail, rows, omissions


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(output, rows, omissions):
    valid = [r for r in rows if r['status'] == 'complete']
    lines = ['# Candidate Omissions and Local-Search Barriers', '',
        'Offline hindsight analysis of stored masks. No new raycasting, policy training or candidate-selection rule.',
        'Original experiment files are unchanged. Counterfactual answers are NOT an online method or a speed benchmark.', '',
        '| Pool | Cases | Raw misses | LS misses | Side restriction | Anchor restriction | Disconnected states | Search loss |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for kind in ('top', 'diverse'):
        use = [r for r in valid if r['pool'] == kind]
        counts = [sum(r[key] > 1e-7 for r in use) for key in ('raw_gap_j', 'ls_gap_j',
            'fixed_side_loss_j', 'anchor_loss_j', 'disconnected_loss_j', 'local_search_loss_j')]
        lines.append(f"| {kind} | {len(use)} | " + ' | '.join(map(str, counts))+' |')
    lines.extend(['', 'Loss columns count conditions with a positive contribution; they are not mutually exclusive.',
        'The exact nested decomposition is global -> fixed sides -> original anchor radius -> feasible action component -> saved LS.',
        'The component permits worsening/plateau moves but never an infeasible intermediate deployment.', '',
        '## Every Diverse+LS Miss', '',
        '| Profile | Target | J shortfall | Side | Anchor | Disconnect | Search | Best one-agent relocation gain | Minimum dip to component optimum |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|'])
    for row in valid:
        if row['pool'] != 'diverse' or row['ls_gap_j'] <= 1e-7:
            continue
        values = [row[k] for k in ('ls_gap_j', 'fixed_side_loss_j', 'anchor_loss_j', 'disconnected_loss_j',
            'local_search_loss_j', 'one_agent_relocation_gain_j', 'minimum_objective_dip')]
        lines.append(f"| {row['profile']} | {row['target']} | " + ' | '.join(f'{v:.3f}' for v in values)+' |')
    reasons = Counter(r['reason'] for r in omissions if r['pool'] == 'diverse')
    lines.extend(['', '## Omitted Candidate Witnesses', '',
        f'Diverse missing-witness categories: {dict(reasons)}.',
        'For every condition all optimal quota pairs are considered. The witness minimizes the NUMBER of missing target candidates;',
        'it is not necessarily unique, and does not minimize the number of distinct shared source IDs.',
        'Adding these candidates is a hindsight capacity diagnostic, not a same-budget algorithmic improvement.',
        'Source ranks are per station/side. Target ranks condition on the OTHER sensor of an optimal witness, not the saved deployment.',
        'See omitted_candidates.csv for source ranks, target conditional ranks, pose attributes and marginal responses.', '',
        '## Interpretation Limits', '',
        'Witness paths certify only this finite labelled action graph and the stored static masks.',
        'A necessary temporary dip is a J loss, not a constraint violation or a physical installation trajectory.',
        'Best one-agent relocation may change several pose coordinates at once; it is not an existing LS action.',
        'Different optimal pairs may give different coverage/overlap/cost tradeoffs at equal J.',
        'These observed maps are development data for any future rule. No MAPPO superiority follows.', ''])
    (output/'REPORT.md').write_text('\n'.join(lines))


def plot(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    use = [r for r in rows if r['status'] == 'complete' and r['ls_gap_j'] > 1e-7]
    if not use:
        return
    fig, ax = plt.subplots(figsize=(12, max(4, .45*len(use))), constrained_layout=True)
    y, left = np.arange(len(use)), np.zeros(len(use))
    for key, label, color in [('fixed_side_loss_j', 'Fixed installation side', '#777777'),
            ('anchor_loss_j', 'Original station radius', '#ce6847'),
            ('disconnected_loss_j', 'Feasible-path disconnection', '#5e8e68'),
            ('local_search_loss_j', 'Search within feasible component', '#198597')]:
        values = np.array([r[key] for r in use])
        ax.barh(y, values, left=left, label=label, color=color)
        left += values
    ax.set_yticks(y, [f"{r['pool']} / {r['profile']} / w{r['road_width_m']:g}, {r['angle_deg']:g} deg" for r in use], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('J shortfall decomposition within each condition (not coverage percentage points)')
    ax.set_title('All nonoptimal LS outputs: restrictions versus search barriers')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2, fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.2)
    fig.savefig(output/'loss_decomposition.png', dpi=160)
    plt.close(fig)


def generate(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError('Diagnostics require a separate output directory')
    if output.exists():
        raise FileExistsError(output)
    inputs = Inputs(root)
    manifest = checked_manifest(inputs)
    base = inputs.json('inputs/base_suite.json')
    pools = inputs.json('source_pools.json')
    if 'source_pools_sha256' in manifest and digest(inputs.file('source_pools.json')) != manifest['source_pools_sha256']:
        raise ValueError('Frozen source pool hash mismatch')
    metadata = inputs.json('library/metadata.json')
    archive = inputs.file('library/library.npz')
    if digest(archive) != metadata['source_pose_mask_hash']:
        raise ValueError('Source archive hash mismatch')
    with np.load(archive, allow_pickle=False) as data:
        source, source_scores = data['canonical'], data['scores']
    output.mkdir(parents=True, exist_ok=False)
    code_hashes = {name: digest(BASE/name) for name in CODE}
    for name in CODE:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE/name, target)
    state = dict(status='running', scope='Offline saved-mask hindsight diagnostic; no training or raycasts',
        python=platform.python_version(), numpy=np.__version__, input_root=str(root), conditions=len(manifest['jobs']))
    save(output/'manifest.json', state)
    started = time.perf_counter()
    rows, omissions, details = [], [], []
    try:
        for job in manifest['jobs']:
            detail, values, missing = analyze_case(inputs, job, base, source, source_scores, pools['pools'])
            details.append(detail)
            rows.extend(values)
            omissions.extend(missing)
            print(f"Diagnosed {len(details)}/{len(manifest['jobs'])}: {job['profile']} {job.get('target_name', job['angle_deg'])}", flush=True)
        write_csv(output/'summary.csv', rows)
        write_csv(output/'omitted_candidates.csv', omissions)
        save(output/'case_diagnostics.json', details)
        report(output, rows, omissions)
        plot(output, rows)
        inputs.unchanged()
        if any(digest(BASE/p) != h for p, h in code_hashes.items()):
            raise ValueError('Diagnostic implementation changed during execution')
        save(output/'provenance.json', dict(input_hashes=inputs.hashes, code_hashes=code_hashes,
            source_pools=pools, scope=state['scope']))
        state.update(status='complete', diagnostic_rows=len(rows), omitted_witness_rows=len(omissions),
            original_input_files_checked=len(inputs.hashes), elapsed_sec=time.perf_counter()-started,
            verification='All original deployment scores/feasibility; full and pool optima; shared-evaluator counterfactual witnesses and paths; nested loss and response decompositions')
        save(output/'manifest.json', state)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        save(output/'manifest.json', state)
        raise
    return output


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE/'results/connected_constraint_holdout_20261021')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest = checked_manifest(Inputs(root))
    if not args.execute:
        print(json.dumps(dict(conditions=len(manifest['jobs']), diagnostic_rows=2*len(manifest['jobs']),
            scope='Stored masks only; no raycasting or training', execute=False, writes_files=False), indent=2))
        return
    print(generate(root, args.output or root/'failure_diagnostics_20261022'))


if __name__ == '__main__':
    main()
