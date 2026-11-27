"""Publish a small, path-free excerpt of a completed audit; never run experiments."""
from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parents[1]
TABLES = ('method_summary.csv', 'timing_summary.csv', 'paired_summary.csv',
          'map_balanced_summary.csv', 'paired_by_geometry.csv', 'feasibility.csv',
          'references.csv', 'infeasibility_proofs.csv', 'case_validation.csv')
VALIDATION_FIELDS = ('status', 'verification_status', 'created_at', 'counts',
                     'direct_raycast', 'rerun_optimization', 'raw_inputs_unchanged',
                     'wall_sec', 'script_sha256', 'limitations')
SUMMARY_FIELDS = ('reference_solver_status', 'reference_exact', 'reference_within_one_percent',
                  'maximum_reference_gap', 'online_above_reference_incumbent', 'actual_q')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(report):
    audit_path = report / 'validation_summary.json'
    audit = json.loads(audit_path.read_text())
    if (audit['status'] != 'passed' or not audit['direct_raycast'] or
            not audit['raw_inputs_unchanged'] or audit['rerun_optimization']):
        raise ValueError('Expected the completed, direct-raycast, read-only audit')
    excerpt = {key: audit[key] for key in VALIDATION_FIELDS}
    excerpt['summary'] = {key: audit['summary'][key] for key in SUMMARY_FIELDS}
    records = {}
    for name in TABLES:
        path = report / name
        if path.stat().st_size > 1024 * 1024:
            raise ValueError('Summary exceeds the 1 MiB per-table publication limit')
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError('Empty summary table: ' + name)
        if any('/home/' in str(v) or '/tmp/' in str(v) for r in rows for v in r.values()):
            raise ValueError('Machine-local path in publication table: ' + name)
        records[name] = dict(sha256=sha256(path), bytes=path.stat().st_size, rows=len(rows))
    provenance = dict(schema='research_summary_export_v1', source_report=report.name,
        source_validation_sha256=sha256(audit_path), audit_created_at=audit['created_at'],
        exporter_sha256=sha256(Path(__file__)), tables=records,
        scope='CSV files are byte-identical copies; validation JSON is a field-selected excerpt. '
              'No poses, masks, checkpoints, host paths or credentials. This export does not rerun verification.')
    return excerpt, provenance


def export(report, output, *, execute=False, check=False):
    if execute and check:
        raise ValueError('Choose export or check')
    excerpt, provenance = prepare(report)
    if check:
        for name, expected in provenance['tables'].items():
            if sha256(output / name) != expected['sha256']:
                raise ValueError('Published table differs: ' + name)
        if json.loads((output / 'validation_excerpt.json').read_text()) != excerpt:
            raise ValueError('Published validation excerpt differs')
        if json.loads((output / 'provenance.json').read_text()) != provenance:
            raise ValueError('Published provenance differs')
    elif execute:
        output.mkdir(parents=True, exist_ok=False)
        for name in TABLES:
            shutil.copyfile(report / name, output / name)
        for name, data in [('validation_excerpt.json', excerpt), ('provenance.json', provenance)]:
            (output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        export(report, output, check=True)
    return dict(status='checked' if check else 'exported' if execute else 'plan_only',
                tables=len(TABLES), table_bytes=sum(v['bytes'] for v in provenance['tables'].values()))


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path,
                        default=BASE / 'results/library_lowk_cost_review_20261127_final')
    parser.add_argument('--output', type=Path,
                        default=BASE / 'docs/research/lowk_cost_20261124')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(export(args.report, args.output, execute=args.execute, check=args.check), indent=2))


if __name__ == '__main__':
    main()
