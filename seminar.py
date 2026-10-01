"""Reproducible preliminary benchmark and seminar artifacts.

Run: .venv/bin/python seminar.py run --root results/seminar_20260919
Report only: .venv/bin/python seminar.py report --root results/seminar_20260919
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from src.environment import MapGeometry, evaluate_coverage, get_scenario, load_config
from src.visualize import export_result_figures

METHODS = ['GREEDY', 'MILP_BUDGET', 'MAPPO']
COLORS = ['#008578', '#3559a8', '#c74865']
SCENARIOS = [0, 1, 2]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def worker(root: Path, sid: int) -> None:
    from src.methods.greedy import run_greedy_budget
    from src.methods.milp import solve_budget_frontier
    from src.methods.mappo import plan_mappo_runs, run_mappo
    import torch

    torch.set_num_threads(1)
    config = load_config(root / 'config.yaml')
    budgets = config.k_values
    scenario = get_scenario(sid)
    timings = []
    for method, function in [('GREEDY', run_greedy_budget), ('MILP_BUDGET', solve_budget_frontier)]:
        start = time.perf_counter()
        print(f'START {method} scenario={sid}', flush=True)
        records = function(scenario, config.lidar_profile, config, budgets, output_tag=root.name)
        timings.append(dict(method=method, scenario_id=sid, K=None, trial=None,
                            seconds=time.perf_counter()-start,
                            scope=f'scenario frontier K={",".join(map(str, budgets))} including candidate build'))
        write_json(root / f'timing_s{sid}.json', timings)
        print(f'FINISHED {method}: {len(records)} results', flush=True)
    rows = plan_mappo_runs([scenario], config.lidar_profile, budgets, config, tag=root.name)
    for row in rows:
        print(f'START MAPPO scenario={sid} K={row.budget_k} seed={row.seed}', flush=True)
        start = time.perf_counter()
        run_mappo([row], config)
        timings.append(dict(method='MAPPO', scenario_id=sid, K=row.budget_k, trial=row.trial,
                            seconds=time.perf_counter()-start, scope='full training, evaluation, yaw tuning and saving'))
        write_json(root / f'timing_s{sid}.json', timings)
    print(f'COMPLETE scenario={sid}', flush=True)


def run(root: Path, steps: int, trials: int, budgets: list[int], time_limit: float | None) -> None:
    if root.exists():
        raise FileExistsError(f'Use a new output root: {root}')
    root.mkdir(parents=True)
    config = yaml.safe_load(Path('configs/experiment.yaml').read_text())
    config.update(scenario_ids=SCENARIOS, k_values=budgets, experiment_tag=root.name)
    config['mappo'].update(episodes=steps, trials=trials)
    if time_limit is not None:
        config['solver']['time_limit_sec'] = time_limit
    config['output']['results_dir'] = str(root)
    (root / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    sources = [Path('main.py'), Path('seminar.py'), *Path('src').rglob('*.py')]
    write_json(root / 'manifest.json', dict(
        command=sys.argv, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        python=sys.version, platform=platform.platform(), scenarios=SCENARIOS, budgets=budgets,
        steps=steps, trials=trials, milp_time_limit_sec=config['solver']['time_limit_sec'], status='running',
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        caveats=['Preliminary: top-10 MILP/GREEDY versus full-orientation MAPPO',
                 'Static geometric coverage; no real object detection or dynamic traffic',
                 'Runtime under concurrent scenario workers; shared visibility cache',
                 'MAPPO includes best-pose selection and coordinate yaw postprocessing']))
    (root / 'packages.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    (root / 'source.patch').write_text(subprocess.check_output(['git', 'diff', '--', 'src', 'tests'], text=True))
    processes = []
    for sid in SCENARIOS:
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(sid), 'OMP_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}
        log = (root / f'scenario_{sid}.log').open('w')
        p = subprocess.Popen([sys.executable, __file__, 'worker', '--root', str(root), '--sid', str(sid)],
                             stdout=log, stderr=subprocess.STDOUT, env=env)
        processes.append((sid, p, log))
    failures = []
    while processes:
        for sid, process, log in processes[:]:
            code = process.poll()
            if code is not None:
                log.close()
                processes.remove((sid, process, log))
                print(f'Worker scenario={sid} exit={code}', flush=True)
                if code:
                    failures.append(sid)
        if processes:
            time.sleep(2)
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest.update(status='failed' if failures else 'runs_complete', failed_scenarios=failures,
                    finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    write_json(root / 'manifest.json', manifest)
    if failures:
        raise RuntimeError(f'Failed scenarios {failures}; see scenario logs')
    report(root)
    manifest['status'] = 'complete'
    write_json(root / 'manifest.json', manifest)


def interactive_scene(root: Path, record: dict, evaluation, geometry, positions: np.ndarray) -> Path:
    import plotly.graph_objects as go

    mesh = geometry.create_environment_mesh()
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    facecolors = ['rgb(%d,%d,%d)' % tuple((colors[f].mean(axis=0)*255).astype(int)) for f in faces]
    fig = go.Figure(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], facecolor=facecolors,
                            opacity=0.28, name='Environment', hoverinfo='skip'))
    for label, mask, color in [('Covered targets', evaluation.covered_mask, '#009e73'),
                               ('Uncovered targets', ~evaluation.covered_mask, '#d55e00')]:
        pts = evaluation.target_points[mask]
        if len(pts) > 6000:
            pts = pts[np.linspace(0, len(pts)-1, 6000).astype(int)]
        fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers',
                                  marker=dict(size=2, color=color), name=label))
    cells = np.arange(geometry.grid_size/2, geometry.map_len, geometry.grid_size)
    install = np.array([(x, y, 0.3) for x in cells for y in cells if geometry.is_installable(x, y)])
    fig.add_trace(go.Scatter3d(x=install[:, 0], y=install[:, 1], z=install[:, 2], mode='markers',
                              marker=dict(size=2, color='#0072b2'), name='Installable cells'))
    fig.add_trace(go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode='markers+text',
                              marker=dict(size=7, color='#cc0077'), text=[f'L{i+1}' for i in range(len(positions))],
                              customdata=positions[:, 3:5], name='LiDAR poses',
                              hovertemplate='x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<br>yaw=%{customdata[0]:.1f}, pitch=%{customdata[1]:.1f}'))
    for x, y, z, yaw, pitch in positions:
        ya, pi = np.deg2rad([yaw, pitch])
        end = np.array([x, y, z]) + 5*np.array([np.cos(pi)*np.cos(ya), np.cos(pi)*np.sin(ya), np.sin(pi)])
        fig.add_trace(go.Scatter3d(x=[x, end[0]], y=[y, end[1]], z=[z, end[2]], mode='lines',
                                  line=dict(color='#cc0077', width=5), showlegend=False, hoverinfo='skip'))
    name = f"s{record['scenario_id']}_k{record['budget_k']}_{record['method'].lower()}"
    fig.update_layout(title=f"{name} | coverage {record['coverage_pct']:.2f}% | trial {record.get('trial')}",
                      scene=dict(aspectmode='data', camera=dict(eye=dict(x=1.8, y=1.8, z=1.5)),
                                 xaxis_title='x [m]', yaxis_title='y [m]', zaxis_title='z [m]'),
                      margin=dict(l=0, r=0, b=0, t=55), legend=dict(bgcolor='rgba(255,255,255,0.85)'))
    path = root / 'interactive_3d' / f'{name}.html'
    path.parent.mkdir(exist_ok=True)
    fig.write_html(path, include_plotlyjs='directory', full_html=True)
    return path


def report(root: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pptx import Presentation
    from pptx.util import Inches, Pt

    config = load_config(root / 'config.yaml')
    budgets = config.k_values
    k_max = max(budgets)
    k_label = ','.join(map(str, budgets))
    records = []
    paths = {}
    for method in ['greedy', 'milp', 'mappo']:
        for path in sorted((root / method).glob('*.json')):
            record = json.loads(path.read_text())
            key = (record['scenario_id'], record['budget_k'], record['method'], record.get('trial'))
            if key in paths:
                raise ValueError(f'Duplicate result: {key}')
            paths[key] = path
            records.append(record)
    expected = {(s, k, m, t) for s in SCENARIOS for k in budgets for m in METHODS
                for t in (range(config.trials) if m == 'MAPPO' else [None])}
    if set(paths) != expected:
        raise ValueError(f'Missing/unexpected result combinations: {set(paths)^expected}')
    for record in records:
        positions = np.load(record['positions_npy'])
        scenario = get_scenario(record['scenario_id'])
        evaluation = evaluate_coverage(positions, scenario, record['lidar_profile'], config)
        if evaluation.covered_count != record['covered_count'] or not np.isclose(evaluation.coverage_pct, record['coverage_pct']):
            raise ValueError(f'Coverage verification failed: {record}')
        if len(positions) > record['budget_k'] or not np.isfinite(positions).all():
            raise ValueError('Invalid poses')
    (root / 'tables').mkdir(exist_ok=True)
    df = pd.DataFrame(records)
    df.drop(columns=['extra']).to_csv(root / 'tables' / 'raw_results.csv', index=False)
    summary = df.groupby(['scenario_id', 'budget_k', 'method']).coverage_pct.agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    summary.to_csv(root / 'tables' / 'coverage_summary.csv', index=False)
    times = [v for p in sorted(root.glob('timing_s*.json')) for v in json.loads(p.read_text())]
    pd.DataFrame(times).to_csv(root / 'tables' / 'runtime.csv', index=False)
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for sid, ax in zip(SCENARIOS, axes):
        for method, color in zip(METHODS, COLORS):
            rows = summary[(summary.scenario_id == sid) & (summary.method == method)].sort_values('budget_k')
            ax.errorbar(rows.budget_k, rows['mean'], yerr=rows['std'].fillna(0), label=method,
                        color=color, marker='o', capsize=4, linewidth=2)
        ax.set(title=f'S{sid}: {get_scenario(sid).topology}', xlabel='LiDAR budget K', xticks=budgets, ylim=(0, 105))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel('Coverage [%]')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Preliminary comparison | MAPPO mean +/- sample SD | candidate domains differ', fontsize=12)
    fig.tight_layout()
    fig.savefig(figures / 'coverage_comparison.png', dpi=200)
    fig.savefig(figures / 'coverage_comparison.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for sid, ax in zip(SCENARIOS, axes):
        for k, color in zip(budgets, plt.cm.viridis(np.linspace(0, 0.9, len(budgets)))):
            curves = []
            for record in records:
                if record['method'] == 'MAPPO' and record['scenario_id'] == sid and record['budget_k'] == k:
                    with np.load(record['train_log_npz']) as data:
                        cov = data['coverage']
                        if len(cov) != config.episodes or not np.isfinite(cov).all():
                            raise ValueError('Incomplete/nonfinite training log')
                        curves.append(np.maximum.accumulate(cov))
            curves = np.array(curves)
            x = np.arange(1, curves.shape[1]+1)
            ax.plot(x, curves.mean(axis=0), color=color, label=f'K={k}')
            ax.fill_between(x, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.12)
        ax.set(title=f'S{sid}', xlabel='Joint environment steps', ylim=(0, 105))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel('Best stochastic rollout coverage [%]')
    axes[-1].legend()
    fig.suptitle('MAPPO search history | mean and seed range | excludes evaluation/yaw tuning')
    fig.tight_layout()
    fig.savefig(figures / 'learning_curves.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    costs = pd.DataFrame(times).groupby(['scenario_id', 'method']).seconds.sum()
    x = np.arange(len(SCENARIOS))
    for j, (method, color) in enumerate(zip(METHODS, COLORS)):
        ax.bar(x + (j-1)*0.25, [costs.loc[(s, method)] for s in SCENARIOS], width=0.24, color=color, label=method)
    ax.set(xticks=x, xticklabels=[f'S{s}' for s in SCENARIOS], ylabel='Measured worker seconds',
           title='Total experiment cost per scenario | not equal-budget speed comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    fig.text(0.5, 0.01, f'Baselines: K={k_label} frontier; MAPPO: K={k_label} x {config.trials} seeds; concurrent workers, cached visibility',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figures / 'runtime.png', dpi=200)
    plt.close(fig)

    representatives = []
    for sid in SCENARIOS:
        for k in budgets:
            for method in METHODS:
                candidates = [r for r in records if (r['scenario_id'], r['budget_k'], r['method']) == (sid, k, method)]
                avg = np.mean([r['coverage_pct'] for r in candidates])
                r = min(candidates, key=lambda r: (abs(r['coverage_pct']-avg), r.get('trial') or 0))
                path = paths[(sid, k, method, r.get('trial'))]
                exported = export_result_figures(path, root / 'config.yaml', figures, save_3d=True, output_3d_dir=root / '3d')
                evaluation = evaluate_coverage(np.load(r['positions_npy']), get_scenario(sid), config.lidar_profile, config)
                html = interactive_scene(root, r, evaluation, MapGeometry(get_scenario(sid)), np.load(r['positions_npy']))
                representatives.append(dict(scenario=sid, K=k, method=method, trial=r.get('trial'),
                                            snapshot=str(exported[2]), interactive=str(html)))
    write_json(root / 'representatives.json', representatives)
    links = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>LiDAR seminar results</title>',
             '<body style="font-family:sans-serif;max-width:1100px;margin:32px auto;padding:16px">',
             '<h1>LiDAR 배치 예비 평가</h1><p><a href="SEMINAR.pptx">발표 PPT</a> | '
             '<a href="SEMINAR_REPORT.md">보고서</a> | <a href="tables/coverage_summary.csv">결과 CSV</a></p>',
             '<p>MAPPO 평균에 가장 가까운 trial을 대표 배치로 표시. 후보 공간 차이가 있는 예비 비교.</p>',
             '<img src="figures/coverage_comparison.png" alt="Coverage comparison" style="width:100%">',
             '<h2>회전 가능한 3D 결과</h2><table cellpadding="8"><tr><th>Scenario</th><th>K</th><th>Method</th><th>Trial</th><th>3D</th></tr>']
    for r in representatives:
        relative = Path(r['interactive']).relative_to(root)
        links.append(f'<tr><td>{r["scenario"]}</td><td>{r["K"]}</td><td>{r["method"]}</td><td>{r["trial"]}</td>'
                     f'<td><a href="{relative}">열기</a></td></tr>')
    links.append('</table></body></html>')
    (root / 'index.html').write_text('\n'.join(links), encoding='utf-8')

    report_lines = ['# LiDAR 배치 성능평가 세미나', '',
                    '## 실험 범위',
                    f'- 시나리오 0/1/2, K={k_label}, MAPPO {config.episodes:,} steps x {config.trials} seeds.',
                    '- 100m x 100m 정적 파라메트릭 맵, 도로 폭 7m, 30m LiDAR.',
                    '- 공통 raycasting/coverage 사용. MILP/GREEDY top-10, MAPPO 전체 orientation.',
                    '- 따라서 MILP는 MAPPO 탐색 공간의 전역 최적 상한이 아님.', '',
                    '## 공통 파이프라인과 평가식',
                    'scenario metadata -> parametric geometry -> LiDAR poses -> raycasting/FOV -> binary union coverage.',
                    '- pose = [x, y, z, yaw, pitch], roll 없음. 도로/건물을 제외한 설치 가능 pole line 사용.',
                    '- target: 0.5m XY cell, 높이 1.0/1.5/2.0m. 거리/FOV와 장애물 가림을 모두 검사.',
                    '- coverage = 100 * sum_j 1[sum_i visibility(i,j) >= 1] / N. 중복 관측은 한 번만 계산.',
                    '- GREEDY: 추가 coverage가 가장 큰 후보를 순차 선택.',
                    '- MILP: maximize sum(y_j), y_j <= sum_i a_ij*x_i, sum_i x_i <= K.',
                    '- MAPPO: 공유 actor와 중앙집중 critic, 11개 pose 이동 행동, scenario/K마다 새로 학습.',
                    '- 모든 최종 JSON/NPY를 공통 evaluator로 재검산한 후 이 보고서를 생성.', '',
                    '## 결과', '|Scenario|K|Method|Mean coverage %|Sample SD|N|', '|---|---|---|---|---|---|']
    for r in summary.to_dict('records'):
        sd = f"{r['std']:.3f}" if np.isfinite(r['std']) else '-'
        report_lines.append(f"|{r['scenario_id']}|{r['budget_k']}|{r['method']}|{r['mean']:.3f}|{sd}|{r['count']}|")
    differences = []
    for sid in SCENARIOS:
        row = summary[(summary.scenario_id == sid) & (summary.budget_k == k_max)].set_index('method')
        difference = row.loc['MAPPO', 'mean'] - row.loc['MILP_BUDGET', 'mean']
        differences.append(f"S{sid}, K={k_max}: MAPPO {row.loc['MAPPO', 'mean']:.2f}% / "
                           f"MILP {row.loc['MILP_BUDGET', 'mean']:.2f}% / GREEDY {row.loc['GREEDY', 'mean']:.2f}% "
                           f"(MAPPO-MILP {difference:+.2f} percentage points)")
    report_lines += ['', f'## K={k_max} 수치 요약', *['- '+line for line in differences]]
    report_lines += ['', '## MILP 종료 상태', '|Scenario|K|Status|Gap|Raw bound|', '|---|---|---|---|---|']
    for r in records:
        if r['method'] == 'MILP_BUDGET':
            report_lines.append(f"|{r['scenario_id']}|{r['budget_k']}|{r['solver_status']}|{r['mip_gap']}|"
                                f"{r['extra'].get('solver_best_bound')}|")
    report_lines += ['', '## 해석과 한계',
                     '- 최종 배치 coverage이며 검출 AP나 추적 정확도가 아님.',
                     '- MAPPO는 최고 관측 배치 선택과 yaw 후처리를 포함. 수렴/일반화 입증은 아님.',
                     '- 각 scenario/K에서 새로 학습. 속도에는 학습 비용을 포함해야 함.',
                     f'- runtime.csv의 MILP/GREEDY는 후보 생성 포함 K={k_label} 전체 frontier 시간.',
                     '- MAPPO 시간은 trial별 전체 run. 병렬 scenario 실행 및 기존 visibility cache 사용.',
                     '- MILP solver 종료 상태·gap·bound는 개별 JSON 확인. 제한시간 결과는 optimum이 아님.',
                     '- 3D 대표 MAPPO는 조합별 평균에 가장 가까운 trial. target는 화면에서 일부 샘플링, coverage는 전체 계산.',
                     '', '## 재생성', '```bash',
                     f'.venv/bin/python seminar.py report --root {root}', '```',
                     '실행 명령과 설정: manifest.json, config.yaml. 버전: packages.txt, source.patch.',
                     '3D: interactive_3d/*.html을 브라우저로 열어 회전. plotly.min.js를 같은 폴더에 유지.',
                     '기존 Matplotlib viewer도 config.yaml과 개별 결과 JSON을 지정하여 사용 가능.']
    (root / 'SEMINAR_REPORT.md').write_text('\n'.join(report_lines), encoding='utf-8')
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    def slide(title: str, lines=(), picture=None):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        box = s.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.8))
        box.text_frame.text = title
        box.text_frame.paragraphs[0].font.size = Pt(28)
        box.text_frame.paragraphs[0].font.name = 'Noto Sans CJK KR'
        if picture:
            from PIL import Image
            with Image.open(picture) as img:
                ratio = img.width/img.height
            width, height = 12.1, 5.6
            if width/height > ratio:
                width = height*ratio
            else:
                height = width/ratio
            s.shapes.add_picture(str(picture), Inches((13.333-width)/2), Inches(1.25), width=Inches(width), height=Inches(height))
        elif lines:
            tf = s.shapes.add_textbox(Inches(0.7), Inches(1.5), Inches(11.9), Inches(5.3)).text_frame
            tf.word_wrap = True
            for i, line in enumerate(lines):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text, p.font.size, p.space_after = line, Pt(23), Pt(20)
                p.font.name = 'Noto Sans CJK KR'
        foot = s.shapes.add_textbox(Inches(0.5), Inches(7.08), Inches(12), Inches(0.3))
        foot.text_frame.text = f'{root.name} | 예비 평가 | {len(prs.slides)}'
        foot.text_frame.paragraphs[0].font.size = Pt(10)

    slide('도로 토폴로지별 3D LiDAR 배치 성능평가', [
        'GREEDY / 시간 제한 MILP / 수정된 MAPPO',
        f'직선 · T자 · 십자 교차로 | K={k_label} | {config.trials} seeds',
        '연구 질문: 제한된 센서 수에서 관측 범위와 계산 비용은 어떻게 달라지는가?',
        '현재 범위: 정적 geometry coverage. 도시 규모 조합과 동적 가림은 후속 연구.'])
    slide('실험 환경과 비교 조건', [
        f'MAPPO: {config.episodes:,} joint steps / seed. PPO rollout v2.',
        'LiDAR: range 30m / horizontal 360° / vertical 26.9°',
        'XY 2m / Z 3~6m / yaw 5° / pitch 2.5°',
        f'MILP: {config.time_limit_sec:g}초 제한 / gap {config.mip_gap:g}',
        '후보 공간 차이: MILP/GREEDY top-10, MAPPO 전체 orientation.'])
    slide('공통 환경과 평가 지표', [
        '시나리오 메타데이터 → 도로/건물 geometry → 설치 후보 → raycasting/FOV',
        'pose = [x, y, z, yaw, pitch]. 인도 중심 설치선에 배치하며 roll은 없음.',
        '도로 target: XY 0.5m 간격, 높이 1.0 / 1.5 / 2.0m',
        'Coverage = 100 × 한 대 이상이 관측한 target 수 / 전체 target 수',
        '같은 target의 중복 관측은 한 번만 계산. 최종 저장 배치를 공통 함수로 재검산.'])
    slide('세 알고리즘의 역할', [
        'GREEDY: 미관측 target을 가장 많이 추가하는 후보를 K개 이하로 선택.',
        'MILP: 이진 후보 선택으로 union coverage 최대화. 시간 제한과 gap 적용.',
        'MAPPO: 공유 actor / 중앙집중 critic, 11개 행동으로 XY·Z·yaw·pitch 이동.',
        'MAPPO는 scenario/K별 재학습. 전이학습 또는 미지 도로 일반화 결과가 아님.',
        '목표: 최종 coverage와 계산 비용의 차이를 측정. 우월성을 미리 가정하지 않음.'])
    slide('Coverage 비교: MAPPO 평균과 표본 표준편차', picture=figures / 'coverage_comparison.png')
    slide(f'K={k_max} 결과 요약', differences + ['양의 차이는 현재 탐색 공간에서의 결과이며 최적 MILP를 초과했다는 뜻이 아님.'])
    slide('MAPPO 탐색 이력: 최고 rollout coverage', picture=figures / 'learning_curves.png')
    slide('전체 실험 소요 시간: 계산 범위에 주의', picture=figures / 'runtime.png')
    for sid in SCENARIOS:
        for method in METHODS:
            rep = next(r for r in representatives if r['scenario'] == sid and r['K'] == k_max and r['method'] == method)
            slide(f'S{sid} {get_scenario(sid).topology} | K={k_max} | {method} | trial={rep["trial"]}', picture=Path(rep['snapshot']))
    slide('해석 시 주의점', ['전체 coverage와 실제 검출·추적 성능은 다름.',
         'MAPPO: 학습 중 최고 배치 + yaw 후처리. 일반화 검증 전.',
         'MILP: 후보 제한 및 solver gap이 있으므로 전역 최적값으로 해석하지 않음.',
         '시간: 후보 생성 포함 frontier 비용과 trial별 학습 비용을 구분.',
         '다음 단계: 후보 공간 통제, 후처리 제거 실험, 학습 안정성 평가.'])
    prs.save(root / 'SEMINAR.pptx')
    print(f'REPORT COMPLETE: {root / "SEMINAR.pptx"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'worker', 'report'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--sid', type=int)
    parser.add_argument('--budgets', default='1,2,3', help='comma-separated K values')
    parser.add_argument('--time-limit', type=float, default=None, help='MILP seconds per K; default keeps YAML')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.root, args.steps, args.trials, [int(k) for k in args.budgets.split(',')], args.time_limit)
    elif args.command == 'worker':
        worker(args.root, args.sid)
    else:
        report(args.root)
