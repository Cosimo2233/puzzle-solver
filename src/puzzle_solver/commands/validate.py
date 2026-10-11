"""可复现的算法测试与图形报告；不访问任何设备。"""
from __future__ import annotations

import hashlib
import html
import json
import logging
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import typer
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from puzzle_solver import PuzzlePiece, PlanningOptions, plan_sheet_moves

app = typer.Typer(add_completion=False)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
COLORS = ['#1588c7', '#e59228', '#28a780', '#a36cce']


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def independent_metrics(pieces, plan):
    moves = plan.moves
    polygons = [Polygon(m.target_vertices_a4_mm) for m in moves]
    union = unary_union(polygons)
    rectangle = union.minimum_rotated_rectangle
    overlap = sum(a.intersection(b).area for i, a in enumerate(polygons) for b in polygons[i+1:])
    gaps = [a.distance(b) for i, a in enumerate(polygons) for b in polygons[i+1:]]
    errors = []
    area_errors = []
    inputs = {p.piece_id: np.asarray(p.vertices_mm) for p in pieces}
    for m in moves:
        angle = np.deg2rad(m.rotation_deg_clockwise)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        predicted = inputs[m.piece_id] @ rotation.T + m.translation_a4_mm
        target = np.asarray(m.target_vertices_a4_mm)
        # Hausdorff distance allows equivalent cyclic vertex ordering.
        errors.append(Polygon(predicted).hausdorff_distance(Polygon(target)))
        pick = np.asarray(m.pick_xy_a4_mm) @ rotation.T + m.translation_a4_mm
        errors.append(float(np.linalg.norm(pick - m.place_xy_a4_mm)))
        area_errors.append(abs(Polygon(inputs[m.piece_id]).area - Polygon(target).area))
    coverage = union.area / rectangle.area
    complete = len(moves) == len(pieces) and {m.piece_id for m in moves} == set(inputs)
    valid = complete and all(p.is_valid and box(0, 0, 207, 292).covers(p) for p in polygons)
    valid = bool(valid and overlap <= 1e-4 and max(errors) <= 1e-5 and max(area_errors) <= 1e-4)
    return {
        'independent_checks_passed': valid,
        'overlap_mm2': overlap,
        'max_rigid_error_mm': max(errors),
        'max_area_error_mm2': max(area_errors),
        'minimum_piece_distance_mm': min(gaps),
        'rectangle_coverage': coverage,
        'rectangle_empty_area_mm2': rectangle.area - union.area,
        'tight_rectangle': bool(valid and coverage >= 0.995),
        'rectangle_vertices': list(rectangle.exterior.coords),
    }


def worker(spec_path):
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    out = spec_path.parent
    record = {**spec, 'status': 'generation_failed', 'hardware_executed': False}
    start = time.perf_counter()
    try:
        if spec.get('baseline'):
            payload = json.loads(Path(spec['baseline']).read_text(encoding='utf-8'))
            pieces = tuple(PuzzlePiece(p['id'], tuple(tuple(v) for v in p['vertices_mm'])) for p in payload['pieces'])
            truth = None
        else:
            from puzzle_solver._engine.generator.fixed_generator import fixed_pieces
            from puzzle_solver._engine.generator.random_generator import random_pieces
            from puzzle_solver._engine.generator.shuffle import shuffle_and_place
            rng = np.random.default_rng(spec['seed'])
            if spec['mode'] == 'fixed':
                generated = fixed_pieces()
            else:
                generated, _, _, metadata = random_pieces(
                    rng, piece_count=spec['count'], target_width=100, target_height=70,
                    generation_mode=spec['generator'], seed=spec['seed'], return_metadata=True)
                record['generator_metadata'] = metadata
            scattered = shuffle_and_place(generated, rng)
            pieces = tuple(PuzzlePiece(p.public_id, tuple(p.clean_vertices)) for p in scattered)
            truth = [{'id': p.public_id, 'vertices_mm': p.source_vertices} for p in scattered]
            payload = {'mode': spec['mode'], 'pieces': [{'id': p.piece_id, 'vertices_mm': p.vertices_mm} for p in pieces]}
        save(out/'input.json', payload)
        if truth is not None:
            save(out/'truth.json', truth)
        record['generation_ms'] = (time.perf_counter() - start) * 1000
        record['status'] = 'solver_failed'
        start = time.perf_counter()
        plan = plan_sheet_moves(pieces, PlanningOptions(mode=spec['mode']))
        record['solve_ms'] = (time.perf_counter() - start) * 1000
        save(out/'plan.json', plan.to_dict())
        record['metrics'] = independent_metrics(pieces, plan)
        record['status'] = 'passed' if record['metrics']['independent_checks_passed'] else 'geometry_check_failed'
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        if record['status'] == 'solver_failed':
            record['solve_ms'] = (time.perf_counter() - start) * 1000
    save(out/'result.json', record)


def draw_pieces(axis, pieces, title, colors, page=False):
    from matplotlib.patches import Polygon as Patch
    for p in pieces:
        vertices = np.asarray(p['vertices_mm'])
        axis.add_patch(Patch(vertices, facecolor=colors[p['id']], edgecolor='#172d40', alpha=0.8, linewidth=1.2))
        center = Polygon(vertices).representative_point()
        axis.text(center.x, center.y, p['id'], fontsize=8, ha='center', va='center')
    axis.autoscale_view()
    axis.margins(0.1)
    if page:
        axis.set_xlim(0, 207)
        axis.set_ylim(146, 0)
    else:
        axis.invert_yaxis()
    axis.set_aspect('equal', adjustable='box')
    axis.set_title(title, fontsize=12, fontweight='bold')
    axis.set_xlabel('x / mm')
    axis.set_ylabel('y / mm')
    axis.grid(alpha=0.15)


def plot_case(folder, record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    payload = json.loads((folder/'input.json').read_text(encoding='utf-8')) if (folder/'input.json').exists() else {'pieces': []}
    colors = {p['id']: COLORS[i] for i, p in enumerate(payload['pieces'])}
    draw_pieces(axes[0], payload['pieces'], '输入：旋转并分开放置', colors, page=True)
    if (folder/'plan.json').exists():
        plan = json.loads((folder/'plan.json').read_text(encoding='utf-8'))
        targets = [{'id': m['piece_id'], 'vertices_mm': m['target_vertices_a4_mm']} for m in plan['moves']]
        draw_pieces(axes[1], targets, '实际输出：保留算法原有间隙', colors)
        rect = np.asarray(record['metrics']['rectangle_vertices'])
        axes[1].plot(rect[:, 0], rect[:, 1], '--', color='#64748b', linewidth=1)
        for m in plan['moves']:
            x, y = m['place_xy_a4_mm']
            axes[1].plot(x, y, '+', color='black')
            axes[1].annotate(str(m['order']), (x, y), xytext=(4, 5), textcoords='offset points', fontsize=8)
        metric = record['metrics']
        axes[1].set_xlabel(f"矩形覆盖率 {metric['rectangle_coverage']:.1%} · 最小片间距 {metric['minimum_piece_distance_mm']:.2f} mm")
    else:
        axes[1].text(0.5, 0.5, '未返回可执行方案\n'+record['status'], ha='center', transform=axes[1].transAxes)
        axes[1].set_axis_off()
    if (folder/'truth.json').exists():
        truth = json.loads((folder/'truth.json').read_text(encoding='utf-8'))
        draw_pieces(axes[2], truth, '生成参考：不提供给求解器', colors)
    else:
        axes[2].set_axis_off()
        axes[2].text(0.05, 0.8, '历史基线输入\n没有单独记录生成真值\n\n十字：放置点\n数字：执行顺序\n虚线：最小外接矩形\n\n同色代表同一片纸', va='top', transform=axes[2].transAxes)
    status_text = record['status']
    if status_text == 'passed':
        status_text = '方案检查通过 / ' + ('紧密矩形' if record['metrics']['tight_rectangle'] else '非紧密矩形')
    fig.suptitle(f"{record['id']}   |   {status_text}   |   主机合成轮廓测试", fontsize=14)
    fig.savefig(folder/'result.png', dpi=140)
    plt.close(fig)


def report(output, records, provenance):
    import matplotlib.pyplot as plt
    groups = {}
    for label in ['fixed4', 'random2', 'random3', 'random4']:
        rows = [r for r in records if f"{r['mode']}{r['count']}" == label]
        successful = [r for r in rows if r['status'] == 'passed']
        groups[label] = {'total': len(rows), 'passed': len(successful),
                         'tight_rectangle': sum(r.get('metrics', {}).get('tight_rectangle', False) for r in rows),
                         'mean_solve_ms': float(np.mean([r['solve_ms'] for r in successful])) if successful else None}
    save(output/'summary.json', {'groups': groups, 'cases': records, 'provenance': provenance})
    selected = [next(r for r in records if r['mode'] == mode and r['count'] == n and not r.get('baseline')) for mode, n in [('fixed', 4), ('random', 2), ('random', 3), ('random', 4)]]
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), layout='constrained')
    for axis, record in zip(axes, selected):
        axis.imshow(plt.imread(output/'cases'/record['id']/'result.png'))
        axis.axis('off')
    fig.savefig(output/'overview.png', dpi=130)
    plt.close(fig)
    rows = ''.join(f'<tr><td><a href="#{r["id"]}">{r["id"]}</a></td><td>{r["status"]}</td><td>{r.get("solve_ms", 0):.0f}</td><td>{r.get("metrics", {}).get("rectangle_coverage", 0):.1%}</td></tr>' for r in records)
    cards = []
    for r in records:
        error = html.escape(r.get('error', ''))
        cards.append(f'<section id="{r["id"]}"><h2>{r["id"]}</h2><p>{html.escape(r["status"])} {error}</p><img src="cases/{r["id"]}/result.png"><p><a href="cases/{r["id"]}/result.json">指标与错误</a> · <a href="cases/{r["id"]}/input.json">输入</a></p></section>')
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>拼图算法测试</title>
<style>body{{font:16px system-ui;background:#f3f6fa;color:#172d40;margin:30px auto;max-width:1500px}}section,header{{background:white;padding:24px;margin:20px 0;border-radius:12px}}img{{width:100%}}td,th{{padding:8px 20px;text-align:left;border-bottom:1px solid #ddd}}a{{color:#0875b5}}pre{{white-space:pre-wrap}}</style>
<header><h1>固定与随机拼图：算法测试报告</h1><p>合成、无噪声、毫米轮廓输入；不涉及相机、PL 或电机。输入和真值分开保存，真值不参与求解。</p>
<p>passed 表示返回方案且通过独立刚体、面积、重叠及工作区检查；不等于紧密矩形。紧密矩形诊断要求最小外接矩形覆盖率 ≥99.5%。历史留缝会降低覆盖率，不能以可执行方案替代该指标。</p><pre>{html.escape(json.dumps(groups,ensure_ascii=False,indent=2))}</pre>
<p>每例独立子进程，生成与求解总预算 {provenance['timeout_seconds']} 秒；耗时为主机计算时间，不是板端速度。失败与超时均保留。</p></header>
<table><tr><th>样例</th><th>状态</th><th>求解 ms</th><th>矩形覆盖率</th></tr>{rows}</table>{''.join(cards)}</html>'''
    (output/'index.html').write_text(body, encoding='utf-8')


@app.command()
def main(
    output: Path = typer.Option(Path('build/algorithm-validation'), '--output', envvar='PUZZLE_VALIDATION_OUTPUT'),
    timeout: float = typer.Option(45.0, '--timeout', min=1),
    worker_spec: Path | None = typer.Option(None, '--worker-spec', hidden=True),
):
    """测试四个基线及二十个新样例，保存全部结果和图形报告。"""
    if worker_spec is not None:
        worker(worker_spec)
        return
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).resolve().parents[1]
    repo = package_root.parents[1]
    source_hash = hashlib.sha256()
    for p in sorted(package_root.rglob('*.py')):
        source_hash.update(p.relative_to(package_root).as_posix().encode())
        source_hash.update(p.read_bytes())
    specs = []
    for mode, n in [('fixed', 4), ('random', 2), ('random', 3), ('random', 4)]:
        label = f'{mode}{n}'
        specs.append({'id': f'{label}-baseline', 'mode': mode, 'count': n, 'baseline': str(repo/'baselines'/f'{label}.json')})
        for i, generator in enumerate(['template', 'recursive', 'arrangement', 'mixed', 'adversarial']):
            specs.append({'id': f'{label}-seed{260200+n*10+i}', 'mode': mode, 'count': n,
                          'seed': 260200+n*10+i, 'generator': 'fixed' if mode == 'fixed' else generator})
    records = []
    for spec in specs:
        folder = output/'cases'/spec['id']
        folder.mkdir(parents=True, exist_ok=True)
        if (folder/'result.json').exists():
            raise typer.BadParameter('输出目录已有结果，请选择新目录，避免混用历史数据。')
        save(folder/'spec.json', spec)
        with (folder/'worker.log').open('w', encoding='utf-8') as log:
            try:
                run = subprocess.run([sys.executable, '-m', 'puzzle_solver.commands.validate', '--worker-spec', str(folder/'spec.json')], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                if run.returncode:
                    save(folder/'result.json', {**spec, 'status': 'worker_failed', 'error': f'exit={run.returncode}'})
            except subprocess.TimeoutExpired:
                save(folder/'result.json', {**spec, 'status': 'timeout', 'error': f'生成与求解超过 {timeout} 秒，未返回方案'})
        record = json.loads((folder/'result.json').read_text(encoding='utf-8'))
        record['input_sha256'] = hashlib.sha256((folder/'input.json').read_bytes()).hexdigest() if (folder/'input.json').exists() else None
        save(folder/'result.json', record)
        plot_case(folder, record)
        records.append(record)
        logger.info('%s: %s', spec['id'], record['status'])
    provenance = {'platform': platform.platform(), 'python': sys.version, 'source_sha256': source_hash.hexdigest(),
                  'timeout_seconds': timeout, 'synthetic': True, 'noise_mm': 0, 'hardware_executed': False}
    report(output, records, provenance)
    logger.info('报告：%s', output/'index.html')


if __name__ == '__main__':
    app()
