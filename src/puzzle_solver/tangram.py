"""标准七巧板图案与无硬件副作用的刚体搬运规划，坐标单位为毫米。"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import math
from types import MappingProxyType

import numpy as np
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import polylabel
from scipy.optimize import Bounds, LinearConstraint, minimize

from .contracts import MovePlan, PlannedMove, PlanningError, PuzzlePiece

S = math.sqrt(2)
PATTERN_NAMES = MappingProxyType(dict(square='正方形', cat='猫', rabbit='兔', fish='鱼', boat='船', house='房子'))
# 精确顶点以整套正方形边长的 1/4 为单位。每行依次为两大、两小、中、方、平行。
_LAYOUTS = {
 'square': (
  ((0,0),(4,0),(2,2)), ((0,0),(2,2),(0,4)),
  ((3,1),(4,0),(4,2)), ((1,3),(3,3),(2,2)),
  ((4,2),(4,4),(2,4)), ((2,2),(3,1),(4,2),(3,3)),
  ((0,4),(1,3),(3,3),(2,4))),
 'cat': (
  ((0,0),(2*S,0),(0,2*S)), ((2*S,0),(2*S,2*S),(0,2*S)),
  ((S-1,-S-1),(S-1,-S-3),(S,-S-2)), ((S,-S-2),(S+1,-S-3),(S+1,-S-1)),
  ((0,0),(2*S,0),(S,-S)), ((S,-S),(S-1,-S-1),(S,-S-2),(S+1,-S-1)),
  ((0,2*S),(-1,2*S-1),(-1,2*S-3),(0,2*S-2))),
 'rabbit': (
  ((0,0),(2*S,0),(0,2*S)), ((2*S,0),(2*S,2*S),(0,2*S)),
  ((0,2*S),(-S,2*S),(0,S)), ((2*S,S),(3*S,S),(2*S,0)),
  ((0,0),(2,0),(0,-2)), ((0,-2-S),(S,-2-S),(S,-2),(0,-2)),
  ((0,-2-S),(1,-3-S),(3,-3-S),(2,-2-S))),
 'fish': (
  ((0,0),(2,-2),(2,2)), ((2,-2),(4,0),(2,2)),
  ((2,-2),(4,-2),(3,-3)), ((2,2),(4,2),(3,3)),
  ((0,0),(0,-2),(2,-2)), ((4,0),(5,-1),(6,0),(5,1)),
  ((0,0),(1,1),(1,3),(0,2))),
 'boat': (
  ((2,-4),(2,0),(4,-2)), ((2,0),(6,0),(4,-2)),
  ((0,0),(-S,0),(0,-S)), ((0,-S),(0,-2*S),(S,-S)),
  ((2,0),(4,0),(4,2)), ((0,0),(S,0),(S,-S),(0,-S)),
  ((0,0),(2,0),(3,1),(1,1))),
 'house': (
  ((-S,0),(S,-2*S),(S,0)), ((S,-2*S),(3*S,0),(S,0)),
  ((S,S),(0,2*S),(0,S)), ((2*S,S),(S,0),(2*S,0)),
  ((S,S),(2*S,2*S),(0,2*S)), ((0,0),(0,S),(S,S),(S,0)),
  ((S,0),(2*S,S),(2*S,2*S),(S,S))),
}
KINDS = ('large', 'large', 'small', 'small', 'medium', 'square', 'parallel')


def _vertices(poly):
    return np.asarray(orient(Polygon(poly), sign=1).exterior.coords[:-1], dtype=float)


@dataclass(frozen=True)
class TangramLayout:
    pattern_id: str
    side_mm: float
    pieces: tuple[tuple[tuple[float, float], ...], ...]


def get_tangram_layout(pattern_id: str, side_mm: float = 100.0) -> TangramLayout:
    if pattern_id not in _LAYOUTS:
        raise ValueError('未知七巧板图案')
    if isinstance(side_mm, bool) or not math.isfinite(side_mm) or side_mm <= 0:
        raise ValueError('side_mm 必须为正有限数')
    pieces = tuple(tuple(map(tuple, _vertices(p) * side_mm / 4)) for p in _LAYOUTS[pattern_id])
    # 不允许错误模板进入运动链路。
    polys = [Polygon(p) for p in pieces]
    expected = np.array([4,4,1,1,2,2,2]) * (side_mm/4)**2
    if not np.allclose([p.area for p in polys], expected, atol=1e-7):
        raise PlanningError(f'{pattern_id}: 模板面积错误')
    if any(a.intersection(b).area > 1e-7 for i,a in enumerate(polys) for b in polys[i+1:]):
        raise PlanningError(f'{pattern_id}: 模板内部重叠')
    return TangramLayout(pattern_id, side_mm, pieces)


@dataclass(frozen=True)
class TangramPlanningOptions:
    pattern_id: str = 'square'
    side_mm: float = 100.0
    source_bounds_mm: tuple[float, float, float, float] = (5.,175.,202.,287.)
    target_bounds_mm: tuple[float, float, float, float] = (5.,5.,202.,165.)
    vertex_tolerance_mm: float = 3.0
    minimum_pick_clearance_mm: float = 4.0
    minimum_source_gap_mm: float = 2.0
    assembly_gap_mm: float = 1.0
    allow_mirrored_pattern: bool = True
    start_xy_a4_mm: tuple[float,float] = (10.,10.)

    def __post_init__(self):
        get_tangram_layout(self.pattern_id, self.side_mm)
        for bounds in (self.source_bounds_mm, self.target_bounds_mm):
            if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
                raise ValueError('工作区必须为四个有限毫米数')
            x0,y0,x1,y1 = bounds
            if not (0 <= x0 < x1 <= 207 and 0 <= y0 < y1 <= 292):
                raise ValueError('工作区必须位于 207×292 mm 范围内')
        if box(*self.source_bounds_mm).intersects(box(*self.target_bounds_mm)):
            raise ValueError('取料区和拼装区必须分离')
        for name in ('vertex_tolerance_mm','minimum_pick_clearance_mm','minimum_source_gap_mm','assembly_gap_mm'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} 必须为正有限数')
        if type(self.allow_mirrored_pattern) is not bool:
            raise ValueError('allow_mirrored_pattern 必须为布尔值')
        if len(self.start_xy_a4_mm) != 2 or not all(math.isfinite(v) for v in self.start_xy_a4_mm):
            raise ValueError('起点必须为有限二维毫米坐标')


def _fits(source, target):
    """保持尺度和手性，枚举循环顶点对应关系，返回所有合适的旋转。"""
    if len(source) != len(target):
        return []
    results = []
    for shift in range(len(target)):
        b = np.roll(target, shift, axis=0)
        ac, bc = source.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((source-ac).T @ (b-bc))
        correction = np.eye(2)
        correction[-1,-1] = np.linalg.det(u @ vt)
        r = u @ correction @ vt
        t = bc-ac @ r
        residual = float(np.max(np.linalg.norm(source @ r+t-b, axis=1)))
        results.append((residual, r, t))
    return results


def classify_tangram_piece(vertices_mm, side_mm=100., tolerance_mm=1.5):
    """按完整顶点刚体残差分类；平行四边形两种手性都可作为观测。"""
    poly = Polygon(vertices_mm)
    if not poly.is_valid or poly.area <= 0 or poly.convex_hull.area-poly.area > 1e-5:
        raise PlanningError('碎片必须为有效凸多边形')
    observed = _vertices(vertices_mm)
    layout = get_tangram_layout('square', side_mm)
    matches = []
    for kind, p in zip(KINDS, layout.pieces):
        for mirror in (1,-1):
            target = _vertices(np.array(p)*[mirror,1])
            for residual, _, _ in _fits(observed, target):
                matches.append((residual,kind))
    residual, kind = min(matches, default=(math.inf,''))
    if residual > tolerance_mm:
        raise PlanningError(f'碎片尺寸或形状不符标准七巧板：最大顶点误差 {residual:.2f} mm')
    return kind


def _target_variants(options):
    layout = get_tangram_layout(options.pattern_id, options.side_mm)
    region = box(*options.target_bounds_mm)
    center = np.array(region.centroid.coords[0])
    for mirrored in ((False,True) if options.allow_mirrored_pattern else (False,)):
        raw = [np.array(p)*[-1 if mirrored else 1,1] for p in layout.pieces]
        # 仅平移碎片中心以留出装配缝；绝不缩放碎片。
        centroids = [np.array(Polygon(p).centroid.coords[0]) for p in raw]
        expansion = 1.
        for _ in range(100):
            spread = [p+c*(expansion-1) for p,c in zip(raw,centroids)]
            polygons = [Polygon(p) for p in spread]
            if all(a.distance(b) >= options.assembly_gap_mm-1e-8 for i,a in enumerate(polygons) for b in polygons[i+1:]):
                break
            expansion += .005
        else:
            raise PlanningError('无法生成带装配间隙的图案')
        for degrees in range(0,360,45):
            a = math.radians(degrees)
            r = np.array([[math.cos(a),math.sin(a)],[-math.sin(a),math.cos(a)]])
            rotated = [p @ r for p in spread]
            allpoints = np.concatenate(rotated)
            offset = center-(allpoints.min(axis=0)+allpoints.max(axis=0))/2
            targets = [_vertices(p+offset) for p in rotated]
            if all(region.covers(Polygon(p)) for p in targets):
                yield mirrored, targets


def _shortest_order(pick_xy, place_xy, start):
    """七点 Held–Karp 动态规划，避免在板端逐次枚举 5040 条路线。"""
    distances = np.linalg.norm(place_xy[:,None,:]-pick_xy[None,:,:],axis=2)
    states = {(1<<j,j):(float(np.linalg.norm(start-pick_xy[j])),(j,)) for j in range(7)}
    for mask in range(1,1<<7):
        for last in range(7):
            current = states.get((mask,last))
            if current is None:
                continue
            for nxt in range(7):
                if mask & (1<<nxt):
                    continue
                key = mask|(1<<nxt),nxt
                candidate = current[0]+distances[last,nxt],current[1]+(nxt,)
                if key not in states or candidate[0] < states[key][0]:
                    states[key] = candidate
    return min((cost+float(np.linalg.norm(place_xy[j]-start)),order)
               for (mask,j),(cost,order) in states.items() if mask == 127)


def _separate_actual_targets(choices, targets, options):
    """按模板的分离轴微调实测刚体位置；保持片型尺寸并留出真实装配间隙。"""
    actual = [c[4] for c in choices]
    polys = [Polygon(p) for p in actual]
    region = box(*options.target_bounds_mm)
    if (all(region.covers(p) for p in polys) and
        all(a.distance(b) >= options.assembly_gap_mm-1e-7 for i,a in enumerate(polys) for b in polys[i+1:])):
        return choices
    rows=[]; lower=[]
    for i in range(7):
        for j in range(i+1,7):
            axes=[]
            for p in (targets[i],targets[j]):
                edges=np.roll(p,-1,axis=0)-p
                for edge in edges:
                    axis=np.array((-edge[1],edge[0]))/np.linalg.norm(edge)
                    if np.dot(targets[j].mean(axis=0)-targets[i].mean(axis=0),axis)<0:axis=-axis
                    axes.append(axis)
            axis=max(axes,key=lambda n:np.min(targets[j]@n)-np.max(targets[i]@n))
            if np.min(targets[j]@axis)-np.max(targets[i]@axis)<-1e-6:
                return None
            row=np.zeros(14);row[2*i:2*i+2]=-axis;row[2*j:2*j+2]=axis
            rows.append(row)
            lower.append(options.assembly_gap_mm+np.max(actual[i]@axis)-np.min(actual[j]@axis))
    x0,y0,x1,y1=options.target_bounds_mm
    lows=[];highs=[]
    for p in actual:
        lows.extend(np.maximum(np.array((x0,y0))-p.min(axis=0),-options.vertex_tolerance_mm))
        highs.extend(np.minimum(np.array((x1,y1))-p.max(axis=0),options.vertex_tolerance_mm))
    if np.any(np.array(lows)>np.array(highs)):return None
    solution=minimize(lambda x:float(x@x),np.clip(np.zeros(14),lows,highs),jac=lambda x:2*x,
        bounds=Bounds(lows,highs),constraints=LinearConstraint(np.array(rows),np.array(lower),np.inf),
        method='SLSQP',options={'ftol':1e-9,'maxiter':80})
    if not solution.success:return None
    deltas=solution.x.reshape(7,2)
    shifted=[(i,e,r,t+delta,p+delta) for (i,e,r,t,p),delta in zip(choices,deltas)]
    polys=[Polygon(c[4]) for c in shifted]
    if (not all(region.buffer(1e-7).covers(p) for p in polys) or
        any(a.distance(b)<options.assembly_gap_mm-1e-6 for i,a in enumerate(polys) for b in polys[i+1:])):
        return None
    return shifted


def plan_tangram_moves(pieces, options=TangramPlanningOptions()):
    """返回恰好七个刚体移动；不支持翻面、占位中转或目标自动缩放。"""
    if not isinstance(options, TangramPlanningOptions):
        raise TypeError('options 必须为 TangramPlanningOptions')
    pieces = tuple(pieces)
    if len(pieces) != 7 or any(not isinstance(p, PuzzlePiece) for p in pieces):
        raise PlanningError('七巧板必须恰好输入七片 PuzzlePiece')
    if len({p.piece_id for p in pieces}) != 7:
        raise PlanningError('碎片 ID 不能重复')
    observed = [_vertices(p.vertices_mm) for p in pieces]
    polygons = [Polygon(p) for p in observed]
    source_region = box(*options.source_bounds_mm)
    if any(not source_region.covers(p) for p in polygons):
        raise PlanningError('请将七片全部放入取料区，并保持拼装区空白')
    if any(a.distance(b) < options.minimum_source_gap_mm for i,a in enumerate(polygons) for b in polygons[i+1:]):
        raise PlanningError('碎片接触、重叠或间隔不足，请分开放置')
    kinds = [classify_tangram_piece(p.vertices_mm,options.side_mm,options.vertex_tolerance_mm) for p in pieces]
    if sorted(kinds) != sorted(KINDS):
        raise PlanningError('片型不完整：需要两大、两小、中三角、正方形和平行四边形')
    picks = [polylabel(p, tolerance=.05) for p in polygons]
    clearances = [p.boundary.distance(q) for p,q in zip(polygons,picks)]
    if min(clearances) < options.minimum_pick_clearance_mm:
        raise PlanningError('碎片没有足够安全的吸取位置')
    assignments = list(product(*(list(permutations([i for i,k in enumerate(kinds) if k == kind]))
                                 for kind in ('large','small','medium','square','parallel'))))
    best = None
    for mirrored, targets in _target_variants(options):
        for groups in assignments:
            mapping = tuple(i for group in groups for i in group)
            choices = []
            for index, target in zip(mapping, targets):
                valid = [fit for fit in _fits(observed[index],target) if fit[0] <= options.vertex_tolerance_mm]
                if not valid:
                    break
                # 对称解只保留等价最短旋转；不使用反射矩阵。
                fit = min(valid,key=lambda f:(round(f[0],6), min(math.atan2(f[1][0,1],f[1][0,0]) % (2*math.pi),
                                                               -math.atan2(f[1][0,1],f[1][0,0]) % (2*math.pi))))
                residual,r,t = fit
                actual = observed[index] @ r+t
                choices.append((index,residual,r,t,actual))
            if len(choices) != 7:
                continue
            choices = _separate_actual_targets(choices,targets,options)
            if choices is None:
                continue
            # 动态规划遍历全部放置顺序。
            pick_xy = np.array([picks[c[0]].coords[0] for c in choices])
            place_xy = np.array([pick_xy[j] @ c[2]+c[3] for j,c in enumerate(choices)])
            start = np.array(options.start_xy_a4_mm)
            transfer = float(np.linalg.norm(place_xy-pick_xy,axis=1).sum())
            travel, order = _shortest_order(pick_xy,place_xy,start)
            cost = transfer+travel
            if best is None or cost < best[0]:
                best = cost,mirrored,choices,order,pick_xy,place_xy
    if best is None:
        raise PlanningError('图案放不下、需要翻面或实测轮廓发生重叠；请调整摆放或标定')
    _,mirrored,choices,order,pick_xy,place_xy = best
    moves=[]
    for sequence,j in enumerate(order,1):
        index,residual,r,t,actual = choices[j]
        moves.append(PlannedMove(sequence,pieces[index].piece_id,tuple(pick_xy[j]),clearances[index],
            math.degrees(math.atan2(r[0,1],r[0,0])) % 360,tuple(place_xy[j]),tuple(t),
            tuple(map(tuple,observed[index])),tuple(map(tuple,actual)),
            f'tangram:{options.pattern_id}:{"mirrored" if mirrored else "normal"}'))
    allpoints=np.concatenate([c[4] for c in choices]); low,high=allpoints.min(axis=0),allpoints.max(axis=0)
    return MovePlan('tangram',1-max(c[1] for c in choices)/(2*options.vertex_tolerance_mm),False,
                    float(high[0]-low[0]),float(high[1]-low[1]),tuple((low+high)/2),tuple(moves))
