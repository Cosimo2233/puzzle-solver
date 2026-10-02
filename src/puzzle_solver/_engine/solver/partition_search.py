"""依据边链和角点对应关系反向解析 2～4 片矩形分割。

本模块不做自由位姿碰撞搜索。每一步只允许两个子装配沿一对反向、近似
全覆盖的连续边链合并；因此完整候选天然带有一棵可复核的分割解析树。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import atan2, degrees
from time import perf_counter
from typing import Any, Callable

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import snap, substring, unary_union

from ..transforms import compose_transform, estimate_rigid, transform_points
from .preprocess import ProcessedPiece


@dataclass(frozen=True)
class EdgeChainEvidence:
    """一次反向合并中两条有向边链的对应证据。"""

    left_members: tuple[str, ...]
    right_members: tuple[str, ...]
    left_edge_count: int
    right_edge_count: int
    left_length_mm: float
    right_length_mm: float
    endpoint_residual_mm: float
    polyline_residual_mm: float
    turn_residual_deg: float
    left_chain: tuple[tuple[float, float], ...]
    right_chain_aligned: tuple[tuple[float, float], ...]

    def as_json(self) -> dict[str, Any]:
        """输出稳定的诊断结构。"""

        return {
            "left_members": list(self.left_members),
            "right_members": list(self.right_members),
            "left_edge_count": self.left_edge_count,
            "right_edge_count": self.right_edge_count,
            "left_length_mm": self.left_length_mm,
            "right_length_mm": self.right_length_mm,
            "endpoint_residual_mm": self.endpoint_residual_mm,
            "polyline_residual_mm": self.polyline_residual_mm,
            "turn_residual_deg": self.turn_residual_deg,
            "left_chain": [list(point) for point in self.left_chain],
            "right_chain_aligned": [
                list(point) for point in self.right_chain_aligned
            ],
        }


@dataclass
class PartitionComponent:
    """已经由若干叶片反向合并出的一个子区域。"""

    poses: dict[str, tuple[float, np.ndarray, np.ndarray]]
    shape: Polygon
    evidence: tuple[EdgeChainEvidence, ...]
    cost: float

    @property
    def members(self) -> tuple[str, ...]:
        """返回排序后的叶片编号。"""

        return tuple(sorted(self.poses))


@dataclass(frozen=True)
class PartitionSearchResult:
    """一次完整分割解析的结果。"""

    poses: dict[str, tuple[float, np.ndarray, np.ndarray]]
    evidence: tuple[EdgeChainEvidence, ...]
    cost: float
    topology_template: str = "recursive_partition"


def _boundary_vertices(shape: Polygon) -> np.ndarray:
    """去除重复和近共线点，得到稳定的逆时针外边界。"""

    simplified = orient(shape.simplify(0.15, preserve_topology=True), sign=1.0)
    points = np.asarray(simplified.exterior.coords[:-1], dtype=float)
    if len(points) < 3:
        return np.asarray(shape.exterior.coords[:-1], dtype=float)
    return points


def _chains(points: np.ndarray) -> list[np.ndarray]:
    """枚举外边界上所有非闭合连续有向边链。"""

    result: list[np.ndarray] = []
    count = len(points)
    for start in range(count):
        for edge_count in range(1, count):
            indices = [
                (start + offset) % count
                for offset in range(edge_count + 1)
            ]
            chain = points[indices]
            if float(LineString(chain).length) >= 5.0:
                result.append(chain)
    return result


def _component_chains(
    component: PartitionComponent, preserve_leaf_edges: bool
) -> list[np.ndarray]:
    """返回合成边链及保留原始身份的叶片单边。

    后者专门承载 T 结点的一对多闭合；合成并集可能删除边内结点，但叶片
    原始边绝不能因此丢失。重复几何在这里去重，不影响其离散解释。
    """

    chains = _chains(_boundary_vertices(component.shape))
    if preserve_leaf_edges:
        for _piece_id, (_angle, _translation, vertices) in component.poses.items():
            for index in range(len(vertices)):
                chains.append(np.asarray(
                    (vertices[index], vertices[(index + 1) % len(vertices)]),
                    dtype=float,
                ))
    unique: list[np.ndarray] = []
    signatures: set[tuple[Any, ...]] = set()
    for chain in chains:
        signature = tuple(
            (round(float(point[0]), 3), round(float(point[1]), 3))
            for point in chain
        )
        if signature not in signatures:
            signatures.add(signature)
            unique.append(chain)
    return unique


def _turns(chain: np.ndarray) -> np.ndarray:
    """返回边链内部顶点的有符号转角。"""

    values: list[float] = []
    for index in range(1, len(chain) - 1):
        incoming = chain[index] - chain[index - 1]
        outgoing = chain[index + 1] - chain[index]
        cross = float(
            incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        )
        dot = float(np.dot(incoming, outgoing))
        values.append(degrees(atan2(cross, dot)))
    return np.asarray(values, dtype=float)


def _turn_residual(first: np.ndarray, second_reversed: np.ndarray) -> float:
    """比较两条已同向边链的角点结构；允许共线边被视觉拆分。"""

    first_turns = _turns(first)
    second_turns = _turns(second_reversed)
    first_turns = first_turns[np.abs(first_turns) >= 4.0]
    second_turns = second_turns[np.abs(second_turns) >= 4.0]
    if len(first_turns) != len(second_turns):
        return 45.0 + 15.0 * abs(len(first_turns) - len(second_turns))
    if not len(first_turns):
        return 0.0
    return float(np.sqrt(np.mean(np.square(first_turns - second_turns))))


def _endpoint_sourced_variants(
    first: np.ndarray, second: np.ndarray, tolerance_mm: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成整链匹配及由短边端点定义的长直边端部区间。

    只允许从长边的首端或尾端截取，不允许在边内任意滑动。截点由另一
    条物理边的长度和端点共同确定，未使用的区间必须由后续拓扑闭合。
    """

    result = [(first, second)]
    if len(first) != 2 or len(second) != 2:
        return result
    first_length = float(LineString(first).length)
    second_length = float(LineString(second).length)
    difference = abs(first_length - second_length)
    if difference <= max(tolerance_mm * 1.8, 0.07 * max(first_length, second_length)):
        return result
    short_length = min(first_length, second_length)
    long_length = max(first_length, second_length)
    if short_length / max(long_length, 1e-9) < 0.18:
        return result
    if first_length > second_length:
        line = LineString(first)
        result.extend((
            (np.asarray(substring(line, 0.0, short_length).coords), second),
            (
                np.asarray(substring(line, long_length - short_length, long_length).coords),
                second,
            ),
        ))
    else:
        line = LineString(second)
        result.extend((
            (first, np.asarray(substring(line, 0.0, short_length).coords)),
            (
                first,
                np.asarray(substring(line, long_length - short_length, long_length).coords),
            ),
        ))
    return result


def _transform_component(
    component: PartitionComponent,
    angle: float,
    translation: np.ndarray,
) -> tuple[dict[str, tuple[float, np.ndarray, np.ndarray]], Polygon]:
    """把整个右子装配作为刚体变换到左子装配坐标系。"""

    poses: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for piece_id, (old_angle, old_translation, vertices) in component.poses.items():
        new_angle, new_translation = compose_transform(
            old_angle, old_translation, angle, translation
        )
        poses[piece_id] = (
            float(new_angle),
            np.asarray(new_translation, dtype=float),
            transform_points(vertices, angle, translation),
        )
    boundary = transform_points(_boundary_vertices(component.shape), angle, translation)
    return poses, Polygon(boundary)


def _repair_union(first: Polygon, second: Polygon, tolerance_mm: float) -> Polygon | None:
    """以仅用于后续边链枚举的小闭运算修复亚毫米观测缝隙。"""

    # 刚体估计的浮点尾差会让本来完全重合的接缝成为 MultiPolygon。先把
    # 右侧边界吸附到左侧真实顶点，再做并集；buffer 只处理真实观测缝隙。
    first = first if first.is_valid else first.buffer(0)
    second = second if second.is_valid else second.buffer(0)
    if first.geom_type != "Polygon" or second.geom_type != "Polygon":
        return None
    snapped_second = snap(second, first, min(tolerance_mm, 1.0))
    if not snapped_second.is_valid:
        snapped_second = snapped_second.buffer(0)
    try:
        merged = unary_union((first, snapped_second))
    except Exception:
        return None
    if merged.geom_type == "Polygon":
        return merged
    if first.distance(second) > tolerance_mm:
        return None
    radius = max(0.15, tolerance_mm * 0.55)
    repaired = unary_union((first.buffer(radius), second.buffer(radius))).buffer(-radius)
    if repaired.geom_type != "Polygon" or repaired.is_empty:
        return None
    return repaired


def _merge_options(
    left: PartitionComponent,
    right: PartitionComponent,
    tolerance_mm: float,
    maximum_options: int,
    *,
    allow_endpoint_partials: bool,
    preserve_leaf_edges: bool,
    deadline: float | None = None,
) -> list[PartitionComponent]:
    """按反向边链对应关系生成两个子装配的有限合并选项。"""

    options: list[PartitionComponent] = []
    signatures: set[tuple[Any, ...]] = set()
    for raw_left_chain in _component_chains(left, preserve_leaf_edges):
        if deadline is not None and perf_counter() >= deadline:
            break
        for raw_right_chain in _component_chains(right, preserve_leaf_edges):
          if deadline is not None and perf_counter() >= deadline:
            break
          variants = (
              _endpoint_sourced_variants(
                  raw_left_chain, raw_right_chain, tolerance_mm
              )
              if allow_endpoint_partials
              else [(raw_left_chain, raw_right_chain)]
          )
          for left_chain, right_chain in variants:
            if deadline is not None and perf_counter() >= deadline:
                break
            left_line = LineString(left_chain)
            left_length = float(left_line.length)
            left_chord = float(np.linalg.norm(left_chain[-1] - left_chain[0]))
            right_line = LineString(right_chain)
            right_length = float(right_line.length)
            length_error = abs(left_length - right_length)
            length_limit = max(tolerance_mm * 1.8, 0.07 * max(left_length, right_length))
            if length_error > length_limit:
                continue
            right_chord = float(np.linalg.norm(right_chain[-1] - right_chain[0]))
            if abs(left_chord - right_chord) > length_limit:
                continue
            angle, translation, endpoint_rms = estimate_rigid(
                np.asarray((right_chain[0], right_chain[-1])),
                np.asarray((left_chain[-1], left_chain[0])),
            )
            aligned_right = transform_points(right_chain, angle, translation)[::-1]
            polyline_residual = float(
                max(
                    left_line.hausdorff_distance(LineString(aligned_right)),
                    LineString(aligned_right).hausdorff_distance(left_line),
                )
            )
            if polyline_residual > max(2.2, tolerance_mm * 1.35):
                continue
            turn_residual = _turn_residual(left_chain, aligned_right)
            if turn_residual > 28.0:
                continue
            right_poses, right_shape = _transform_component(
                right, angle, np.asarray(translation, dtype=float)
            )
            overlap = float(left.shape.intersection(right_shape).area)
            seam_length = min(left_length, right_length)
            if overlap > max(4.0, tolerance_mm * seam_length * 0.65):
                continue
            merged_shape = _repair_union(left.shape, right_shape, tolerance_mm)
            if merged_shape is None or not merged_shape.is_valid:
                continue
            expected_area = float(left.shape.area + right_shape.area)
            # 中间组件的并集仅用于继续枚举边链，不是最终物理几何。端点在
            # 观测误差内吸附时，会沿相邻长边形成面积约为“端点误差×接缝
            # 长度”的修复楔形；用固定 8 mm² 会把长接缝的正确解释误杀。
            # 最终候选仍使用 poses 中未吸附的原始刚体多边形检查面积和重叠。
            repair_area_loss_limit = max(
                8.0,
                1.2 * overlap,
                tolerance_mm * seam_length * 0.25,
            )
            if (
                float(merged_shape.area)
                < expected_area - repair_area_loss_limit
            ):
                continue
            poses = dict(left.poses)
            poses.update(right_poses)
            centers = tuple(
                sorted(
                    (
                        piece_id,
                        round(float(np.mean(vertices[:, 0])), 1),
                        round(float(np.mean(vertices[:, 1])), 1),
                    )
                    for piece_id, (_a, _t, vertices) in poses.items()
                )
            )
            if centers in signatures:
                continue
            signatures.add(centers)
            evidence = EdgeChainEvidence(
                left.members,
                right.members,
                len(left_chain) - 1,
                len(right_chain) - 1,
                left_length,
                right_length,
                float(endpoint_rms),
                polyline_residual,
                turn_residual,
                tuple(map(tuple, left_chain.tolist())),
                tuple(map(tuple, aligned_right.tolist())),
            )
            cost = (
                left.cost + right.cost + length_error
                + 2.0 * float(endpoint_rms)
                + 2.5 * polyline_residual
                + 0.04 * turn_residual
                + 0.2 * overlap
                + 0.05 * (
                    float(LineString(raw_left_chain).length) - left_length
                    + float(LineString(raw_right_chain).length) - right_length
                )
            )
            options.append(
                PartitionComponent(
                    poses,
                    merged_shape,
                    left.evidence + right.evidence + (evidence,),
                    float(cost),
                )
            )
    options.sort(key=lambda item: (item.cost, item.members))
    return options[:maximum_options]


def _state_signature(components: list[PartitionComponent]) -> tuple[Any, ...]:
    """构造保留叶片相对位姿、与整体平移无关的有限状态签名。"""

    rows = []
    for component in components:
        pose_rows = []
        ordered = sorted(component.poses.items())
        base_angle = float(ordered[0][1][0])
        centers = {
            piece_id: np.mean(pose[2], axis=0)
            for piece_id, pose in ordered
        }
        for piece_id, (angle, _translation, _vertices) in ordered:
            distances = tuple(
                round(float(np.linalg.norm(centers[piece_id] - centers[other_id])), 1)
                for other_id, _other_pose in ordered
            )
            pose_rows.append((
                piece_id,
                round((float(angle) - base_angle) % 360.0, 1),
                distances,
            ))
        rows.append((component.members, tuple(pose_rows)))
    return tuple(sorted(rows))


def _partition_signature(
    components: list[PartitionComponent],
) -> tuple[tuple[str, ...], ...]:
    """返回与位姿无关的成员分区，用于维持拓扑搜索多样性。"""

    return tuple(sorted(component.members for component in components))


def _prune_frontier_with_partition_diversity(
    states: list[list[PartitionComponent]],
    maximum_states: int,
) -> list[list[PartitionComponent]]:
    """按成员分区轮询保留状态，避免单一局部形状占满整个束宽。

    四片在第二次合并后既可能形成 ``2+2``，也可能形成 ``3+1``；二者
    都只是通用二叉分割树的中间状态，不代表预设答案模板。逐分区轮询可
    保证每一种仍有几何证据的邻接结构至少继续接受全局矩形检验。
    """

    if len(states) <= maximum_states:
        return sorted(states, key=_state_rank)
    groups: dict[
        tuple[tuple[str, ...], ...],
        list[list[PartitionComponent]],
    ] = {}
    for state in states:
        groups.setdefault(_partition_signature(state), []).append(state)
    for group in groups.values():
        group.sort(key=_state_rank)
    ordered_groups = sorted(
        groups.values(),
        key=lambda group: _state_rank(group[0]),
    )
    retained: list[list[PartitionComponent]] = []
    layer = 0
    while len(retained) < maximum_states:
        added = False
        for group in ordered_groups:
            if layer < len(group):
                retained.append(group[layer])
                added = True
                if len(retained) >= maximum_states:
                    break
        if not added:
            break
        layer += 1
    return sorted(retained, key=_state_rank)


def _rectangle_rank(component: PartitionComponent) -> tuple[float, ...]:
    """用旋转不变的赛题矩形约束排序完整解析，不负责安全准入。"""

    physical_union = unary_union([
        Polygon(pose[2]) for pose in component.poses.values()
    ])
    rectangle = physical_union.minimum_rotated_rectangle
    coordinates = np.asarray(rectangle.exterior.coords[:-1], dtype=float)
    sides = sorted(
        float(np.linalg.norm(coordinates[(index + 1) % 4] - coordinates[index]))
        for index in range(4)
    )
    short_side = float(np.mean(sides[:2]))
    long_side = float(np.mean(sides[2:]))
    dimension_violation = (
        max(0.0, 90.0 - long_side)
        + max(0.0, long_side - 120.0)
        + max(0.0, 50.0 - short_side)
        + max(0.0, short_side - 90.0)
    )
    iou = float(
        physical_union.intersection(rectangle).area
        / max(physical_union.union(rectangle).area, 1e-9)
    )
    return dimension_violation, 1.0 - iou, component.cost


def _state_rank(state: list[PartitionComponent]) -> tuple[float, ...]:
    """优先保留可闭合的紧凑子装配，并强优先最终合法矩形。"""

    if len(state) == 1:
        return _rectangle_rank(state[0])
    compactness = sum(
        1.0 - float(
            component.shape.area
            / max(component.shape.convex_hull.area, 1e-9)
        )
        for component in state
    )
    rectangle_sprawl = sum(
        1.0 - float(
            component.shape.area
            / max(component.shape.minimum_rotated_rectangle.area, 1e-9)
        )
        for component in state
    )
    return compactness, rectangle_sprawl, sum(item.cost for item in state)


def _enumerate_recursive_partition_candidates(
    pieces: list[ProcessedPiece],
    *,
    tolerance_mm: float,
    deadline: float | None,
    maximum_results: int,
    maximum_merge_options: int = 18,
    maximum_states_per_depth: int = 120,
    on_state: Callable[[], bool] | None = None,
    allow_endpoint_partials: bool = True,
    preserve_leaf_edges: bool = True,
    preserve_partition_diversity: bool = False,
    _run_strict_probe: bool = True,
) -> tuple[list[PartitionSearchResult], dict[str, Any]]:
    """枚举所有 2～4 片二叉反向合并树及其边链对应解释。"""

    started = perf_counter()
    strict_probe_applicable = (
        len(pieces) < 4
        or all(len(piece.vertices) == 4 for piece in pieces)
    )
    if (
        allow_endpoint_partials
        and _run_strict_probe
        and strict_probe_applicable
    ):
        strict_deadline = (
            None if deadline is None
            else min(deadline, perf_counter() + 5.0)
        )
        strict_results, strict_diagnostics = _enumerate_recursive_partition_candidates(
            pieces,
            tolerance_mm=tolerance_mm,
            deadline=strict_deadline,
            maximum_results=maximum_results,
            maximum_merge_options=24,
            maximum_states_per_depth=64,
            on_state=on_state,
            allow_endpoint_partials=False,
            preserve_leaf_edges=False,
            preserve_partition_diversity=preserve_partition_diversity,
            _run_strict_probe=False,
        )
        exact_strict: list[PartitionSearchResult] = []
        for result in strict_results:
            strict_rank = _rectangle_rank(PartitionComponent(
                result.poses,
                unary_union([
                    Polygon(pose[2]) for pose in result.poses.values()
                ]),
                result.evidence,
                result.cost,
            ))
            if strict_rank[0] <= 1.0 and strict_rank[1] <= 0.025:
                exact_strict.append(result)
        if exact_strict:
            strict_diagnostics["partition_search_phase"] = "strict_edge_chains"
            strict_diagnostics["partition_search_strict_probe_succeeded"] = True
            return exact_strict[:maximum_results], strict_diagnostics
    initial = [
        PartitionComponent(
            {
                piece.piece_id: (
                    0.0,
                    np.zeros(2, dtype=float),
                    piece.vertices.copy(),
                )
            },
            Polygon(piece.vertices),
            (),
            0.0,
        )
        for piece in pieces
    ]
    frontier: list[list[PartitionComponent]] = [initial]
    generated = 0
    expanded = 0
    depth_counts: list[int] = []
    for _depth in range(len(pieces) - 1):
        next_frontier: list[list[PartitionComponent]] = []
        exact_final_frontier: list[list[PartitionComponent]] = []
        seen: set[tuple[Any, ...]] = set()
        stop_depth = False
        exact_result_target = (
            2
            if preserve_partition_diversity and len(pieces) == 4
            else 1
        )
        for state in frontier:
            if deadline is not None and perf_counter() >= deadline:
                break
            if on_state is not None and not on_state():
                break
            expanded += 1
            for left_index, right_index in combinations(range(len(state)), 2):
                left = state[left_index]
                right = state[right_index]
                for merged in _merge_options(
                    left,
                    right,
                    tolerance_mm,
                    maximum_merge_options,
                    allow_endpoint_partials=allow_endpoint_partials,
                    preserve_leaf_edges=preserve_leaf_edges,
                    deadline=deadline,
                ):
                    generated += 1
                    remainder = [
                        component
                        for index, component in enumerate(state)
                        if index not in (left_index, right_index)
                    ]
                    new_state = remainder + [merged]
                    signature = _state_signature(new_state)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    next_frontier.append(new_state)
                    if len(new_state) == 1:
                        rectangle_rank = _rectangle_rank(new_state[0])
                        if rectangle_rank[0] <= 1.0 and rectangle_rank[1] <= 0.025:
                            exact_final_frontier.append(new_state)
                            if len(exact_final_frontier) >= exact_result_target:
                                stop_depth = True
                                break
                if stop_depth:
                    break
            if stop_depth:
                break
        if exact_final_frontier:
            next_frontier = exact_final_frontier
        if preserve_partition_diversity and len(pieces) == 4:
            frontier = _prune_frontier_with_partition_diversity(
                next_frontier,
                maximum_states_per_depth,
            )
        else:
            next_frontier.sort(key=_state_rank)
            frontier = next_frontier[:maximum_states_per_depth]
        depth_counts.append(len(frontier))
        if not frontier:
            break
    results = [
        PartitionSearchResult(
            state[0].poses,
            state[0].evidence,
            state[0].cost,
            "recursive_partition",
        )
        for state in frontier
        if len(state) == 1
    ]
    results.sort(key=lambda item: _rectangle_rank(PartitionComponent(
        item.poses,
        unary_union([Polygon(pose[2]) for pose in item.poses.values()]),
        item.evidence,
        item.cost,
    )))
    return results[:maximum_results], {
        "partition_search_generated_states": generated,
        "partition_search_expanded_states": expanded,
        "partition_search_depth_counts": depth_counts,
        "partition_search_elapsed_seconds": perf_counter() - started,
        "partition_search_policy": "directed_edge_chain_reverse_partition",
        "partition_search_tolerance_mm": tolerance_mm,
        "partition_search_phase": (
            "endpoint_sourced_t_junctions"
            if allow_endpoint_partials else "strict_edge_chains"
        ),
        "partition_search_strict_probe_succeeded": False,
    }


def enumerate_two_piece_candidates(
    pieces: list[ProcessedPiece],
    **kwargs: Any,
) -> tuple[list[PartitionSearchResult], dict[str, Any]]:
    """仅枚举两片完整直边或折线接口，不开放 T 结点搜索空间。"""

    results, diagnostics = _enumerate_recursive_partition_candidates(
        pieces,
        allow_endpoint_partials=False,
        preserve_leaf_edges=False,
        _run_strict_probe=False,
        **kwargs,
    )
    diagnostics["piece_count_branch"] = "two_piece_edge_chain"
    return results, diagnostics


def enumerate_three_piece_candidates(
    pieces: list[ProcessedPiece],
    **kwargs: Any,
) -> tuple[list[PartitionSearchResult], dict[str, Any]]:
    """枚举三片条带、单 T 结点和三向中心的紧凑搜索空间。"""

    results, diagnostics = _enumerate_recursive_partition_candidates(
        pieces,
        allow_endpoint_partials=True,
        preserve_leaf_edges=True,
        _run_strict_probe=True,
        **kwargs,
    )
    diagnostics["piece_count_branch"] = "three_piece_topology"
    return results, diagnostics


def enumerate_partition_candidates(
    pieces: list[ProcessedPiece],
    **kwargs: Any,
) -> tuple[list[PartitionSearchResult], dict[str, Any]]:
    """按碎片数量分派互不共享搜索预算的 2/3/4 片解析器。"""

    count = len(pieces)
    if count == 2:
        return enumerate_two_piece_candidates(pieces, **kwargs)
    if count == 3:
        return enumerate_three_piece_candidates(pieces, **kwargs)
    if count == 4:
        from .four_piece_topology import enumerate_four_piece_candidates

        return enumerate_four_piece_candidates(pieces, **kwargs)
    raise ValueError("随机拼图只支持 2～4 片")
