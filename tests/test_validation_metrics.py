"""用已知几何验证报告指标，避免把留缝和重叠误判为紧密矩形。"""
from dataclasses import replace
import pytest
from puzzle_solver import PuzzlePiece, PlanningOptions, PlannedMove, MovePlan
from puzzle_solver.commands.validate import independent_metrics


def sample(offset):
    vertices = ((0., 0.), (10., 0.), (10., 10.), (0., 10.))
    pieces = (PuzzlePiece('a', vertices), PuzzlePiece('b', vertices))
    moves = []
    for i, piece in enumerate(pieces):
        x = i * offset
        moves.append(PlannedMove(
            order=i+1, piece_id=piece.piece_id, pick_xy_a4_mm=(5., 5.),
            pick_clearance_mm=5., rotation_deg_clockwise=0.,
            place_xy_a4_mm=(5.+x, 155.), translation_a4_mm=(x, 150.),
            observed_vertices_a4_mm=vertices,
            target_vertices_a4_mm=tuple((px+x, py+150.) for px, py in vertices),
        ))
    return pieces, MovePlan('random', 1., False, 20., 10., (10., 155.), tuple(moves))


@pytest.mark.parametrize('offset,valid,tight,coverage', [(10., True, True, 1.), (15., True, False, .8), (8., False, False, 1.)])
def test_known_rectangles_gap_and_overlap(offset, valid, tight, coverage):
    pieces, plan = sample(offset)
    metrics = independent_metrics(pieces, plan)
    assert metrics['independent_checks_passed'] == valid
    assert metrics['tight_rectangle'] == tight
    assert metrics['rectangle_coverage'] == pytest.approx(coverage)


def test_incorrect_rigid_transform_is_rejected():
    pieces, plan = sample(10.)
    plan = replace(plan, moves=(replace(plan.moves[0], translation_a4_mm=(1., 150.)), plan.moves[1]))
    assert not independent_metrics(pieces, plan)['independent_checks_passed']
