import math
from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import Polygon
from puzzle_solver import (PATTERN_NAMES,PuzzlePiece,TangramPlanningOptions,get_tangram_layout,
                           plan_tangram_moves,classify_tangram_piece,PlanningError)


def scattered():
    a = 50*math.sqrt(2)
    b = 25*math.sqrt(2)
    polygons = (
        ((8,177),(8+a,177),(8,177+a)),
        ((88,177),(88+a,177),(88,177+a)),
        ((50,215),(50+b,215),(50+b,215+b)),
        ((164,181),(164+b,181),(164+b,181+b)),
        ((148,235),(198,235),(198,285)),
        ((10,250),(10+b,250),(10+b,250+b),(10,250+b)),
        ((48,260),(98,260),(123,285),(73,285)),
    )
    return tuple(PuzzlePiece(str(i),tuple(tuple(float(v) for v in p) for p in poly)) for i,poly in enumerate(polygons))


@pytest.mark.parametrize('name',PATTERN_NAMES)
def test_templates_use_exact_standard_pieces_without_overlap(name):
    layout=get_tangram_layout(name)
    polys=[Polygon(p) for p in layout.pieces]
    assert sum(p.area for p in polys) == pytest.approx(10000)
    assert [classify_tangram_piece(p) for p in layout.pieces] == ['large','large','small','small','medium','square','parallel']
    assert all(a.intersection(b).area < 1e-7 for i,a in enumerate(polys) for b in polys[i+1:])


@pytest.mark.parametrize('name',PATTERN_NAMES)
def test_seven_piece_rigid_plan_preserves_size_and_pick_location(name):
    pieces=scattered()
    plan=plan_tangram_moves(pieces,TangramPlanningOptions(pattern_id=name))
    assert plan.mode == 'tangram' and len(plan.moves)==7
    assert {m.piece_id for m in plan.moves} == {p.piece_id for p in pieces}
    for move in plan.moves:
        angle=math.radians(move.rotation_deg_clockwise)
        rotation=np.array([[math.cos(angle),math.sin(angle)],[-math.sin(angle),math.cos(angle)]])
        assert np.allclose(np.array(move.observed_vertices_a4_mm)@rotation+move.translation_a4_mm,move.target_vertices_a4_mm)
        assert np.allclose(np.array(move.pick_xy_a4_mm)@rotation+move.translation_a4_mm,move.place_xy_a4_mm)
        assert move.pick_clearance_mm >= 4
    polys=[Polygon(m.target_vertices_a4_mm) for m in plan.moves]
    assert all(a.intersection(b).area < 1e-6 for i,a in enumerate(polys) for b in polys[i+1:])


def test_missing_duplicate_touching_wrong_size_and_occupied_target_rejected():
    pieces=scattered()
    for invalid in (pieces[:-1],pieces[:-1]+(pieces[0],),
                    pieces[:-1]+(PuzzlePiece('x',pieces[0].vertices_mm),),
                    (PuzzlePiece('x',((10.,10.),(20.,10.),(10.,20.))),)+pieces[1:]):
        with pytest.raises(PlanningError):plan_tangram_moves(invalid)
    with pytest.raises(PlanningError):
        classify_tangram_piece(((0,0),(40,0),(0,40)))


def test_reflected_parallelogram_requires_mirrored_pattern():
    pieces=list(scattered())
    original=plan_tangram_moves(pieces)
    p=np.array(pieces[-1].vertices_mm)
    p[:,0]=171-p[:,0]
    pieces[-1]=PuzzlePiece('6',tuple(map(tuple,p)))
    # Both possible source chiralities remain within the same source bounding box.
    plan=plan_tangram_moves(pieces)
    assert plan.moves[0].template_id != original.moves[0].template_id
    if plan.moves[0].template_id.endswith(':normal'):
        pieces=list(scattered())
    with pytest.raises(PlanningError):
        plan_tangram_moves(pieces,TangramPlanningOptions(allow_mirrored_pattern=False))


def test_tiny_target_never_scales_pieces():
    with pytest.raises(PlanningError):
        plan_tangram_moves(scattered(),TangramPlanningOptions(target_bounds_mm=(5,5,50,50)))


@pytest.mark.parametrize('name',PATTERN_NAMES)
def test_real_white_pieces_keep_measured_shape_and_assembly_clearance(name):
    import json
    from pathlib import Path
    data=json.loads((Path(__file__).parent/'fixtures/tangram-white7-20261027.json').read_text())
    pieces=tuple(PuzzlePiece(str(i),tuple(map(tuple,p))) for i,p in enumerate(data['pieces']))
    options=TangramPlanningOptions(pattern_id=name,source_bounds_mm=(2,2,205,138),target_bounds_mm=(2,140,205,290))
    plan=plan_tangram_moves(pieces,options)
    polygons=[Polygon(m.target_vertices_a4_mm) for m in plan.moves]
    assert len(plan.moves)==7 and not plan.ambiguous and plan.confidence>=.5
    for move in plan.moves:
        assert Polygon(move.target_vertices_a4_mm).area==pytest.approx(Polygon(move.observed_vertices_a4_mm).area)
    assert all(a.distance(b)>=options.assembly_gap_mm-1e-6 for i,a in enumerate(polygons) for b in polygons[i+1:])
