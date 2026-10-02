"""通过公共 API 检查固定/随机完整方案；也可在 wheel 环境单独运行。"""
import json
from pathlib import Path
import sys
import pytest
from puzzle_solver import PuzzlePiece, PlanningOptions, plan_sheet_moves

@pytest.mark.parametrize('name', ['fixed4', 'random2', 'random3', 'random4'])
def test_baseline_complete_plan(name):
    payload = json.loads((Path(__file__).resolve().parents[1] / 'baselines' / f'{name}.json').read_text())
    pieces = tuple(PuzzlePiece(p['id'], tuple(tuple(v) for v in p['vertices_mm'])) for p in payload['pieces'])
    plan = plan_sheet_moves(pieces, PlanningOptions(mode=payload['mode']))
    assert len(plan.moves) == len(pieces)
    assert {m.piece_id for m in plan.moves} == {p.piece_id for p in pieces}
    assert not any(m == 'ec_26_e' or m.startswith('ec_26_e.') for m in sys.modules)

def test_reject_duplicate_piece_ids():
    p = PuzzlePiece('same', ((0, 0), (10, 0), (0, 10)))
    with pytest.raises(ValueError, match='piece_id'):
        plan_sheet_moves((p, p))
