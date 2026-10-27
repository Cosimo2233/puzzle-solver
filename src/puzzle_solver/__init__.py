"""公开强类型纸片拼图控制契约和无副作用求解入口。"""

from puzzle_solver.contracts import (
    MovePlan,
    PlannedMove,
    PlanningError,
    PlanningOptions,
    PuzzlePiece,
)
from puzzle_solver.planner import plan_sheet_moves
from puzzle_solver.tangram import (
    PATTERN_NAMES, TangramLayout, TangramPlanningOptions,
    get_tangram_layout, classify_tangram_piece, plan_tangram_moves,
)

__all__ = [
    "MovePlan",
    "PlannedMove",
    "PlanningError",
    "PlanningOptions",
    "PuzzlePiece",
    "plan_sheet_moves",
    "PATTERN_NAMES", "TangramLayout", "TangramPlanningOptions",
    "get_tangram_layout", "classify_tangram_piece", "plan_tangram_moves",
]
