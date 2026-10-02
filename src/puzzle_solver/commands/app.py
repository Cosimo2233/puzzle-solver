"""几何求解命令，不访问硬件。"""
from pathlib import Path
import json
import logging
import typer
from puzzle_solver import PuzzlePiece, PlanningOptions, PlanningError, plan_sheet_moves

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

@app.command()
def solve(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False, envvar="PUZZLE_INPUT"),
    output: Path = typer.Option(..., "--output", dir_okay=False, envvar="PUZZLE_OUTPUT"),
):
    """从毫米轮廓 JSON 计算完整移动方案。"""
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        pieces = tuple(PuzzlePiece(p["id"], tuple(tuple(v) for v in p["vertices_mm"])) for p in payload["pieces"])
        options = PlanningOptions(mode=payload["mode"], noise_bound_mm=payload.get("noise_bound_mm", 0.0))
        plan = plan_sheet_moves(pieces, options)
    except (PlanningError, ValueError, TypeError, KeyError) as error:
        logger.error("求解失败：%s", error)
        raise typer.Exit(code=1) from error
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已写入移动方案：%s", output)

if __name__ == "__main__":
    app()
