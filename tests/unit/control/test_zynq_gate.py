import unittest
from unittest.mock import patch
from puzzle_solver import PlanningOptions,PlanningError,plan_sheet_moves
from tests.unit.control.test_planner import FIXED_PIECES

class ExecutionGateTest(unittest.TestCase):
    def test_success_flag_does_not_override_execution_gate(self):
        for result in ({'success':True,'executable':False,'competition_gate_passed':True},
                       {'success':True,'executable':True,'competition_gate_passed':False},
                       {'success':True,'executable':True,'competition_gate_passed':True,'moves':[]}):
            with self.subTest(result=result), patch('puzzle_solver.planner.solve_observations',return_value=result):
                with self.assertRaises(PlanningError):plan_sheet_moves(FIXED_PIECES,PlanningOptions(mode='fixed'))
