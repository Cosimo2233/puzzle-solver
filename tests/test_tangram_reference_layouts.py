"""名称和逐片拼法对照独立采样的参考图，不能只凭面积正确判定图案正确。"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from puzzle_solver import PATTERN_NAMES, get_tangram_layout

DATA = json.loads((Path(__file__).parent/'fixtures/tangram-reference-layouts.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('name', DATA['patterns'])
def test_named_layout_matches_independently_digitized_reference(name):
    reference = [np.array(p, dtype=float) for p in DATA['patterns'][name]['vertices']]
    actual = [np.array(p) for p in get_tangram_layout(name).pieces]
    a, b = np.concatenate(actual), np.concatenate(reference)
    # 截图自身有线宽、轻微拉伸和人工取点误差，仅用于核对相对拼法。
    # 这里只给整套图案拟合一个显示尺度和平移，不允许单片自由移动。
    size_a, size_b = np.ptp(a,axis=0), np.ptp(b,axis=0)
    scale = float(np.dot(size_a,size_b)/np.dot(size_a,size_a))
    offset = (b.min(axis=0)+b.max(axis=0))/2 - (a.min(axis=0)+a.max(axis=0))/2*scale
    observed = [Polygon(p*scale+offset).boundary for p in actual]
    targets = [Polygon(p).boundary for p in reference]
    maximum = 0.
    for group in ((0,1),(2,3),(4,),(5,),(6,)):
        costs = np.array([[observed[i].hausdorff_distance(targets[j]) for j in group] for i in group])
        rows, cols = linear_sum_assignment(costs)
        maximum = max(maximum, float(costs[rows,cols].max()))
    assert maximum <= .05*max(size_b), (name, maximum, .05*max(size_b))


def test_chinese_names_match_reference_labels():
    assert dict(PATTERN_NAMES) == dict(square='正方形',cat='猫',rabbit='兔',fish='鱼',boat='船',house='房子',
        dog='狗',swan='天鹅',bird='鸟',penguin='企鹅',seal='海豹',chair='椅子')
