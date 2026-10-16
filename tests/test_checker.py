import numpy as np
import torch

from whatchanged.checker import model, train
from whatchanged.checker.pairs import point_in_poly


def test_checker_output_shape_and_low_level_inputs():
    net = model.Checker().eval()
    fr = torch.zeros(1, 128, 6, 8)
    ren = np.zeros((24, 32, 3), np.uint8)
    d_render = np.ones((24, 32), np.float32)
    d_real = d_render * 1.2
    known = np.ones((24, 32), bool)
    low = model.make_low(ren, ren, d_render, d_real, known)
    assert low.shape == (9, 24, 32)
    assert np.allclose(low[6], 0.8)  # +20 % depth, scaled by 4
    with torch.no_grad():
        out = net(fr, fr, torch.as_tensor(low)[None])
    assert out.shape == (1, 3, 24, 32)


def test_label_block_vote_keeps_thin_changes():
    lab = np.zeros((8, 8), np.uint8)
    lab[0:4, 0] = model.APPEARED  # a one-pixel-wide sliver covers 25 % of the block
    lab[4:8, 4:8] = train.IGNORE
    q = train.quarter_labels(lab, 2, 2)
    assert q[0, 0] == model.APPEARED and q[1, 1] == train.IGNORE and q[0, 1] == 0


def test_point_in_convex_footprint():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    inside = point_in_poly(np.array([[0.5, 0.5], [1.5, 0.5], [0.02, 0.98]]), square)
    assert inside.tolist() == [True, False, True]
