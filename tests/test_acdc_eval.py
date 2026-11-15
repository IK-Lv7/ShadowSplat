import numpy as np

from workstation.eval.acdc_eval import confusion, miou_from_conf


def test_miou_ignores_255_and_absent_classes():
    gt = np.array([[0, 0, 1, 255]], np.uint8)
    pred = np.array([[0, 1, 1, 5]], np.int64)
    conf = confusion(pred, gt)
    assert conf.sum() == 3  # ignored pixels are not counted
    m, per = miou_from_conf(conf)
    # class0: inter 1 / union 2 = 0.5, class1: inter 1 / union 2 = 0.5; other classes are absent from the ground truth and excluded
    assert abs(m - 0.5) < 1e-9 and per[0] == 0.5 and per[1] == 0.5 and per[2] is None
