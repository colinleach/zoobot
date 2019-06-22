import os
from zoobot.active_learning import misc

def test_get_latest_checkpoint_dir(estimators_dir):
    latest_ckpt = misc.get_latest_checkpoint_dir(estimators_dir)
    assert os.path.split(latest_ckpt)[-1] == '157003'
