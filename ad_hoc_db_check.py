import os
import sqlite3

from zoobot.active_learning import make_shards

if __name__ == '__main__':
    shard_dir = 'data/decals/shards/decals_multiq_128_sim'

    db = sqlite3.connect(os.path.join(shard_dir, 'static_shard_db.db'))
    make_shards.check_all_ids_are_in_db(shard_dir, db)

    iteration_dir = 'data/experiments/decals_multiq_sim/iteration_0'
    db = sqlite3.connect(os.path.join(iteration_dir, 'iteration.db'))
    make_shards.check_all_ids_are_in_db(shard_dir, db)

    iteration_dir = 'data/experiments/decals_multiq_sim/iteration_1'
    db = sqlite3.connect(os.path.join(iteration_dir, 'iteration.db'))
    make_shards.check_all_ids_are_in_db(shard_dir, db)