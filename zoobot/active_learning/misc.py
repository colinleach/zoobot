import os

def get_latest_checkpoint_dir(base_dir):
    saved_models = [x for x in os.listdir(base_dir) if x.startswith('15')]  # subfolders with timestamps
    saved_models.sort(reverse=True)  # sort by name i.e. timestamp, early timestamp first
    return os.path.join(base_dir, saved_models[0])  # the subfolder with the most recent time

