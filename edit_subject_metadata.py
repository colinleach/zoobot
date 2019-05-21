import json
from tqdm import tqdm

from panoptes_client import Panoptes, Workflow, SubjectSet, Subject

def get_subjects(subject_set_id, login_loc):
    with open(login_loc, 'r') as f:
        zooniverse_login = json.load(f)
    Panoptes.connect(**zooniverse_login)

    subjects = SubjectSet.find(subject_set_id).subjects

    if subject_set_id == '74807':  # priority
        total = 1664  # actually 5766
    elif subject_set_id == '74905':  # random
        total = 6000
    else:
        raise ValueError('subject set not recognised')

    with tqdm(total=total) as pbar:  # or 6000 in random set
        while True:
            subject = next(subjects)
            assert '#retirement_limit' in subject.metadata.keys()
            included_keys = get_classification_columns().intersection(
                set(subject.metadata.keys())
                )
            # print(included_keys)
            for key in included_keys:
                try:
                    del subject.metadata[key]
                except KeyError:
                    pass
            subject.save()
            pbar.update()

    # subject = Subject.find('32410759')
    # metadata = subject.metadata
    # print(metadata.keys())


    # subject.metadata.update({})

def get_classification_columns():
    return set(['!subject_id', '!subject_url', '!id_str', '!bar_no','!bar_weak', '!png_ready', '!Unnamed: 0', '!bar_no_max', '!bar_no_min', '!bar_strong', '!fits_ready', '!bar_weak_max', '!bar_weak_min', '!merging_none', '!bar_prediction', '!bar_strong_max', '!bar_strong_min','!merging_merger', '!bar_no_fraction', '!bar_total-votes', '!bulge-size_none', '!disk-edge-on_no', '!merging_both-v1', '!bulge-size_large', '!bulge-size_small', '!disk-edge-on_yes', '!merging_none_max', '!merging_none_min', '!bar_weak_fraction', '!how-rounded_round', '!edge-on-bulge_boxy', '!edge-on-bulge_none', '!has-spiral-arms_no', '!merging_merger_max', '!merging_merger_min', '!merging_neither-v1', '!merging_prediction', '!spiral-arm-count_1', '!spiral-arm-count_2', '!spiral-arm-count_3', '!spiral-arm-count_4', '!bar_prediction-conf', '!bar_strong_fraction', '!bulge-size_dominant', '!bulge-size_moderate', '!bulge-size_none_max', '!bulge-size_none_min', '!disk-edge-on_no_max', '!disk-edge-on_no_min', '!has-spiral-arms_yes', '!merging_both-v1_max', '!merging_both-v1_min', '!merging_total-votes', '!bulge-size_large_max', '!bulge-size_large_min', '!bulge-size_small_max', '!bulge-size_small_min', '!disk-edge-on_yes_max', '!disk-edge-on_yes_min', '!spiral-winding_loose', '!spiral-winding_tight', '!bulge-size_prediction', '!edge-on-bulge_rounded', '!how-rounded_round_max', '!how-rounded_round_min', '!merging_none_fraction', '!spiral-winding_medium', '!bar_prediction-encoded', '!bulge-size_total-votes', '!edge-on-bulge_boxy_max', '!edge-on-bulge_boxy_min', '!edge-on-bulge_none_max', '!edge-on-bulge_none_min', '!has-spiral-arms_no_max', '!has-spiral-arms_no_min', '!how-rounded_in-between', '!how-rounded_prediction', '!merging_neither-v1_max', '!merging_neither-v1_min', '!spiral-arm-count_1_max', '!spiral-arm-count_1_min', '!spiral-arm-count_2_max', '!spiral-arm-count_2_min', '!spiral-arm-count_3_max', '!spiral-arm-count_3_min', '!spiral-arm-count_4_max', '!spiral-arm-count_4_min', '!bulge-size_dominant_max', '!bulge-size_dominant_min', '!bulge-size_moderate_max', '!bulge-size_moderate_min', '!disk-edge-on_prediction', '!has-spiral-arms_yes_max', '!has-spiral-arms_yes_min', '!how-rounded_total-votes', '!merging_merger_fraction', '!merging_prediction-conf', '!merging_tidal-debris-v1', '!bulge-size_none_fraction', '!disk-edge-on_no_fraction', '!disk-edge-on_total-votes', '!edge-on-bulge_prediction', '!how-rounded_cigar-shaped', '!merging_both-v1_fraction', '!spiral-winding_loose_max', '!spiral-winding_loose_min', '!spiral-winding_tight_max', '!spiral-winding_tight_min', '!bulge-size_large_fraction', '!bulge-size_small_fraction', '!disk-edge-on_yes_fraction', '!edge-on-bulge_rounded_max', '!edge-on-bulge_rounded_min', '!edge-on-bulge_total-votes', '!merging_major-disturbance', '!merging_minor-disturbance', '!smooth-or-featured_smooth', '!spiral-winding_medium_max', '!spiral-winding_medium_min', '!spiral-winding_prediction', '!bulge-size_prediction-conf', '!has-spiral-arms_prediction', '!how-rounded_in-between_max', '!how-rounded_in-between_min', '!how-rounded_round_fraction', '!merging_prediction-encoded', '!spiral-arm-count_cant-tell', '!spiral-winding_total-votes', '!edge-on-bulge_boxy_fraction', '!edge-on-bulge_none_fraction', '!has-spiral-arms_no_fraction', '!has-spiral-arms_total-votes', '!how-rounded_prediction-conf', '!merging_neither-v1_fraction', '!merging_tidal-debris-v1_max', '!merging_tidal-debris-v1_min', '!smooth-or-featured_artifact', '!spiral-arm-count_1_fraction', '!spiral-arm-count_2_fraction', '!spiral-arm-count_3_fraction', '!spiral-arm-count_4_fraction', '!spiral-arm-count_prediction', '!bulge-size_dominant_fraction', '!bulge-size_moderate_fraction', '!disk-edge-on_prediction-conf', '!has-spiral-arms_yes_fraction', '!how-rounded_cigar-shaped_max', '!how-rounded_cigar-shaped_min', '!spiral-arm-count_more-than-4', '!spiral-arm-count_total-votes', '!bulge-size_prediction-encoded', '!edge-on-bulge_prediction-conf', '!merging_major-disturbance_max', '!merging_major-disturbance_min', '!merging_minor-disturbance_max', '!merging_minor-disturbance_min', '!smooth-or-featured_prediction', '!smooth-or-featured_smooth_max', '!smooth-or-featured_smooth_min', '!spiral-winding_loose_fraction', '!spiral-winding_tight_fraction', '!edge-on-bulge_rounded_fraction', '!how-rounded_prediction-encoded', '!smooth-or-featured_total-votes', '!spiral-arm-count_cant-tell_max', '!spiral-arm-count_cant-tell_min', '!spiral-winding_medium_fraction', '!spiral-winding_prediction-conf', '!disk-edge-on_prediction-encoded', '!has-spiral-arms_prediction-conf', '!how-rounded_in-between_fraction', '!smooth-or-featured_artifact_max', '!smooth-or-featured_artifact_min', '!edge-on-bulge_prediction-encoded', '!merging_tidal-debris-v1_fraction', '!spiral-arm-count_more-than-4_max', '!spiral-arm-count_more-than-4_min', '!spiral-arm-count_prediction-conf', '!how-rounded_cigar-shaped_fraction', '!spiral-winding_prediction-encoded', '!has-spiral-arms_prediction-encoded', '!merging_major-disturbance_fraction', '!merging_minor-disturbance_fraction', '!smooth-or-featured_prediction-conf', '!smooth-or-featured_smooth_fraction', '!smooth-or-featured_featured-or-disk', '!spiral-arm-count_cant-tell_fraction', '!spiral-arm-count_prediction-encoded', '!smooth-or-featured_artifact_fraction', '!smooth-or-featured_prediction-encoded', '!spiral-arm-count_more-than-4_fraction', '!smooth-or-featured_featured-or-disk_max', '!smooth-or-featured_featured-or-disk_min', '!smooth-or-featured_featured-or-disk_fraction'])

if __name__ == '__main__':
 
    login_loc = 'zooniverse_login.json'
    subject_set_id = '74905'  # random
    # subject_set_id = '74807'  # priority
    get_subjects(subject_set_id, login_loc)
