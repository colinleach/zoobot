import os
import json
import io

import pandas as pd
import matplotlib.pyplot as plt

# def get_jpg_name(subject_row):
#     try:
#         name = json.loads(subject_row['metadata'])['jpg_file']
#     except KeyError:
#         try:
#             name = json.loads(subject_row['metadata'])['jpg_file_after']
#         except KeyError:
#             print(json.loads(subject_row['metadata']))
#             raise KeyError
#     return name.replace('_dg_', '_planet_')
# # .replace('before', 'before-reprojected')

def get_jpg_loc(subject_row):
    jpg_dir = '/Users/mikewalmsley/repos/zoobot/zoobot/prn/data/jpg/aws'
    return jpg_dir + '/' + str(subject_row['aws_loc'].split('/')[-1])


def get_aws_loc(subject_row):
    locations = json.loads(subject_row['locations'])
    try:
        return locations['1']
    except KeyError:
        return locations['0']


    # return json.loads(subject_row['metadata'])['jpg_file_after']

def main():
    new_subjects = True
    subjects_loc = 'data/subjects.csv'
    if new_subjects:
        subjects_a = pd.read_csv('/Users/mikewalmsley/repos/zoobot/zoobot/prn/data/planetary-response-network-and-rescue-global-caribbean-storms-2017-subjects_enhancedinfo_ssids_14929.csv')
        subjects_b = pd.read_csv('/Users/mikewalmsley/repos/zoobot/zoobot/prn/data/planetary-response-network-and-rescue-global-caribbean-storms-2017-subjects_enhancedinfo_ssids_14988.csv')
        subjects_c = pd.read_csv('/Users/mikewalmsley/repos/zoobot/zoobot/prn/data/planetary-response-network-and-rescue-global-caribbean-storms-2017-subjects_enhancedinfo_ssids_15178_15208.csv')
        subjects = pd.concat([subjects_a, subjects_b, subjects_c])
        subjects['aws_loc'] = subjects.apply(get_aws_loc, axis=1)
        subjects['jpg_loc'] = subjects.apply(get_jpg_loc, axis=1)
        subjects[['subject_id', 'jpg_loc', 'aws_loc', 'classifications_count', 'imsize_x_pix', 'imsize_y_pix']].to_csv(subjects_loc, index=False)
    else:
        subjects = pd.read_csv(subjects_loc)

    print(len(subjects))

    # create wget file
    with io.open('subjects_for_wget.txt', 'w') as f:
        for row_n, row in subjects.iterrows():
            f.write('{}\n'.format(row['aws_loc']))


    new_classifications = True
    classifications_loc = 'data/classifications.csv'
    if new_classifications:
        classifications = pd.read_csv('/Users/mikewalmsley/repos/zoobot/zoobot/prn/data/5030_puerto_rico.csv')
        del classifications['data.1']
        classifications['votes_for_bad_img'] = classifications['data.0']
        del classifications['data.0']
        classifications.to_csv(classifications_loc)
    else:
        classifications = pd.read_csv(classifications_loc)

    labels_loc = 'data/labels.csv'
    new_labels = True
    if new_labels:
        subject_classifications = classifications.groupby('subject_id').agg({'votes_for_bad_img': 'sum'}).reset_index()
        subject_classifications['bad_img'] = subject_classifications['votes_for_bad_img'] >= 5
        print(subject_classifications.iloc[0])
        labels = pd.merge(subjects, subject_classifications, on='subject_id', how='inner')
        labels['image_exists'] = labels['jpg_loc'].apply(lambda x: os.path.exists(x))
        assert len(labels) > 0
        labels.to_csv(labels_loc)
    else:
        labels = pd.read_csv(labels_loc)


    print(labels[labels['image_exists']])


if __name__ == '__main__':
    main()

    # before/first-set/tiles before jpg - 'puerto_rico_planet_before-reprojected_xxx_xxx.jpg
    # after/first_set/tiles/tiles_after_jpg - 'puerto_rico_planet_after_maria_xxx_xxx.jpg

    # zoobot/prn/data/jpg/dominica_planet_after_02_02.jpg
    # 'data/jpg/puerto_rico_planet_before_009_014.jpg'