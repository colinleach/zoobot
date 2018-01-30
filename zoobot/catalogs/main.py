from zoobot.catalogs.get_galaxy_zoo_catalog import get_catalog

if __name__ == '__main__':

    nrows = None

    catalog_dir = '/data/galaxy_zoo/gz2/subjects'
    published_data_loc = '{}/gz2_hart16.csv'.format(catalog_dir)  # volunteer labels
    subject_manifest_loc = '{}/galaxyzoo2_sandor.csv'.format(catalog_dir)  # subjects on AWS

    labels_loc = '{}/all_labels.csv'.format(catalog_dir)  # will place catalog of file list and labels here

    labels = get_catalog(published_data_loc, subject_manifest_loc, labels_loc)
