
# Notebooks

coverage.ipynb investigates why decals dr5 merged with the NSA catalog includes galaxies not previously classified in GZ2. See notebook for conclusions.

debiasing.ipynb replicates the morphology/redshift plots from 0805.2612 Fig A12 to see if debiasing is needed for decals - it definitely is.

decals_dr12_fixes.ipynb looks for issues in the DR1/2 classifications. There are some galaxies repeating in both lists of subjects, but these are only classified in one, so no fix is needed. 7.5k DR1 subjects were never classified (excluding the duplicates), and having checked the original images, they were never actually uploaded. Future work might consider re-adding these. Finally, checking for 20% completeness in all galaxies, 1k subjects weren't complete and are discarded. The aggregate classifications, with only good images, are saved to dr2_aggregated_votes_good_subjects.csv (92,960 galaxies).

decals_dr5_fixes.ipynb does the same for DR5. This makes several changes. 1) Remove 12k galaxies which were mistakenly uploaded despite having bad images (fits_filled=False in the joint catalog), and also to filter the joint catalog to exclude these, creating final_dr5_uploadable_catalog.csv 2) add a column labelling the origin of each galaxy (e.g. calibration, pre-active, etc), resolving duplicates with the priority pre_active > priority > random (see code for details) to preserve the selection functions, and add a flag for galaxies in the dr2/5/nair overlap.

decals_prep.ipynb is a sense check of the galaxy metadata and images for DR1/2/5. I find that 1300 galaxies are in the DR5 joint catalog but were not in the DR1/2 metadata catalog despite passing the expected selection cuts, which is surprising but should self-correct since they'll be classified in DR5 now. 51 galaxies were classified in DR5 but are not in the upload catalog, these were test uploads and must be removed. 12k had fits_filled=False, which I remove in decals_dr5_fixes.ipynb. Finally, I find 26k galaxies not-yet-uploaded galaxies in the joint catalog that are not classified in DR1/2, pass DR5 selection cuts, and have good images. These are saved as dr5_missing_galaxies.csv, and uploaded to subject sets missing_decals_dr5_subjects and missing_decals_dr5_subjects_d (split into two uploads due to connection dropping) using gz-scripts.


# Process to create data release

The joint NSA/DR5 catalog, after selection cuts and without excluding any previously-uploaded galaxies, is /media/walml/beta/galaxy_zoo/decals/catalogs/dr5_nsa1_0_0_to_upload.fits

The DR1/2 subject catalog (subject_id, metadata) is /media/walml/beta/galaxy_zoo/decals/dr1_dr2/subjects/decals_dr1_and_dr2_with_subj.csv. (subject_id is extracted from the id column).


The DR1/2 classifications (classification_id, user, task_0, task_1, ...) are:
dr1_c = pd.read_csv('/media/walml/beta/galaxy_zoo/decals/gzreduction_ouroborous/working_dir/raw/classifications/2017-10-15_galaxy_zoo_decals_classifications.csv')
dr2_c = pd.read_csv('/media/walml/beta/galaxy_zoo/decals/gzreduction_ouroborous/working_dir/raw/classifications/2017-10-15_galaxy_zoo_decals_dr2_classifications.csv')
These are converted to aggregated classifications (including fractions and totals) using using gz-panoptes-reduction/gz_reduction/ouroborous/ouroborous_extract_to_votes.py
The aggregated votes are at /media/walml/beta/galaxy_zoo/decals/gzreduction_ouroborous/working_dir/votes/dr2_aggregated_votes.csv

The DR5 aggregated classifications come from Tobias' aggregation of an all-classifications export (latest is currently /home/walml/Downloads/classifications_200520.csv). This is run through decals_dr5_fixes.ipynb to make various tweaks (see above) then saved as current_final_dr5_result_without_metadata.csv. Tobias' code handles the change in merger question by wiping the merger answers for classifications made with a workflow_id from before the change. Galaxies active during the switch will have fewer merger answers than smooth/featured answers, which is a bit odd but probably the best option.

The latest subjects, used for metadata checks and finding missing subjects, are from a Panoptes export, currently /home/walml/repos/gz-panoptes-reduction/data/latest_subjects_export.csv

The results of each 'fixes.ipynb' notebook, plus the two subject catalogs, are 

Tobias powers the final catalog, while Spark-derived classifications are used for checking per-user statistics and rejecting high artifact users.

I need to check Tobias + Spark at least roughly agree. 
