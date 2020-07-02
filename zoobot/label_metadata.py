
decals_questions = [
    'smooth-or-featured',
    'disk-edge-on',
    'has-spiral-arms',
    'bar',
    'bulge-size',
    'how-rounded',
    'edge-on-bulge',
    'spiral-winding',
    'spiral-arm-count',  # bad naming
    'merging'
]

decals_label_cols = [
    'smooth-or-featured_smooth',
    'smooth-or-featured_featured-or-disk',
    'smooth-or-featured_artifact',
    'disk-edge-on_yes',
    'disk-edge-on_no',
    'has-spiral-arms_yes',
    'has-spiral-arms_no',
    'bar_strong',
    'bar_weak',
    'bar_no',
    #  5 answers for bulge size, pushing it - best to aggregate large/dominant
    'bulge-size_dominant',
    'bulge-size_large',
    'bulge-size_moderate',
    'bulge-size_small',
    'bulge-size_none',
    'how-rounded_round',
    'how-rounded_in-between',
    'how-rounded_cigar-shaped',
    'edge-on-bulge_boxy',
    'edge-on-bulge_none',
    'edge-on-bulge_rounded',
    'spiral-winding_tight',
    'spiral-winding_medium',
    'spiral-winding_loose',
    #  6 answers for spiral count, definitely too many
    'spiral-arm-count_1',
    'spiral-arm-count_2',
    'spiral-arm-count_3',
    'spiral-arm-count_4',
    'spiral-arm-count_more-than-4',
    'spiral-arm-count_cant-tell',
    'merging_none',
    'merging_minor-disturbance',
    'merging_major-disturbance',
    'merging_merger'
    #  also have v1 bulge size and merger answers, to be sure to exclude from catalog
]

decals_partial_questions = [
    'smooth-or-featured',
    'has-spiral-arms',
    'bar',
    'bulge-size'
]

decals_partial_label_cols = [
    'smooth-or-featured_smooth',
    'smooth-or-featured_featured-or-disk',
    'has-spiral-arms_yes',
    'has-spiral-arms_no',
    'spiral-winding_tight',
    'spiral-winding_medium',
    'spiral-winding_loose',
    'bar_strong',
    'bar_weak',
    'bar_no',
    'bulge-size_dominant',
    'bulge-size_large',
    'bulge-size_moderate',
    'bulge-size_small',
    'bulge-size_none'
]




gz2_partial_questions = [
    'smooth-or-featured',
    'has-spiral-arms',
    'bar',
    'bulge-size'
]

gz2_questions = [
    'smooth-or-featured',
    'disk-edge-on',
    'has-spiral-arms',
    'bar',
    'bulge-size',
    'something-odd',
    'how-rounded',
    'bulge-shape',
    'spiral-winding',
    'spiral-count'
]

gz2_partial_label_cols = [
    'smooth-or-featured_smooth',
    'smooth-or-featured_featured-or-disk',
    'has-spiral-arms_yes',
    'has-spiral-arms_no',
    'bar_yes',
    'bar_no',
    'bulge-size_dominant',
    'bulge-size_obvious',
    'bulge-size_just-noticeable',
    'bulge-size_no'
]

gz2_label_cols = [
    'smooth-or-featured_smooth',
    'smooth-or-featured_featured-or-disk',
    'disk-edge-on_yes',
    'disk-edge-on_no',
    'has-spiral-arms_yes',
    'has-spiral-arms_no',
    'bar_yes',
    'bar_no',
    'bulge-size_dominant',
    'bulge-size_obvious',
    'bulge-size_just-noticeable',
    'bulge-size_no',
    'something-odd_yes',
    'something-odd_no',
    'how-rounded_round',
    'how-rounded_in-between',
    'how-rounded_cigar',
    'bulge-shape_round',
    'bulge-shape_boxy',
    'bulge-shape_no-bulge',
    'spiral-winding_tight',
    'spiral-winding_medium',
    'spiral-winding_loose',
    'spiral-count_1',
    'spiral-count_2',
    'spiral-count_3',
    'spiral-count_4',
    'spiral-count_more-than-4',
    'spiral-count_cant-tell'
]