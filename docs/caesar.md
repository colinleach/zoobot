This is (probably) the current configuration of Caesar for cascade filtering in GZ Mobile.

### Featured to Spiral Arms

If there are more answers of 'featured' (stats_reducer.1) than smooth (stats_reducer.0), and there are more than 20 total answers, execute a rule.
(The rule, specified via the UI, is to move the subject to the 'spiral arms' workflow)

["and", 
    ["gt", 
        ["lookup", "stats_reducer.1", ["const", 0]],
        ["lookup", "stats_reducer.0", ["const", 1]]
    ],
    ["gt",
        ["lookup","count_reducer.classifications",["const", 0]],
        ["const", 20]
    ]
]

Pasteable version:

["and",["gt",["lookup","stats_reducer.1",["const", 0]],["lookup", "stats_reducer.0", ["const", 1]]],["gt",["lookup","count_reducer.classifications",["const", 0]],["const", 20]]]


### More to come?