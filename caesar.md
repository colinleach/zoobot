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

["and",["gt",["lookup","stats_reducer.1",["const", 0]],["lookup", "stats_reducer.0", ["const", 1]]],["gt",["lookup","count_reducer.classifications",["const", 0]],["const", 20]]]