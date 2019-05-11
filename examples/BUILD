py_test(
    name = "dqn_basic",
    srcs = ["dqn_basic.py"],
    deps = [
        ":run_dqn",
    ],
    data = [
        "//examples/example_params:example_params",
    ]
)

py_library(
    name = "run_dqn",
    srcs = ["dqn_basic.py"],
    deps = [
        "//diadem:__init__",
    ],
    data = [
        "//examples/example_params:example_params",
    ]
)

py_test(
    name = "dqn_distributional_quantile",
    srcs = ["dqn_distributional_quantile.py"],
    deps = [
        ":run_dqn",
    ],
    data = [
        "//examples/example_params:example_params",
    ]
)

py_test(
    name = "dqn_distributional_categorical",
    srcs = ["dqn_distributional_categorical.py"],
    deps = [
        ":run_dqn",
    ],
    data = [
        "//examples/example_params:example_params",
    ],
    timeout = "long"
)


py_test(
    name = "video_rendering",
    srcs = ["video_rendering.py"],
    deps = [
        "//diadem:__init__",
    ],
    data = [
        "//examples/example_params:example_params",
    ]
)

py_test(
    name = "online_rendering",
    srcs = ["online_rendering.py"],
    deps = [
        "//diadem:__init__",
    ],
    data = [
        "//examples/example_params:example_params",
    ]
)

test_suite(
    name = "dqn_cartpole",
    tests = [
        "dqn_basic",
        "dqn_distributional_categorical",
        "dqn_distributional_quantile",
    ]
)

test_suite(
    name = "visualization",
    tests = [
        "video_rendering",
        "online_rendering"
    ]
)



py_test(
    name = "bark_dqn",
    srcs = ["bark_dqn.py"],
    deps = [
        "//diadem:__init__",
    ],
    data = [
        "//examples/example_params:example_params",
    ]
)