py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":environment",
        ":gym_environment"
    ],
    visibility = ["//diadem:__pkg__"],
)

py_library(
    name = "environment",
    srcs = ["environment.py"],
    deps = [
        "//diadem/common:__init__",
    ],
)

py_library(
    name = "gym_environment",
    srcs = ["gym_environment.py"],
    deps = [
        ":environment",
    ],
)
