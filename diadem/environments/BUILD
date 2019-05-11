py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":environment",
        ":gym_environment",
        ":bark_highway"
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


py_library(
    name = "bark_highway",
    srcs = ["bark_highway.py"],
    data = ['@bark_project//python:bark.so'],
    imports = ['../../bark_project/python/'],
    deps = [
        ":environment",
        "@bark_project//modules/runtime:runtime"
    ],
)