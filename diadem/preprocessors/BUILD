py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":preprocessor",
        ":normalization",
    ],
    visibility = ["//diadem:__pkg__"],
)

py_library(
    name = "preprocessor",
    srcs = ["preprocessor.py"],
    deps = [
        "//diadem/common:__init__",
    ],
)

py_library(
    name = "normalization",
    srcs = ["normalization.py"],
    deps = [
        ":preprocessor",
    ],
)
