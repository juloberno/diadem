py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":base_summary",
        ":console_summary",
        ":pandas_summary",
        "//diadem/summary/tf_summary:__init__"
    ],
    visibility = ["//diadem:__pkg__"],
)

py_library(
    name = "base_summary",
    srcs = ["base_summary.py"],
    deps = [
        "//diadem/common:__init__",
    ]
)

py_library(
    name = "console_summary",
    srcs = ["console_summary.py"],
    deps = [":base_summary"],
)

py_library(
    name = "pandas_summary",
    srcs = ["pandas_summary.py"],
    deps = [
        ":base_summary",
    ],
)
