py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":reader",
        ":analyzer",
    ],
    visibility = ["//diadem/summary:__pkg__"],
)

py_binary(
    name = "analyzer",
    srcs = ["analyzer.py"],
    deps = [
        ":reader",
    ],
)

py_binary(
    name = "reader",
    srcs = ["reader.py"],
)
