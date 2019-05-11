
py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":buffer",
        ":prioritized",
        ":random"],
    visibility = ["//diadem/agents:__pkg__"],
)

py_library(
    name = "buffer",
    srcs = ["buffer.py"],
)

py_library(
    name = "prioritized",
    srcs = ["prioritized.py"],
    deps = [":buffer"],
)

py_library(
    name = "random",
    srcs = ["random.py"],
    deps = [":buffer"],
)
