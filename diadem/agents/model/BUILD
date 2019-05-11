
py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":dq_network",
        ":q_network",
    ],
    visibility = ["//diadem/agents:__pkg__"],
)

py_library(
    name = "dq_network",
    srcs = ["dq_network.py"],
    deps = [
        ":q_network",
        "//diadem/agents/utils:z_range",
    ],
    visibility = ["//diadem/agents:__pkg__"],
)

py_library(
    name = "q_network",
    srcs = ["q_network.py"],
    visibility = ["//diadem/agents:__pkg__"],
)
