py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":categorical_loss",
        ":huber_loss",
        ":quantile_loss",
    ],
    visibility = ["//diadem/agents:__pkg__"],
)

py_binary(
    name = "categorical_loss",
    srcs = ["categorical_loss.py"],
    deps = [
        "//diadem/agents/utils:z_range",
    ],
    visibility = ["//diadem/agents:__pkg__"],
)

py_binary(
    name = "huber_loss",
    srcs = ["huber_loss.py"],
    visibility = ["//diadem/agents:__pkg__"],
)

py_binary(
    name = "quantile_loss",
    srcs = ["quantile_loss.py"],
    deps = [
        ":categorical_loss",
        ":huber_loss",
    ],
    visibility = ["//diadem/agents:__pkg__"],
)
