py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [":base_object",
            ":config_logging",
            ":extract_params",
            ":anneal",
            ":params",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "anneal",
    srcs = ["anneal.py"],
)

py_library(
    name = "base_object",
    srcs = ["base_object.py"],
    deps = [":params"],
)

py_library(
    name = "config_logging",
    srcs = ["config_logging.py"],
)

py_library(
    name = "extract_params",
    srcs = ["extract_params.py"],
    deps = [":params"]
)

py_library(
    name = "params",
    srcs = ["params.py"],
    deps = [":anneal"],
)
