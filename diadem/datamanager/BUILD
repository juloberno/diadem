py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":data_manager",
        ":data_manager_csv",
        ":data_manager_numpy",
        ":data_manager_pandas",
        ":data_manager_pickle"
    ],
    visibility = ["//diadem:__pkg__"],
)

py_library(
    name = "data_manager",
    srcs = ["data_manager.py"]
)

py_library(
    name = "data_manager_csv",
    srcs = ["data_manager_csv.py"],
    deps = [":data_manager"],
)

py_library(
    name = "data_manager_numpy",
    srcs = ["data_manager_numpy.py"],
    deps = [":data_manager"],
)

py_library(
    name = "data_manager_pandas",
    srcs = ["data_manager_pandas.py"],
    deps = [":data_manager"],
)

py_library(
    name = "data_manager_pickle",
    srcs = ["data_manager_pickle.py"],
    deps = [":data_manager"],
)