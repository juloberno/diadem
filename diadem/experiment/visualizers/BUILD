py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":base_visualizer",
        ":online_visualizers",
        ":image_visualizers",
        ":video_visualizers"
    ],
    visibility = ["//diadem/experiment:__pkg__"],
)

py_library(
    name = "base_visualizer",
    srcs = ["base_visualizer.py"],
    deps = [
        "//diadem/common:__init__"
    ],

)

py_library(
    name = "online_visualizers",
    srcs = ["online_visualizers.py"],
    deps = [
        ":base_visualizer"
    ],
)

py_library(
    name = "image_visualizers",
    srcs = ["image_visualizers.py"],
    deps = [
        ":base_visualizer"
    ],
)

py_library(
    name = "video_visualizers",
    srcs = ["video_visualizers.py"],
    deps = [
        ":base_visualizer",
        ":image_visualizers",
        "//diadem/experiment/evaluation/export:__init__"
    ],
)
