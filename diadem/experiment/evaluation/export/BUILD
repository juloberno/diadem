
py_binary(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":export_image",
        ":export_latex",
        ":svg_export",
        ":video_renderer"
    ],
    visibility = [
        "//diadem/experiment/evaluation:__pkg__",
        "//diadem/experiment/visualizers:__pkg__"
    ],
)


py_binary(
    name = "export_image",
    srcs = ["export_image.py"],
)

py_library(
    name = "export_latex",
    srcs = ["export_latex.py"],
)

py_binary(
    name = "svg_export",
    srcs = ["svg_export.py"],
)

py_library(
    name = "video_renderer",
    srcs = ["video_renderer.py"],
)
