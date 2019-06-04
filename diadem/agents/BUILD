py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":agent",
        ":agent_context",
        ":dqfd_agent",
        ":hba_agent",
        ":distributional_dqfd_agent",
        ":agent_manager",
        ":experience_replay_agent",
        ":ensemble_agent"
    ],
    visibility = ["//diadem:__pkg__"],
)


py_library(
    name = "agents",
    deps = [
        ":agent",
        ":dqfd_agent",
        ":hba_agent",
        ":distributional_dqfd_agent",
        ":experience_replay_agent",
    ],
)

py_library(
    name = "agent",
    srcs = ["agent.py"],
)

py_library(
    name = "agent_context",
    srcs = ["agent_context.py"],
    deps = [
        ":agent",
    ]
)

py_library(
    name = "dqfd_agent",
    srcs = ["dqfd_agent.py"],
    deps = [":agent_context",
            ":experience_replay_agent",
            "//diadem/agents/model:__init__" ,
    ],
)

py_library(
    name = "experience_replay_agent",
    srcs = ["experience_replay_agent.py"],
    deps = [
        ":agent",
        "//diadem/agents/buffer:__init__",
    ],
)

py_binary(
    name = "hba_agent",
    srcs = ["hba_agent.py"],
    deps = [
        ":agent",
        "//diadem/agents/exploration:__init__",
    ],
)

py_binary(
    name = "ensemble_agent",
    srcs = ["ensemble_agent.py"]
    deps = [
        ":agent",
    ]
)

py_library(
    name = "agent_manager",
    srcs = ["agent_manager.py"],
    deps = [
        ":agents",
        ":agent_context"
    ],
)

py_library(
    name = "distributional_dqfd_agent",
    srcs = ["distributional_dqfd_agent.py"],
    deps = [
        "//diadem/agents/losses:__init__",
        "//diadem/agents/utils:__init__",
    ],
)
