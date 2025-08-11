# Run terminal bench task using a single agent

# 1. select a task from the terminal bench tasks
from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.terminal.models import TerminalCommand
from terminal_bench.terminal.terminal import Terminal, spin_up_terminal
from terminal_bench.harness.models import FailureMode
from pathlib import Path

task_name = "simple-web-scraper"
task_path = Path(__file__).parent.parent.parent / "dataset" / "tbench-tasks" / task_name
print(f"Using task path: {task_path}")
# Check if the task path exists
if not task_path.exists():
    print(f"Task path {task_path} does not exist. Please check the path.")

attempt = 1; _n_attempts = 1; _run_id = "test_run"; output_path = Path("output") / f"{_run_id}"
trial_name = f"{task_path.name}.{attempt}-of-{_n_attempts}.{_run_id}"

trial_handler = TrialHandler(
    trial_name=trial_name,
    input_path=task_path,
    output_path=output_path,
)

task_instruction = trial_handler.instruction
print(f"Task instruction: {task_instruction}")

# 2. spin up a docker container and tmux session inside the container
# terminal = Terminal(
#     client_container_name=trial_handler.client_container_name,
#     client_image_name=trial_handler.client_image_name,
#     docker_image_name_prefix=trial_handler.docker_image_name_prefix,
#     docker_compose_path=trial_handler.task_paths.docker_compose_path,
#     sessions_logs_path=trial_handler.trial_paths.sessions_path,
#     agent_logs_path=trial_handler.trial_paths.agent_logging_dir,
#     commands_path=trial_handler.trial_paths.commands_path,
#     no_rebuild=True,
#     cleanup=False,
#     livestream=False,
#     disable_recording=False,
# )
# terminal.start()
# ==========================================================================================
#
#               create chat agent with terminal toolkit docker
# ==========================================================================================
# 4. Run the agent
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkitDocker
from camel.types import ModelPlatformType, ModelType


with spin_up_terminal(
        client_container_name=trial_handler.client_container_name,
        client_image_name=trial_handler.client_image_name,
        docker_image_name_prefix=trial_handler.docker_image_name_prefix,
        docker_compose_path=trial_handler.task_paths.docker_compose_path,
        sessions_logs_path=trial_handler.trial_paths.sessions_path,
        agent_logs_path=trial_handler.trial_paths.agent_logging_dir,
        commands_path=trial_handler.trial_paths.commands_path,
        no_rebuild=True,
        cleanup=False,
        livestream=False,
        disable_recording=False,
        ) as terminal:
    # 3. Create chat agent and run the task, pass in the terminal_toolkit_docker, and container name
    session = terminal.create_session(
        "agent", is_active_stream=False, as_configured_user=True
    )


    # ==========================================================================================
    #
    #               create terminal toolkit docker instance and check basic functionality
    # ==========================================================================================
    from camel.toolkits import TerminalToolkitDocker

    user = terminal.container.attrs["Config"].get("User", "")
    print(f"Using user: {user}")
    log_path = trial_handler.trial_paths.sessions_path / 'logs'
    print(f"Using log path: {log_path}")
    terminal_toolkit_docker = TerminalToolkitDocker(
                                                    session_name = 'agent',
                                                    container_name = terminal._client_container_name,
                                                    timeout=60,
                                                    log_path=log_path,
                                                    container_log_path = '/logs',
                                                    disable_recording= True,
                                                    user = user,
                                                    safe_mode=False
                                                    )

    output = terminal_toolkit_docker.shell_exec(command="echo 'Hello, World!'")
    print(f"Output of echo command: {output}")

    output = terminal_toolkit_docker.session_view()
    print(f"Session view: {output}")


# exit()
    print("Terminal Toolkit Docker instance created successfully.")
    # Get tools from terminal toolkit docker
    print("Getting tools from Terminal Toolkit Docker...")
    tools = terminal_toolkit_docker.get_tools()
    # Define system message
    sys_msg = (
        "You are a System Administrator helping with log management tasks. "
        "You have access to terminal tools that can help you execute "
        "shell commands and search files. "
    )

    model_config_dict = ChatGPTConfig(
        temperature=0.0,
    ).as_dict()

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1_MINI,
        model_config_dict=model_config_dict,
    )

    # Set agent
    camel_agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        tools=tools,
    )
    camel_agent.reset()

    # Define a user message for creating logs directory
    usr_msg = (
        f"Task instruction: {task_instruction}"
    )
    print(f"User message: {usr_msg}")
    # Get response information
    response = camel_agent.step(usr_msg)
    print(str(response.info['tool_calls'])[:1000])

    usage = response.info['usage']
    print(f"Prompt tokens: {usage['prompt_tokens']}, completion tokens: {usage['completion_tokens']}")

    memory_list = camel_agent._memory._chat_history_block.storage.memory_list
    from typing import List, Tuple
    def create_timestamped_marker_from_memory(records: List[dict]) -> Tuple[float, str]:
        """Create a timestamped marker from memory records."""
        results = []
        print(f"Total records: {len(records)}")
        for record in records:
            if 'func_name' in record['message'].keys():
                timestamp = record['timestamp']
                func_name = record['message']['func_name']
                args = record['message'].get('args', {})
                if args:
                    command = args.get('command', '')
                else:
                    command = ''
                results.append((timestamp, f"Called tool: {func_name} with args: {command}"))
        return results

    timestamped_markers = create_timestamped_marker_from_memory(memory_list)
    print("\n".join(f"{ts}: {msg}" for ts, msg in timestamped_markers))




    # exit()
    print("Run test script inside the container...")
# ==========================================================================================
# 5. Copy the test script to the container and run it

    session = terminal.create_session(
                        "tests", is_active_stream=False, as_configured_user=False
                )
    def _setup_test_env(terminal: Terminal, trial_handler: TrialHandler) -> None:
        paths = [
            trial_handler.task_paths.run_tests_path,
        ]

        if trial_handler.task_paths.test_dir.exists():
            paths.append(trial_handler.task_paths.test_dir)

        terminal.copy_to_container(
            paths=paths,
            container_dir=str(terminal._compose_manager.CONTAINER_TEST_DIR),
        )

    def _run_tests(
        terminal: Terminal,
        session,
        trial_handler: TrialHandler,
    ) -> FailureMode:
        _setup_test_env(terminal, trial_handler)

        _global_test_timeout_sec = None
        if _global_test_timeout_sec:
            test_timeout_sec = _global_test_timeout_sec
        else:
            test_timeout_sec = (
                trial_handler.task.max_test_timeout_sec
                * 1.0
            )

        try:
            session.send_keys(
                [
                    "bash ",
                    str(
                        terminal._compose_manager.CONTAINER_TEST_DIR
                        / trial_handler.task_paths.run_tests_path.name
                    ),
                    "Enter",
                ],
                block=True,
                max_timeout_sec=test_timeout_sec,
            )
        except TimeoutError:
            self._logger.warning(
                "Test command timed out after "
                f"{test_timeout_sec}s for task "
                f"{trial_handler.task_id}."
            )

            return FailureMode.TEST_TIMEOUT

        return FailureMode.NONE


    test_failure_mode = _run_tests(
        terminal=terminal,
        session=session,
        trial_handler=trial_handler,
    )

    # import pdb
    # pdb.set_trace()
# 6. Copy the output files from the container to the host machine


# 7. Clean up the container and tmux session
# terminal.stop()