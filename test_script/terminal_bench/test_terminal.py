from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.terminal.models import TerminalCommand
from terminal_bench.terminal.terminal import spin_up_terminal
from pathlib import Path

task_path = Path(__file__).parent.parent.parent / "dataset" / "tbench-tasks" / "simple-web-scraper"
print(f"Using task path: {task_path}")
# Check if the task path exists
if not task_path.exists():
    print(f"Task path {task_path} does not exist. Please check the path.")
else:
    # print the contents of the task path
    print(f"Contents of task path {task_path}:")
    for item in task_path.iterdir():
        print(item)

trial_handler = TrialHandler(
    trial_name='test',
    input_path=task_path,
    output_path=Path("output"),
)

print(f"Trial handler initialized with task paths: {trial_handler.task_paths}")
print(f"Trial handler initialized with trial paths: {trial_handler.trial_paths}")
print(f"Trial handler initialized with client container name: {trial_handler.client_container_name}")
print(f"Trial handler initialized with client image name: {trial_handler.client_image_name}")
print(f"Trial handler initialized with docker image name prefix: {trial_handler.docker_image_name_prefix}")
print(f"Trial handler initialized with commands path: {trial_handler.trial_paths.commands_path}")

# exit()
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
    disable_recording=trial_handler.task.disable_asciinema,
) as terminal:
    session = terminal.create_session(
        "agent", is_active_stream=True, as_configured_user=True
    )
    print(f"Created session: {session._session_name}")

    session.send_command(TerminalCommand(command="echo 'Hello, World!'"))
    session.send_command(TerminalCommand(command="ls -la"))
    session.send_command(TerminalCommand(command="pwd"))