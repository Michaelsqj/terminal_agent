# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

import asyncio
import datetime
import os
import platform
import uuid

from camel.agents.chat_agent import ChatAgent
from camel.logger import get_logger
from camel.messages.base import BaseMessage
from camel.models import BaseModelBackend, ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks.task import Task
from camel.toolkits import (
    AgentCommunicationToolkit,
    NoteTakingToolkit,
    TerminalToolkit,
    ToolkitMessageIntegration,
)
from camel.types import ModelPlatformType, ModelType
from camel.utils.commons import api_keys_required

logger = get_logger(__name__)

WORKING_DIRECTORY = os.environ.get("CAMEL_WORKDIR") or os.path.abspath(
    "working_dir/"
)


def send_message_to_user(
    message_title: str,
    message_description: str,
    message_attachment: str = "",
) -> str:
    r"""Use this tool to send a tidy message to the user, including a
    short title, a one-sentence description, and an optional attachment.

    This one-way tool keeps the user informed about your progress,
    decisions, or actions. It does not require a response.
    You should use it to:
    - Announce what you are about to do.
      For example:
      message_title="Starting Task"
      message_description="Searching for papers on GUI Agents."
    - Report the result of an action.
      For example:
      message_title="Search Complete"
      message_description="Found 15 relevant papers."
    - Report a created file.
      For example:
      message_title="File Ready"
      message_description="The report is ready for your review."
      message_attachment="report.pdf"
    - State a decision.
      For example:
      message_title="Next Step"
      message_description="Analyzing the top 10 papers."
    - Give a status update during a long-running task.

    Args:
        message_title (str): The title of the message.
        message_description (str): The short description.
        message_attachment (str): The attachment of the message,
            which can be a file path or a URL.

    Returns:
        str: Confirmation that the message was successfully sent.
    """
    print(f"\nAgent Message:\n{message_title} " f"\n{message_description}\n")
    if message_attachment:
        print(message_attachment)
    logger.info(
        f"\nAgent Message:\n{message_title} "
        f"{message_description} {message_attachment}"
    )
    return (
        f"Message successfully sent to user: '{message_title} "
        f"{message_description} {message_attachment}'"
    )


def developer_agent_factory(
    model: BaseModelBackend,
    task_id: str,
):
    r"""Factory for creating a developer agent."""
    # Initialize message integration
    message_integration = ToolkitMessageIntegration(
        message_handler=send_message_to_user
    )

    # Initialize toolkits
    terminal_toolkit = TerminalToolkit(safe_mode=True, clone_current_env=False)
    note_toolkit = NoteTakingToolkit(working_directory=WORKING_DIRECTORY)

    # Add messaging to toolkits
    terminal_toolkit = message_integration.register_toolkits(terminal_toolkit)
    note_toolkit = message_integration.register_toolkits(note_toolkit)

    # Get enhanced tools
    tools = [
        *terminal_toolkit.get_tools(),
        *note_toolkit.get_tools(),
    ]

    system_message = f"""
<role>
You are a Lead Software Engineer, a master-level coding assistant with a 
powerful and unrestricted terminal. Your primary role is to solve any 
technical task by writing and executing code, installing necessary libraries, 
interacting with the operating system, and deploying applications. You are the 
team's go-to expert for all technical implementation.
</role>

<team_structure>
You collaborate with the following agents who can work in parallel:
- **Senior Research Analyst**: Gathers information from the web to support 
your development tasks.
- **Documentation Specialist**: Creates and manages technical and user-facing 
documents.
- **Creative Content Specialist**: Handles image, audio, and video processing 
and generation.
</team_structure>

<operating_environment>
- **System**: {platform.system()} ({platform.machine()})
- **Working Directory**: `{WORKING_DIRECTORY}`. All local file operations must 
occur here, but you can access files from any place in the file system. For 
all file system operations, you MUST use absolute paths to ensure precision 
and avoid ambiguity.
- **Current Date**: {datetime.date.today()}.
</operating_environment>

<mandatory_instructions>
- You MUST use the `read_note` tool to read the notes from other agents.

- When you complete your task, your final response must be a comprehensive
summary of your work and the outcome, presented in a clear, detailed, and
easy-to-read format. Avoid using markdown tables for presenting data; use
plain text formatting instead.
<mandatory_instructions>

<capabilities>
Your capabilities are extensive and powerful:
- **Unrestricted Code Execution**: You can write and execute code in any
  language to solve a task. You MUST first save your code to a file (e.g.,
  `script.py`) and then run it from the terminal (e.g.,
  `python script.py`).
- **Full Terminal Control**: You have root-level access to the terminal. You
  can run any command-line tool, manage files, and interact with the OS. If
  a tool is missing, you MUST install it with the appropriate package manager
  (e.g., `pip3`, `uv`, or `apt-get`). Your capabilities include:
    - **Text & Data Processing**: `awk`, `sed`, `grep`, `jq`.
    - **File System & Execution**: `find`, `xargs`, `tar`, `zip`, `unzip`,
      `chmod`.
    - **Networking & Web**: `curl`, `wget` for web requests; `ssh` for
      remote access.
- **Screen Observation**: You can take screenshots to analyze GUIs and visual
  context, enabling you to perform tasks that require sight.
- **Desktop Automation**: You can control desktop applications
  programmatically.
  - **On macOS**, you MUST prioritize using **AppleScript** for its robust
    control over native applications. Execute simple commands with
    `osascript -e '...'` or run complex scripts from a `.scpt` file.
  - **On other systems**, use **pyautogui** for cross-platform GUI
    automation.
  - **IMPORTANT**: Always complete the full automation workflow—do not just
    prepare or suggest actions. Execute them to completion.
- **Solution Verification**: You can immediately test and verify your
  solutions by executing them in the terminal.
- **Web Deployment**: You can deploy web applications and content, serve
  files, and manage deployments.
- **Human Collaboration**: If you are stuck or need clarification, you can
  ask for human input via the console.
- **Note Management**: You can write and read notes to coordinate with other
  agents and track your work.
</capabilities>

<philosophy>
- **Bias for Action**: Your purpose is to take action. Don't just suggest
solutions—implement them. Write code, run commands, and build things.
- **Complete the Full Task**: When automating GUI applications, always finish
what you start. If the task involves sending something, send it. If it
involves submitting data, submit it. Never stop at just preparing or
drafting—execute the complete workflow to achieve the desired outcome.
- **Embrace Challenges**: Never say "I can't." If you
encounter a limitation, find a way to overcome it.
- **Resourcefulness**: If a tool is missing, install it. If information is
lacking, find it. You have the full power of a terminal to acquire any
resource you need.
- **Think Like an Engineer**: Approach problems methodically. Analyze
requirements, execute it, and verify the results. Your
strength lies in your ability to engineer solutions.
</philosophy>

<terminal_tips>
The terminal tools are session-based, identified by a unique `id`. Master
these tips to maximize your effectiveness:

- **GUI Automation Strategy**:
  - **AppleScript (macOS Priority)**: For robust control of macOS apps, use
    `osascript`.
    - Example (open Slack):
      `osascript -e 'tell application "Slack" to activate'`
    - Example (run script file): `osascript my_script.scpt`
  - **pyautogui (Cross-Platform)**: For other OSes or simple automation.
    - Key functions: `pyautogui.click(x, y)`, `pyautogui.typewrite("text")`,
      `pyautogui.hotkey('ctrl', 'c')`, `pyautogui.press('enter')`.
    - Safety: Always use `time.sleep()` between actions to ensure stability
      and add `pyautogui.FAILSAFE = True` to your scripts.
    - Workflow: Your scripts MUST complete the entire task, from start to
      final submission.

- **Command-Line Best Practices**:
  - **Be Creative**: The terminal is your most powerful tool. Use it boldly.
  - **Automate Confirmation**: Use `-y` or `-f` flags to avoid interactive
    prompts.
  - **Manage Output**: Redirect long outputs to a file (e.g., `> output.txt`).
  - **Chain Commands**: Use `&&` to link commands for sequential execution.
  - **Piping**: Use `|` to pass output from one command to another.
  - **Permissions**: Use `ls -F` to check file permissions.
  - **Installation**: Use `pip3 install` or `apt-get install` for new
    packages.

- Stop a Process: If a process needs to be terminated, use
    `shell_kill_process(id="...")`.
</terminal_tips>

<collaboration_and_assistance>
- If you get stuck, encounter an issue you cannot solve (like a CAPTCHA),
    or need clarification, use the `ask_human_via_console` tool.
- Document your progress and findings in notes so other agents can build
    upon your work.
</collaboration_and_assistance>
    """

    return ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Developer Agent",
            content=system_message,
        ),
        model=model,
        tools=tools,
        toolkits_to_register_agent=[screenshot_toolkit],
    )



async def main():
    # Ensure working directory exists
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)

    # Initialize the AgentCommunicationToolkit
    msg_toolkit = AgentCommunicationToolkit(max_message_history=100)

    # Initialize message integration for use in coordinator and task agents
    message_integration = ToolkitMessageIntegration(
        message_handler=send_message_to_user
    )

    # await google_drive_mcp_toolkit.connect()

    # Create a single model backend for all agents
    model_backend = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1,
        model_config_dict={
            "stream": False,
        },
    )

    model_backend_reason = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1,
        model_config_dict={
            "stream": False,
        },
    )

    task_id = 'workforce_task'

    # Create custom agents for the workforce
    coordinator_agent = ChatAgent(
        system_message=(
            f""""
You are a helpful coordinator.
- You are now working in system {platform.system()} with architecture
{platform.machine()} at working directory `{WORKING_DIRECTORY}`. All local
file operations must occur here, but you can access files from any place in
the file system. For all file system operations, you MUST use absolute paths
to ensure precision and avoid ambiguity.
The current date is {datetime.date.today()}. For any date-related tasks, you 
MUST use this as the current date.

- If a task assigned to another agent fails, you should re-assign it to the 
`Developer_Agent`. The `Developer_Agent` is a powerful agent with terminal 
access and can resolve a wide range of issues. 
            """
        ),
        model=model_backend_reason,
        tools=[
            *NoteTakingToolkit(
                working_directory=WORKING_DIRECTORY
            ).get_tools(),
        ],
    )
    task_agent = ChatAgent(
        f"""

You are a helpful task planner.
- You are now working in system {platform.system()} with architecture
{platform.machine()} at working directory `{WORKING_DIRECTORY}`. All local
file operations must occur here, but you can access files from any place in
the file system. For all file system operations, you MUST use absolute paths
to ensure precision and avoid ambiguity.
The current date is {datetime.date.today()}. For any date-related tasks, you 
MUST use this as the current date.
        """,
        model=model_backend_reason,
        tools=[
            *NoteTakingToolkit(
                working_directory=WORKING_DIRECTORY
            ).get_tools(),
        ],
    )
    new_worker_agent = ChatAgent(
        f"You are a helpful worker. When you complete your task, your "
        "final response "
        f"must be a comprehensive summary of your work, presented in a clear, "
        f"detailed, and easy-to-read format. Avoid using markdown tables for "
        f"presenting data; use plain text formatting instead. You are now "
        f"working in "
        f"`{WORKING_DIRECTORY}` All local file operations must occur here, "
        f"but you can access files from any place in the file system. For all "
        f"file system operations, you MUST use absolute paths to ensure "
        f"precision and avoid ambiguity."
        "directory. You can also communicate with other agents "
        "using messaging tools - use `list_available_agents` to see "
        "available team members and `send_message` to coordinate work "
        "and ask for help when needed. "
        "### Note-Taking: You have access to comprehensive note-taking tools "
        "for documenting work progress and collaborating with team members. "
        "Use create_note, append_note, read_note, and list_note to track "
        "your work, share findings, and access information from other agents. "
        "Create notes for work progress, discoveries, and collaboration "
        "points.",
        model=model_backend,
        tools=[
            *message_integration.register_toolkits(
                NoteTakingToolkit(working_directory=WORKING_DIRECTORY)
            ).get_tools(),
        ],
    )

    # Create agents using factory functions
    developer_agent = developer_agent_factory(
        model_backend_reason,
        task_id,
    )


    msg_toolkit.register_agent("Worker", new_worker_agent)
    msg_toolkit.register_agent("Developer_Agent", developer_agent)

    # Create workforce instance before adding workers
    workforce = Workforce(
        'A workforce',
        graceful_shutdown_timeout=30.0,  # 30 seconds for debugging
        share_memory=False,
        coordinator_agent=coordinator_agent,
        task_agent=task_agent,
        new_worker_agent=new_worker_agent,
        use_structured_output_handler=False,
        task_timeout_seconds=900.0,
    )

    workforce.add_single_agent_worker(
        "Developer Agent: A master-level coding assistant with a powerful "
        "terminal. It can write and execute code, manage files, automate "
        "desktop tasks, and deploy web applications to solve complex "
        "technical challenges.",
        worker=developer_agent,
    )

    # specify the task to be solved
    human_task = Task(
        content=(
            """
search 50 different papers related to llm agent and write a html report about 
them.
            """
        ),
        id='0',
    )

    # Use the async version directly to avoid hanging with async tools
    await workforce.process_task_async(human_task)

    # Test WorkforceLogger features
    print("\n--- Workforce Log Tree ---")
    print(workforce.get_workforce_log_tree())

    print("\n--- Workforce KPIs ---")
    kpis = workforce.get_workforce_kpis()
    for key, value in kpis.items():
        print(f"{key}: {value}")

    log_file_path = "eigent_logs.json"
    print(f"\n--- Dumping Workforce Logs to {log_file_path} ---")
    workforce.dump_workforce_logs(log_file_path)
    print(f"Logs dumped. Please check the file: {log_file_path}")


if __name__ == "__main__":
    asyncio.run(main())
