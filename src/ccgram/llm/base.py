"""Base types for LLM command generation.

Defines the protocol and result types that LLM command generators
must follow. Used by the shell provider to convert natural language
descriptions into shell commands.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class CommandResult:
    """Result of LLM command generation."""

    command: str
    explanation: str
    is_dangerous: bool = False


class CommandGenerator(Protocol):
    """Protocol for LLM-based shell command generators."""

    async def generate_command(
        self,
        description: str,
        *,
        cwd: str = "",
        shell: str = "",
        os_info: str = "",
        recent_output: str = "",
        shell_tools: str = "",
    ) -> CommandResult:
        """Generate a shell command from a natural language description.

        Args:
            description: Natural language description of the desired command.
            cwd: Current working directory for context.
            shell: Shell type (bash, zsh, fish, etc.).
            os_info: OS information string.
            recent_output: Recent terminal output for context.
            shell_tools: Available CLI tools and their descriptions.

        Returns:
            CommandResult with the generated command and explanation.
        """
        ...
