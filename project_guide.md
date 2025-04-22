# Project Guide: openmanus

## Project Overview

### Entry Points
- **Package**: `run_flow.py`

## File Structure
```
├── app\__init__.py
├── app\agent\__init__.py
├── app\agent\base.py
├── app\agent\browser.py
├── app\agent\manus.py
├── app\agent\mcp.py
├── app\agent\react.py
├── app\agent\swe.py
├── app\agent\toolcall.py
├── app\bedrock.py
├── app\cache.py
├── app\config.py
├── app\exceptions.py
├── app\flow\__init__.py
├── app\flow\base.py
├── app\flow\flow_factory.py
├── app\flow\planning.py
├── app\llm.py
├── app\logger.py
├── app\mcp\__init__.py
├── app\mcp\server.py
├── app\prompt\__init__.py
├── app\prompt\browser.py
├── app\prompt\cot.py
├── app\prompt\manus.py
├── app\prompt\mcp.py
├── app\prompt\planning.py
├── app\prompt\swe.py
├── app\prompt\toolcall.py
├── app\sandbox\__init__.py
├── app\sandbox\client.py
├── app\sandbox\core\exceptions.py
├── app\sandbox\core\manager.py
├── app\sandbox\core\sandbox.py
├── app\sandbox\core\terminal.py
├── app\schema.py
├── app\tool\__init__.py
├── app\tool\ask_human.py
├── app\tool\base.py
├── app\tool\bash.py
├── app\tool\browser_use_tool.py
├── app\tool\cache_management.py
├── app\tool\create_chat_completion.py
├── app\tool\deep_research.py
├── app\tool\enhanced_web_search.py
├── app\tool\file_operators.py
├── app\tool\mcp.py
├── app\tool\planning.py
├── app\tool\python_execute.py
├── app\tool\search\__init__.py
├── app\tool\search\baidu_search.py
├── app\tool\search\base.py
├── app\tool\search\bing_search.py
├── app\tool\search\duckduckgo_search.py
├── app\tool\search\google_search.py
├── app\tool\str_replace_editor.py
├── app\tool\terminate.py
├── app\tool\tool_collection.py
├── app\tool\web_search.py
├── examples\benchmarks\__init__.py
├── main.py
├── run_flow.py
├── run_mcp.py
├── run_mcp_server.py
├── setup.py
├── tests\sandbox\test_client.py
├── tests\sandbox\test_docker_terminal.py
├── tests\sandbox\test_sandbox.py
└── tests\sandbox\test_sandbox_manager.py
```

## Key Modules

### app\agent\manus.py
This Python file defines the `Manus` class, a versatile agent designed for general-purpose task solving using both local and remote tools accessed via an MCP (Message Communication Protocol) system. The agent extends `ToolCallAgent` and manages a collection of tools including Python execution, web search, human interaction, and specialized MCP client tools.  It handles connecting to and disconnecting from MCP servers, dynamically adding/removing their associated tools, and includes logic for browser context management during operation, ensuring proper tool selection and prompt formatting based on recent interactions.

<details><summary>Imports</summary>

- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `pydantic.Field` as `Field`
- `pydantic.model_validator` as `model_validator`
- `app.agent.browser.BrowserContextHelper` as `BrowserContextHelper`
- `app.agent.toolcall.ToolCallAgent` as `ToolCallAgent`
- `app.config.config` as `config`
- `app.logger.logger` as `logger`
- `app.prompt.manus.NEXT_STEP_PROMPT` as `NEXT_STEP_PROMPT`
- `app.prompt.manus.SYSTEM_PROMPT` as `SYSTEM_PROMPT`
- `app.tool.Terminate` as `Terminate`
- `app.tool.ToolCollection` as `ToolCollection`
- `app.tool.ask_human.AskHuman` as `AskHuman`
- `app.tool.browser_use_tool.BrowserUseTool` as `BrowserUseTool`
- `app.tool.mcp.MCPClients` as `MCPClients`
- `app.tool.mcp.MCPClientTool` as `MCPClientTool`
- `app.tool.python_execute.PythonExecute` as `PythonExecute`
- `app.tool.str_replace_editor.StrReplaceEditor` as `StrReplaceEditor`
- `app.tool.cache_management.CacheManagement` as `CacheManagement`
- `app.tool.enhanced_web_search.EnhancedWebSearch` as `EnhancedWebSearch`
</details>

#### Classes

<details><summary><code>Manus</code></summary>

A versatile general-purpose agent with support for both local and MCP tools.

Inherits from: `ToolCallAgent`

Methods:
- `initialize_helper(self) -> Manus`
  - Initialize basic components synchronously.
</details>

#### Functions
- `initialize_helper(self) -> Manus`
  - Initialize basic components synchronously.

### app\llm.py
This is a very comprehensive and well-structured implementation of an OpenAI API client with caching and retry mechanisms. Here's a breakdown of the code's strengths, potential improvements, and key considerations:

**Strengths:**

*   **Comprehensive Functionality:** The client supports a wide range of OpenAI features, including chat completions, function/tool calls, image support (multimodal models), streaming responses, caching, and retry mechanisms.
*   **Robust Error Handling:**  The code includes robust error handling with specific exception types for token limits, API errors, and validation errors. It also implements retry logic to handle transient API issues.
*   **Caching Mechanism:** The client incorporates a caching mechanism to reduce latency and API costs.

<details><summary>Imports</summary>

- `math`
- `typing.Any` as `Any`
- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `typing.Union` as `Union`
- `app.cache.PromptCache` as `PromptCache`
- `app.config.PROJECT_ROOT` as `PROJECT_ROOT`
- `tiktoken`
- `openai.APIError` as `APIError`
- `openai.AsyncAzureOpenAI` as `AsyncAzureOpenAI`
- `openai.AsyncOpenAI` as `AsyncOpenAI`
- `openai.AuthenticationError` as `AuthenticationError`
- `openai.OpenAIError` as `OpenAIError`
- `openai.RateLimitError` as `RateLimitError`
- `openai.types.chat.ChatCompletion` as `ChatCompletion`
- `openai.types.chat.ChatCompletionMessage` as `ChatCompletionMessage`
- `tenacity.retry` as `retry`
- `tenacity.retry_if_exception_type` as `retry_if_exception_type`
- `tenacity.stop_after_attempt` as `stop_after_attempt`
- `tenacity.wait_random_exponential` as `wait_random_exponential`
- `app.bedrock.BedrockClient` as `BedrockClient`
- `app.config.LLMSettings` as `LLMSettings`
- `app.config.config` as `config`
- `app.exceptions.TokenLimitExceeded` as `TokenLimitExceeded`
- `app.logger.logger` as `logger`
- `app.schema.ROLE_VALUES` as `ROLE_VALUES`
- `app.schema.TOOL_CHOICE_TYPE` as `TOOL_CHOICE_TYPE`
- `app.schema.TOOL_CHOICE_VALUES` as `TOOL_CHOICE_VALUES`
- `app.schema.Message` as `Message`
- `app.schema.ToolChoice` as `ToolChoice`
</details>

#### Classes

<details><summary><code>TokenCounter</code></summary>


Methods:
- `__init__(self, tokenizer)`
- `count_text(self, text) -> int`
  - Calculate tokens for a text string
- `count_image(self, image_item) -> int`
  - Calculate tokens for an image based on detail level and dimensions
- `count_content(self, content) -> int`
  - Calculate tokens for message content
- `count_tool_calls(self, tool_calls) -> int`
  - Calculate tokens for tool calls
- `count_message_tokens(self, messages) -> int`
  - Calculate the total number of tokens in a message list
</details>

<details><summary><code>LLM</code></summary>


Methods:
- `__init__(self, config_name, llm_config)`
- `count_tokens(self, text) -> int`
  - Calculate the number of tokens in a text
- `count_message_tokens(self, messages) -> int`
- `update_token_count(self, input_tokens, completion_tokens) -> None`
  - Update token counts
- `check_token_limit(self, input_tokens) -> bool`
  - Check if token limits are exceeded
- `get_limit_error_message(self, input_tokens) -> str`
  - Generate error message for token limit exceeded
- `format_messages(messages, supports_images) -> List[dict]`
  - Format messages for LLM by converting them to OpenAI message format.
- `get_cache_stats(self) -> Dict[str, Any]`
  - Get statistics about the cache.
</details>

#### Functions
- `count_text(self, text) -> int`
  - Calculate tokens for a text string
- `count_image(self, image_item) -> int`
  - Calculate tokens for an image based on detail level and dimensions
- `count_content(self, content) -> int`
  - Calculate tokens for message content
- `count_tool_calls(self, tool_calls) -> int`
  - Calculate tokens for tool calls
- `count_message_tokens(self, messages) -> int`
- `count_tokens(self, text) -> int`
  - Calculate the number of tokens in a text
- `update_token_count(self, input_tokens, completion_tokens) -> None`
  - Update token counts
- `check_token_limit(self, input_tokens) -> bool`
  - Check if token limits are exceeded
- `get_limit_error_message(self, input_tokens) -> str`
  - Generate error message for token limit exceeded
- `format_messages(messages, supports_images) -> List[dict]`
  - Format messages for LLM by converting them to OpenAI message format.
- `get_cache_stats(self) -> Dict[str, Any]`
  - Get statistics about the cache.

### app\agent\toolcall.py
```python
import asyncio
import json
import logging
from typing import Any, Optional

# Configure logging (adjust level as needed)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Message:
    """Represents a message in the agent's memory."""

    def __init__(self, content: str, tool_call_id: Optional[str] = None, name: Optional[str] = None, base64_image: Optional[str] = None):
        self.content = content
        self.tool_call_id = tool_call_id
        self.name =

<details><summary>Imports</summary>

- `asyncio`
- `json`
- `re`
- `typing.Any` as `Any`
- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `typing.Union` as `Union`
- `pydantic.Field` as `Field`
- `app.agent.react.ReActAgent` as `ReActAgent`
- `app.exceptions.TokenLimitExceeded` as `TokenLimitExceeded`
- `app.logger.logger` as `logger`
- `app.prompt.toolcall.NEXT_STEP_PROMPT` as `NEXT_STEP_PROMPT`
- `app.prompt.toolcall.SYSTEM_PROMPT` as `SYSTEM_PROMPT`
- `app.schema.TOOL_CHOICE_TYPE` as `TOOL_CHOICE_TYPE`
- `app.schema.AgentState` as `AgentState`
- `app.schema.Message` as `Message`
- `app.schema.ToolCall` as `ToolCall`
- `app.schema.ToolChoice` as `ToolChoice`
- `app.tool.CreateChatCompletion` as `CreateChatCompletion`
- `app.tool.Terminate` as `Terminate`
- `app.tool.ToolCollection` as `ToolCollection`
- `uuid.uuid4` as `uuid4`
</details>

#### Classes

<details><summary><code>ToolCallAgent</code></summary>

Base agent class for handling tool/function calls with enhanced abstraction

Inherits from: `ReActAgent`

Methods:
</details>

#### Functions

### app\agent\mcp.py
This Python file defines `MCPAgent`, an agent designed to interact with servers adhering to the Model Context Protocol (MCP). The agent establishes a connection—either via SSE or stdio—to an MCP server, exposing its tools for use within the agent's workflow.  It dynamically manages available tools by periodically refreshing them from the server and includes specific handling for multimedia responses and a "terminate" tool to end execution, ensuring proper cleanup of the connection when finished.

<details><summary>Imports</summary>

- `typing.Any` as `Any`
- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `typing.Tuple` as `Tuple`
- `pydantic.Field` as `Field`
- `app.agent.toolcall.ToolCallAgent` as `ToolCallAgent`
- `app.logger.logger` as `logger`
- `app.prompt.mcp.MULTIMEDIA_RESPONSE_PROMPT` as `MULTIMEDIA_RESPONSE_PROMPT`
- `app.prompt.mcp.NEXT_STEP_PROMPT` as `NEXT_STEP_PROMPT`
- `app.prompt.mcp.SYSTEM_PROMPT` as `SYSTEM_PROMPT`
- `app.schema.AgentState` as `AgentState`
- `app.schema.Message` as `Message`
- `app.tool.base.ToolResult` as `ToolResult`
- `app.tool.mcp.MCPClients` as `MCPClients`
</details>

#### Classes

<details><summary><code>MCPAgent</code></summary>

Agent for interacting with MCP (Model Context Protocol) servers.

Inherits from: `ToolCallAgent`

Methods:
</details>

#### Functions

### app\sandbox\__init__.py
This Python file defines a Docker-based sandbox environment for securely executing untrusted code. It provides classes like `DockerSandbox` and `SandboxManager` to manage container creation, resource limits, and execution, along with client interfaces (`BaseSandboxClient`, `LocalSandboxClient`) for interacting with the sandbox.  The module also includes custom exceptions for handling potential sandbox-related errors such as timeouts or resource exhaustion.

<details><summary>Imports</summary>

- `app.sandbox.client.BaseSandboxClient` as `BaseSandboxClient`
- `app.sandbox.client.LocalSandboxClient` as `LocalSandboxClient`
- `app.sandbox.client.create_sandbox_client` as `create_sandbox_client`
- `app.sandbox.core.exceptions.SandboxError` as `SandboxError`
- `app.sandbox.core.exceptions.SandboxResourceError` as `SandboxResourceError`
- `app.sandbox.core.exceptions.SandboxTimeoutError` as `SandboxTimeoutError`
- `app.sandbox.core.manager.SandboxManager` as `SandboxManager`
- `app.sandbox.core.sandbox.DockerSandbox` as `DockerSandbox`
</details>

### app\tool\deep_research.py
This is a remarkably well-structured and comprehensive implementation of a deep research agent! The code demonstrates excellent design principles, including modularity, error handling, and the use of asynchronous programming. Here's a detailed breakdown of its strengths and potential areas for improvement:

**Strengths:**

*   **Modularity:**  The code is broken down into well-defined functions (e.g., `_search_web`, `_extract_insights`, `_generate_follow_ups`), making it easy to understand, maintain, and extend.
*   **Asynchronous Programming:** The use of `async` and `await` allows the agent to perform multiple tasks concurrently, significantly improving its efficiency.  This is crucial for web

<details><summary>Imports</summary>

- `asyncio`
- `json`
- `re`
- `time`
- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `typing.Set` as `Set`
- `pydantic.BaseModel` as `BaseModel`
- `pydantic.ConfigDict` as `ConfigDict`
- `pydantic.Field` as `Field`
- `pydantic.model_validator` as `model_validator`
- `app.exceptions.ToolError` as `ToolError`
- `app.llm.LLM` as `LLM`
- `app.logger.logger` as `logger`
- `app.schema.ToolChoice` as `ToolChoice`
- `app.tool.base.BaseTool` as `BaseTool`
- `app.tool.base.ToolResult` as `ToolResult`
- `app.tool.web_search.SearchResult` as `SearchResult`
- `app.tool.web_search.WebSearch` as `WebSearch`
</details>

#### Classes

<details><summary><code>ResearchInsight</code></summary>

A single insight discovered during research.

Inherits from: `BaseModel`

Methods:
</details>

<details><summary><code>ResearchContext</code></summary>

Research context for tracking research progress.

Inherits from: `BaseModel`
</details>

<details><summary><code>ResearchSummary</code></summary>

Comprehensive summary of deep research results.

Inherits from: `ToolResult`

Methods:
- `populate_output(self) -> ResearchSummary`
  - Populate the output field after validation.
</details>

<details><summary><code>DeepResearch</code></summary>

Advanced research tool that explores a topic through iterative web searches.

Inherits from: `BaseTool`
</details>

#### Functions
- `populate_output(self) -> ResearchSummary`
  - Populate the output field after validation.

### app\tool\str_replace_editor.py
```python
import re
from typing import List, Optional

SNIPPET_LINES = 5
MAX_FILE_SIZE = 1024 * 1024  # 1MB


def maybe_truncate(text: str) -> str:
    """Truncate text if it exceeds MAX_FILE_SIZE."""
    if len(text) > MAX_FILE_SIZE:
        return text[:MAX_FILE_SIZE] + "\n... (truncated)"
    return text


class FileOperator:
    """Interface for file operations."""

    async def read_file(self, path: str) -> str:
        """Read the content of a file."""

<details><summary>Imports</summary>

- `collections.defaultdict` as `defaultdict`
- `pathlib.Path` as `Path`
- `typing.Any` as `Any`
- `typing.DefaultDict` as `DefaultDict`
- `typing.List` as `List`
- `typing.Literal` as `Literal`
- `typing.Optional` as `Optional`
- `typing.get_args` as `get_args`
- `app.config.config` as `config`
- `app.exceptions.ToolError` as `ToolError`
- `app.tool.BaseTool` as `BaseTool`
- `app.tool.base.CLIResult` as `CLIResult`
- `app.tool.base.ToolResult` as `ToolResult`
- `app.tool.file_operators.FileOperator` as `FileOperator`
- `app.tool.file_operators.LocalFileOperator` as `LocalFileOperator`
- `app.tool.file_operators.PathLike` as `PathLike`
- `app.tool.file_operators.SandboxFileOperator` as `SandboxFileOperator`
</details>

#### Classes

<details><summary><code>StrReplaceEditor</code></summary>

A tool for viewing, creating, and editing files with sandbox support.

Inherits from: `BaseTool`

Methods:
</details>

#### Functions
- `maybe_truncate(content, truncate_after) -> str`
  - Truncate content and append a notice if content exceeds the specified length.

### app\agent\base.py
This Python file defines an abstract base class `BaseAgent` for building autonomous agents. It establishes a foundational structure for managing agent state (like IDLE, RUNNING, FINISHED), handling memory via a `Memory` object, and executing a step-by-step process defined by the abstract method `step()`.  The class includes features like context management for safe state transitions, methods for updating memory with messages, and mechanisms to detect and handle "stuck" states where the agent repeats itself.

<details><summary>Imports</summary>

- `abc.ABC` as `ABC`
- `abc.abstractmethod` as `abstractmethod`
- `contextlib.asynccontextmanager` as `asynccontextmanager`
- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `pydantic.BaseModel` as `BaseModel`
- `pydantic.Field` as `Field`
- `pydantic.model_validator` as `model_validator`
- `app.llm.LLM` as `LLM`
- `app.logger.logger` as `logger`
- `app.sandbox.client.SANDBOX_CLIENT` as `SANDBOX_CLIENT`
- `app.schema.ROLE_TYPE` as `ROLE_TYPE`
- `app.schema.AgentState` as `AgentState`
- `app.schema.Memory` as `Memory`
- `app.schema.Message` as `Message`
</details>

#### Classes

<details><summary><code>BaseAgent</code></summary>

Abstract base class for managing agent state and execution.

Inherits from: `BaseModel`, `ABC`

Methods:
- `initialize_agent(self) -> BaseAgent`
  - Initialize agent with default settings if not provided.
- `update_memory(self, role, content, base64_image) -> None`
  - Add a message to the agent's memory.
- `handle_stuck_state(self)`
  - Handle stuck state by adding a prompt to change strategy
- `is_stuck(self) -> bool`
  - Check if the agent is stuck in a loop by detecting duplicate content
- `messages(self, value)`
  - Set the list of messages in the agent's memory.
</details>

<details><summary><code>Config</code></summary>

</details>

#### Functions
- `initialize_agent(self) -> BaseAgent`
  - Initialize agent with default settings if not provided.
- `update_memory(self, role, content, base64_image) -> None`
  - Add a message to the agent's memory.
- `handle_stuck_state(self)`
  - Handle stuck state by adding a prompt to change strategy
- `is_stuck(self) -> bool`
  - Check if the agent is stuck in a loop by detecting duplicate content
- `messages(self, value)`
  - Set the list of messages in the agent's memory.

### app\agent\browser.py
This Python file defines a `BrowserAgent` that leverages a browser to perform tasks, along with a helper class `BrowserContextHelper` for managing browser state and interactions. The `BrowserAgent` extends `ToolCallAgent` and uses tools like `BrowserUseTool` and `Terminate` to navigate web pages, extract content, and ultimately achieve its goals, incorporating current browser information into its decision-making process via the formatted prompts.  The `BrowserContextHelper` retrieves browser state (URL, tabs, screenshots) and prepares it for inclusion in the agent's prompt, ensuring context awareness during each step of operation.

<details><summary>Imports</summary>

- `json`
- `typing.TYPE_CHECKING` as `TYPE_CHECKING`
- `typing.Optional` as `Optional`
- `pydantic.Field` as `Field`
- `pydantic.model_validator` as `model_validator`
- `app.agent.toolcall.ToolCallAgent` as `ToolCallAgent`
- `app.logger.logger` as `logger`
- `app.prompt.browser.NEXT_STEP_PROMPT` as `NEXT_STEP_PROMPT`
- `app.prompt.browser.SYSTEM_PROMPT` as `SYSTEM_PROMPT`
- `app.schema.Message` as `Message`
- `app.schema.ToolChoice` as `ToolChoice`
- `app.tool.BrowserUseTool` as `BrowserUseTool`
- `app.tool.Terminate` as `Terminate`
- `app.tool.ToolCollection` as `ToolCollection`
- `app.agent.base.BaseAgent` as `BaseAgent`
</details>

#### Classes

<details><summary><code>BrowserContextHelper</code></summary>


Methods:
- `__init__(self, agent)`
</details>

<details><summary><code>BrowserAgent</code></summary>

A browser agent that uses the browser_use library to control a browser.

Inherits from: `ToolCallAgent`

Methods:
- `initialize_helper(self) -> BrowserAgent`
</details>

#### Functions
- `initialize_helper(self) -> BrowserAgent`

### app\flow\planning.py
```python
from enum import Enum, auto

class PlanStepStatus(Enum):
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    BLOCKED = auto()
    COMPLETED = auto()

    @classmethod
    def get_active_statuses(cls):
        return [cls.IN_PROGRESS, cls.BLOCKED]

    @classmethod
    def get_all_statuses(cls):
        return [cls.NOT_STARTED, cls.IN_PROGRESS, cls.BLOCKED, cls.COMPLETED]

    @classmethod
    def get_status_marks(cls):
        return {
            cls.NOT_STARTED: "☐",

<details><summary>Imports</summary>

- `json`
- `re`
- `time`
- `sys`
- `enum.Enum` as `Enum`
- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `typing.Union` as `Union`
- `pydantic.Field` as `Field`
- `app.agent.base.BaseAgent` as `BaseAgent`
- `app.flow.base.BaseFlow` as `BaseFlow`
- `app.llm.LLM` as `LLM`
- `app.logger.logger` as `logger`
- `app.schema.AgentState` as `AgentState`
- `app.schema.Message` as `Message`
- `app.schema.ToolChoice` as `ToolChoice`
- `app.tool.PlanningTool` as `PlanningTool`
</details>

#### Classes

<details><summary><code>PlanStepStatus</code></summary>

Enum class defining possible statuses of a plan step

Inherits from: `str`, `Enum`

Methods:
- `get_all_statuses(cls) -> list[str]`
  - Return a list of all possible step status values
- `get_active_statuses(cls) -> list[str]`
  - Return a list of values representing active statuses (not started or in progress)
- `get_status_marks(cls) -> Dict[str, str]`
  - Return a mapping of statuses to their marker symbols
</details>

<details><summary><code>PlanningFlow</code></summary>

A flow that manages planning and execution of tasks using agents.

Inherits from: `BaseFlow`

Methods:
- `__init__(self, agents)`
- `get_executor(self, step_type) -> BaseAgent`
  - Get an appropriate executor agent for the current step.
</details>

#### Functions
- `get_all_statuses(cls) -> list[str]`
  - Return a list of all possible step status values
- `get_active_statuses(cls) -> list[str]`
  - Return a list of values representing active statuses (not started or in progress)
- `get_status_marks(cls) -> Dict[str, str]`
  - Return a mapping of statuses to their marker symbols
- `get_executor(self, step_type) -> BaseAgent`
  - Get an appropriate executor agent for the current step.

### app\mcp\server.py
This Python file defines an `MCPServer` class that manages a set of tools accessible through a FastMCP server. The core functionality revolves around registering these tools—including built-in options like Bash, Browser control, and text editing—and making them available for execution via the MCP server interface.  It handles tool registration with parameter validation (using JSON schema), docstring/signature building, and includes cleanup routines, ultimately running the server in either `stdio` mode based on command-line arguments.

<details><summary>Imports</summary>

- `logging`
- `sys`
- `argparse`
- `asyncio`
- `atexit`
- `json`
- `inspect.Parameter` as `Parameter`
- `inspect.Signature` as `Signature`
- `typing.Any` as `Any`
- `typing.Dict` as `Dict`
- `typing.Optional` as `Optional`
- `mcp.server.fastmcp.FastMCP` as `FastMCP`
- `app.logger.logger` as `logger`
- `app.tool.base.BaseTool` as `BaseTool`
- `app.tool.bash.Bash` as `Bash`
- `app.tool.browser_use_tool.BrowserUseTool` as `BrowserUseTool`
- `app.tool.str_replace_editor.StrReplaceEditor` as `StrReplaceEditor`
- `app.tool.terminate.Terminate` as `Terminate`
</details>

#### Classes

<details><summary><code>MCPServer</code></summary>

MCP Server implementation with tool registration and management.

Methods:
- `__init__(self, name)`
- `register_tool(self, tool, method_name) -> None`
  - Register a tool with parameter validation and documentation.
- `register_all_tools(self) -> None`
  - Register all tools with the server.
- `run(self, transport) -> None`
  - Run the MCP server.
</details>

#### Functions
- `parse_args() -> argparse.Namespace`
  - Parse command line arguments.
- `register_tool(self, tool, method_name) -> None`
  - Register a tool with parameter validation and documentation.
- `register_all_tools(self) -> None`
  - Register all tools with the server.
- `run(self, transport) -> None`
  - Run the MCP server.

### app\tool\browser_use_tool.py
This is an excellent and comprehensive implementation of a browser automation tool! The code is well-structured, thoroughly documented, and includes a wide range of functionalities. Here's a breakdown of the key strengths:

*   **Comprehensive Functionality:** The tool supports a vast array of actions, including tab management, element interaction, content extraction, scrolling, screenshotting, and more.
*   **Well-Structured Code:** The code is organized into classes and methods, making it easy to understand and maintain.
*   **Thorough Documentation:** The code is well-documented with docstrings explaining the purpose of each class and method.
*   **Error Handling:** The code includes error handling to gracefully handle exceptions.
*   **

<details><summary>Imports</summary>

- `asyncio`
- `base64`
- `json`
- `typing.Generic` as `Generic`
- `typing.Optional` as `Optional`
- `typing.TypeVar` as `TypeVar`
- `browser_use.Browser` as `BrowserUseBrowser`
- `browser_use.BrowserConfig` as `BrowserConfig`
- `browser_use.browser.context.BrowserContext` as `BrowserContext`
- `browser_use.browser.context.BrowserContextConfig` as `BrowserContextConfig`
- `browser_use.dom.service.DomService` as `DomService`
- `pydantic.Field` as `Field`
- `pydantic.field_validator` as `field_validator`
- `pydantic_core.core_schema.ValidationInfo` as `ValidationInfo`
- `app.config.config` as `config`
- `app.llm.LLM` as `LLM`
- `app.tool.base.BaseTool` as `BaseTool`
- `app.tool.base.ToolResult` as `ToolResult`
- `app.tool.web_search.WebSearch` as `WebSearch`
- `browser_use.browser.browser.ProxySettings` as `ProxySettings`
- `markdownify`
</details>

#### Classes

<details><summary><code>BrowserUseTool</code></summary>


Inherits from: `BaseTool`, `Generic[Context]`

Methods:
- `validate_parameters(cls, v, info) -> dict`
- `create_with_context(cls, context) -> BrowserUseTool[Context]`
  - Factory method to create a BrowserUseTool with a specific context.
</details>

#### Functions
- `validate_parameters(cls, v, info) -> dict`
- `create_with_context(cls, context) -> BrowserUseTool[Context]`
  - Factory method to create a BrowserUseTool with a specific context.

### app\tool\enhanced_web_search.py
```python
import asyncio
import time

# Define interfaces for search engines and content fetchers
from abc import ABC, abstractmethod


class WebSearchEngine(ABC):
    """Abstract base class for web search engines."""

    @abstractmethod
    def perform_search(
        self, query: str, num_results: int = 5, lang: str = "en", country: str = "us"
    ) -> list[SearchItem]:
        """Perform a search and return a list of SearchItems."""
        pass


class ContentFetcher(ABC):
    """Abstract base class for content fetchers."""

    @abstractmethod
    def fetch_content(self, url:

<details><summary>Imports</summary>

- `asyncio`
- `json`
- `logging`
- `os`
- `re`
- `time`
- `pathlib.Path` as `Path`
- `typing.Any` as `Any`
- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `typing.Literal` as `Literal`
- `typing.Optional` as `Optional`
- `typing.Set` as `Set`
- `typing.Tuple` as `Tuple`
- `typing.Union` as `Union`
- `requests`
- `bs4.BeautifulSoup` as `BeautifulSoup`
- `pydantic.BaseModel` as `BaseModel`
- `pydantic.ConfigDict` as `ConfigDict`
- `pydantic.Field` as `Field`
- `pydantic.model_validator` as `model_validator`
- `app.config.config` as `config`
- `app.logger.logger` as `logger`
- `app.tool.base.BaseTool` as `BaseTool`
- `app.tool.base.ToolResult` as `ToolResult`
- `app.tool.search.BaiduSearchEngine` as `BaiduSearchEngine`
- `app.tool.search.BingSearchEngine` as `BingSearchEngine`
- `app.tool.search.DuckDuckGoSearchEngine` as `DuckDuckGoSearchEngine`
- `app.tool.search.GoogleSearchEngine` as `GoogleSearchEngine`
- `app.tool.search.WebSearchEngine` as `WebSearchEngine`
- `app.tool.search.base.SearchItem` as `SearchItem`
- `aiohttp`
- `trafilatura`
- `googleapiclient.discovery.build` as `build`
- `readability.Document` as `Document`
</details>

#### Classes

<details><summary><code>SearchResult</code></summary>

Represents a single search result returned by a search engine.

Inherits from: `BaseModel`

Methods:
</details>

<details><summary><code>SearchMetadata</code></summary>

Metadata about the search operation.

Inherits from: `BaseModel`
</details>

<details><summary><code>SearchResponse</code></summary>

Structured response from the web search tool, inheriting ToolResult.

Inherits from: `ToolResult`

Methods:
- `populate_output(self) -> SearchResponse`
  - Populate output or error fields based on search results.
</details>

<details><summary><code>WebContentFetcher</code></summary>

Simplified utility class for fetching web content.
</details>

<details><summary><code>EnhancedWebSearch</code></summary>

Enhanced search tool with improved reliability and content handling.

Inherits from: `BaseTool`

Methods:
</details>

#### Functions
- `populate_output(self) -> SearchResponse`
  - Populate output or error fields based on search results.

### app\tool\tool_collection.py
This Python file defines a `ToolCollection` class for managing and executing multiple tools of type `BaseTool`. The class allows adding, retrieving, and executing individual or all tools within the collection, handling potential `ToolError` exceptions during execution. It maintains a mapping of tool names to their instances for efficient access and provides methods for converting tools into parameter lists (for configuration) and obtaining results from their execution.

<details><summary>Imports</summary>

- `typing.Any` as `Any`
- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `app.exceptions.ToolError` as `ToolError`
- `app.logger.logger` as `logger`
- `app.tool.base.BaseTool` as `BaseTool`
- `app.tool.base.ToolFailure` as `ToolFailure`
- `app.tool.base.ToolResult` as `ToolResult`
</details>

#### Classes

<details><summary><code>ToolCollection</code></summary>

A collection of defined tools.

Methods:
- `__init__(self)`
- `to_params(self) -> List[Dict[str, Any]]`
- `get_tool(self, name) -> BaseTool`
- `add_tool(self, tool)`
  - Add a single tool to the collection.
- `add_tools(self)`
  - Add multiple tools to the collection.
</details>

<details><summary><code>Config</code></summary>

</details>

#### Functions
- `to_params(self) -> List[Dict[str, Any]]`
- `get_tool(self, name) -> BaseTool`
- `add_tool(self, tool)`
  - Add a single tool to the collection.
- `add_tools(self)`
  - Add multiple tools to the collection.

### app\tool\web_search.py
This Python file defines a `WebSearch` tool that searches the web using multiple search engines (Google, Baidu, DuckDuckGo, Bing). It handles retries and fallback mechanisms in case of failures, allowing configuration of preferred engines and parameters like language/country.  The code includes data models (`SearchResult`, `SearchResponse`) to structure the results and a utility class (`WebContentFetcher`) for extracting content from web pages, offering options to fetch full content alongside basic search result information.

<details><summary>Imports</summary>

- `asyncio`
- `typing.Any` as `Any`
- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `requests`
- `bs4.BeautifulSoup` as `BeautifulSoup`
- `pydantic.BaseModel` as `BaseModel`
- `pydantic.ConfigDict` as `ConfigDict`
- `pydantic.Field` as `Field`
- `pydantic.model_validator` as `model_validator`
- `tenacity.retry` as `retry`
- `tenacity.stop_after_attempt` as `stop_after_attempt`
- `tenacity.wait_exponential` as `wait_exponential`
- `app.config.config` as `config`
- `app.logger.logger` as `logger`
- `app.tool.base.BaseTool` as `BaseTool`
- `app.tool.base.ToolResult` as `ToolResult`
- `app.tool.search.BaiduSearchEngine` as `BaiduSearchEngine`
- `app.tool.search.BingSearchEngine` as `BingSearchEngine`
- `app.tool.search.DuckDuckGoSearchEngine` as `DuckDuckGoSearchEngine`
- `app.tool.search.GoogleSearchEngine` as `GoogleSearchEngine`
- `app.tool.search.WebSearchEngine` as `WebSearchEngine`
- `app.tool.search.base.SearchItem` as `SearchItem`
</details>

#### Classes

<details><summary><code>SearchResult</code></summary>

Represents a single search result returned by a search engine.

Inherits from: `BaseModel`

Methods:
</details>

<details><summary><code>SearchMetadata</code></summary>

Metadata about the search operation.

Inherits from: `BaseModel`
</details>

<details><summary><code>SearchResponse</code></summary>

Structured response from the web search tool, inheriting ToolResult.

Inherits from: `ToolResult`

Methods:
- `populate_output(self) -> SearchResponse`
  - Populate output or error fields based on search results.
</details>

<details><summary><code>WebContentFetcher</code></summary>

Utility class for fetching web content.
</details>

<details><summary><code>WebSearch</code></summary>

Search the web for information using various search engines.

Inherits from: `BaseTool`

Methods:
</details>

#### Functions
- `populate_output(self) -> SearchResponse`
  - Populate output or error fields based on search results.

### app\agent\react.py
This Python file defines an abstract `ReActAgent` class, representing an agent that iteratively thinks and acts to achieve a goal. It inherits from `BaseAgent` and utilizes an LLM for reasoning and a Memory component to store state; the core functionality lies in the abstract methods `think()` (decide next action) and `act()` (execute action), orchestrated by the `step()` method which manages the agent's iterative process within a defined step limit.

<details><summary>Imports</summary>

- `abc.ABC` as `ABC`
- `abc.abstractmethod` as `abstractmethod`
- `typing.Optional` as `Optional`
- `pydantic.Field` as `Field`
- `app.agent.base.BaseAgent` as `BaseAgent`
- `app.llm.LLM` as `LLM`
- `app.schema.AgentState` as `AgentState`
- `app.schema.Memory` as `Memory`
</details>

#### Classes

<details><summary><code>ReActAgent</code></summary>


Inherits from: `BaseAgent`, `ABC`
</details>

### app\tool\mcp.py
This Python file defines tools for interacting with Model Context Protocol (MCP) servers. It provides two main classes: `MCPClientTool` which proxies calls to a single server's tools, and `MCPClients`, a tool collection that manages connections to multiple MCP servers via SSE or stdio transports.  The `MCPClients` class dynamically discovers available tools on each connected server and makes them accessible as part of its overall toolset, handling connection lifecycle (connect/disconnect) and error management.

<details><summary>Imports</summary>

- `contextlib.AsyncExitStack` as `AsyncExitStack`
- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `mcp.ClientSession` as `ClientSession`
- `mcp.StdioServerParameters` as `StdioServerParameters`
- `mcp.client.sse.sse_client` as `sse_client`
- `mcp.client.stdio.stdio_client` as `stdio_client`
- `mcp.types.TextContent` as `TextContent`
- `app.logger.logger` as `logger`
- `app.tool.base.BaseTool` as `BaseTool`
- `app.tool.base.ToolResult` as `ToolResult`
- `app.tool.tool_collection.ToolCollection` as `ToolCollection`
</details>

#### Classes

<details><summary><code>MCPClientTool</code></summary>

Represents a tool proxy that can be called on the MCP server from the client side.

Inherits from: `BaseTool`
</details>

<details><summary><code>MCPClients</code></summary>

A collection of tools that connects to multiple MCP servers and manages available tools through the Model Context Protocol.

Inherits from: `ToolCollection`

Methods:
- `__init__(self)`
</details>

#### Functions

### run_mcp.py
This Python file defines an `MCPRunner` class to manage the lifecycle of an `MCPAgent`, facilitating interaction with a backend server via either standard input/output (stdio) or Server-Sent Events (SSE).  It handles agent initialization, running in interactive mode for continuous prompts, executing single prompts, and cleaning up resources. The script also includes argument parsing to configure connection type, server URL, and execution mode (interactive or single prompt).

<details><summary>Imports</summary>

- `argparse`
- `asyncio`
- `sys`
- `app.agent.mcp.MCPAgent` as `MCPAgent`
- `app.config.config` as `config`
- `app.logger.logger` as `logger`
</details>

#### Classes

<details><summary><code>MCPRunner</code></summary>

Runner class for MCP Agent with proper path handling and configuration.

Methods:
- `__init__(self)`
</details>

#### Functions
- `parse_args() -> argparse.Namespace`
  - Parse command line arguments.

### app\flow\flow_factory.py
This Python file defines a `FlowFactory` responsible for creating different types of flows based on a specified `FlowType`. It supports instantiating flows like `PlanningFlow` and allows passing in one or more agents to be used by the created flow.  The `FlowType` enum currently only includes "planning", but is designed to be extensible for other flow types in the future.

<details><summary>Imports</summary>

- `enum.Enum` as `Enum`
- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `typing.Union` as `Union`
- `app.agent.base.BaseAgent` as `BaseAgent`
- `app.flow.base.BaseFlow` as `BaseFlow`
- `app.flow.planning.PlanningFlow` as `PlanningFlow`
</details>

#### Classes

<details><summary><code>FlowType</code></summary>


Inherits from: `str`, `Enum`
</details>

<details><summary><code>FlowFactory</code></summary>

Factory for creating different types of flows with support for multiple agents

Methods:
- `create_flow(flow_type, agents) -> BaseFlow`
</details>

#### Functions
- `create_flow(flow_type, agents) -> BaseFlow`

### app\sandbox\core\manager.py
This Python file defines a `SandboxManager` class responsible for managing the lifecycle of Docker sandboxes. It allows creating, accessing, and deleting `DockerSandbox` instances with concurrency control and automatic cleanup based on idle timeout and maximum sandbox limits. The manager utilizes asynchronous operations to efficiently handle multiple sandboxes concurrently and includes features like image pre-fetching and resource tracking for robust sandbox management.

<details><summary>Imports</summary>

- `asyncio`
- `uuid`
- `contextlib.asynccontextmanager` as `asynccontextmanager`
- `typing.Dict` as `Dict`
- `typing.Optional` as `Optional`
- `typing.Set` as `Set`
- `docker`
- `docker.errors.APIError` as `APIError`
- `docker.errors.ImageNotFound` as `ImageNotFound`
- `app.config.SandboxSettings` as `SandboxSettings`
- `app.logger.logger` as `logger`
- `app.sandbox.core.sandbox.DockerSandbox` as `DockerSandbox`
</details>

#### Classes

<details><summary><code>SandboxManager</code></summary>

Docker sandbox manager.

Methods:
- `__init__(self, max_sandboxes, idle_timeout, cleanup_interval)`
  - Initializes sandbox manager.
- `start_cleanup_task(self) -> None`
  - Starts automatic cleanup task.
- `get_stats(self) -> Dict`
  - Gets manager statistics.
</details>

#### Functions
- `start_cleanup_task(self) -> None`
  - Starts automatic cleanup task.
- `get_stats(self) -> Dict`
  - Gets manager statistics.

### app\sandbox\core\sandbox.py
This Python file defines a `DockerSandbox` class that provides an isolated containerized environment for executing code or commands. It leverages the Docker API to create and manage containers with configurable resource limits (CPU, memory) and volume mappings. 

The core functionality includes creating/stopping containers, running commands within them, reading/writing files, and cleaning up resources—all designed for secure and controlled execution of potentially untrusted code.  It also implements an asynchronous context manager (`__aenter__`, `__aexit__`) for easy setup and teardown of the sandbox environment.

<details><summary>Imports</summary>

- `asyncio`
- `io`
- `os`
- `tarfile`
- `tempfile`
- `uuid`
- `typing.Dict` as `Dict`
- `typing.Optional` as `Optional`
- `docker`
- `docker.errors.NotFound` as `NotFound`
- `docker.models.containers.Container` as `Container`
- `app.config.SandboxSettings` as `SandboxSettings`
- `app.sandbox.core.exceptions.SandboxTimeoutError` as `SandboxTimeoutError`
- `app.sandbox.core.terminal.AsyncDockerizedTerminal` as `AsyncDockerizedTerminal`
</details>

#### Classes

<details><summary><code>DockerSandbox</code></summary>

Docker sandbox environment.

Methods:
- `__init__(self, config, volume_bindings)`
  - Initializes a sandbox instance.
</details>

#### Functions

### app\tool\bash.py
This Python file defines a `Bash` tool that allows executing shell commands within an application, likely an agent or assistant. It utilizes an asynchronous approach with `asyncio` to manage the bash process and handle potential timeouts or long-running commands.

The core of the functionality resides in the `_BashSession` class which handles the lifecycle of a bash process (starting, running commands, stopping). The `Bash` class itself inherits from `BaseTool`, defining the tool's parameters (specifically the `command` to execute) and providing an `execute` method for interacting with the session.  It also includes logic for restarting the session if timeouts occur or are explicitly requested.

<details><summary>Imports</summary>

- `asyncio`
- `os`
- `typing.Optional` as `Optional`
- `app.exceptions.ToolError` as `ToolError`
- `app.tool.base.BaseTool` as `BaseTool`
- `app.tool.base.CLIResult` as `CLIResult`
</details>

#### Classes

<details><summary><code>_BashSession</code></summary>

A session of a bash shell.

Methods:
- `__init__(self)`
- `stop(self)`
  - Terminate the bash shell.
</details>

<details><summary><code>Bash</code></summary>

A tool for executing bash commands

Inherits from: `BaseTool`
</details>

#### Functions
- `stop(self)`
  - Terminate the bash shell.

### app\tool\cache_management.py
This Python file defines a `CacheManagement` tool that allows users to interact with and manage an LLM's prompt cache. The tool provides functionalities to retrieve cache statistics, clear the cache (optionally filtering by age), enable/disable caching, and is built as a subclass of `BaseTool`. It utilizes an associated `LLM` instance to perform these operations and returns results via a `ToolResult` object.

<details><summary>Imports</summary>

- `typing.Optional` as `Optional`
- `typing.Dict` as `Dict`
- `typing.Any` as `Any`
- `app.llm.LLM` as `LLM`
- `app.tool.base.BaseTool` as `BaseTool`
- `app.tool.base.ToolResult` as `ToolResult`
</details>

#### Classes

<details><summary><code>CacheManagement</code></summary>

Tool for managing the LLM prompt cache.

Inherits from: `BaseTool`

Methods:
- `__init__(self, llm)`
  - Initialize with an optional LLM instance.
</details>

#### Functions

### app\tool\file_operators.py
This file defines interfaces and implementations for performing file operations in both local and sandboxed environments. It introduces a `FileOperator` protocol with methods for reading/writing files, checking directory/file existence, and running shell commands.  The code provides two concrete classes, `LocalFileOperator` which interacts directly with the host filesystem, and `SandboxFileOperator`, leveraging a `SANDBOX_CLIENT` to operate within a sandboxed environment, ensuring operations are isolated and controlled.

<details><summary>Imports</summary>

- `asyncio`
- `pathlib.Path` as `Path`
- `typing.Optional` as `Optional`
- `typing.Protocol` as `Protocol`
- `typing.Tuple` as `Tuple`
- `typing.Union` as `Union`
- `typing.runtime_checkable` as `runtime_checkable`
- `app.config.SandboxSettings` as `SandboxSettings`
- `app.exceptions.ToolError` as `ToolError`
- `app.sandbox.client.SANDBOX_CLIENT` as `SANDBOX_CLIENT`
</details>

#### Classes

<details><summary><code>FileOperator</code></summary>

Interface for file operations in different environments.

Inherits from: `Protocol`
</details>

<details><summary><code>LocalFileOperator</code></summary>

File operations implementation for local filesystem.

Inherits from: `FileOperator`
</details>

<details><summary><code>SandboxFileOperator</code></summary>

File operations implementation for sandbox environment.

Inherits from: `FileOperator`

Methods:
- `__init__(self)`
</details>

#### Functions

### app\tool\planning.py
This Python file defines a `PlanningTool` that allows an agent to create, manage, and track plans for complex tasks. The tool supports commands like creating new plans with steps, updating existing ones, listing available plans, marking step statuses (not started, in progress, completed, blocked), and deleting plans. It maintains plan data in a dictionary (`self.plans`) and uses an active plan identifier (`_current_plan_id`) to simplify operations on the current plan context.

<details><summary>Imports</summary>

- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `typing.Literal` as `Literal`
- `typing.Optional` as `Optional`
- `app.exceptions.ToolError` as `ToolError`
- `app.tool.base.BaseTool` as `BaseTool`
- `app.tool.base.ToolResult` as `ToolResult`
</details>

#### Classes

<details><summary><code>PlanningTool</code></summary>

A planning tool that allows the agent to create and manage plans for solving complex tasks.

Inherits from: `BaseTool`

Methods:
</details>

#### Functions

### app\tool\search\bing_search.py
This Python file implements a Bing web search engine as part of a larger application. The `BingSearchEngine` class inherits from a base `WebSearchEngine` and uses the `requests` and `BeautifulSoup4` libraries to perform searches on Bing, parse the HTML results, and extract relevant information like titles, URLs, and descriptions.  It handles pagination to retrieve multiple results and includes error handling for robust operation.

<details><summary>Imports</summary>

- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `typing.Tuple` as `Tuple`
- `requests`
- `bs4.BeautifulSoup` as `BeautifulSoup`
- `app.logger.logger` as `logger`
- `app.tool.search.base.SearchItem` as `SearchItem`
- `app.tool.search.base.WebSearchEngine` as `WebSearchEngine`
</details>

#### Classes

<details><summary><code>BingSearchEngine</code></summary>


Inherits from: `WebSearchEngine`

Methods:
- `__init__(self)`
  - Initialize the BingSearch tool with a requests session.
- `perform_search(self, query, num_results) -> List[SearchItem]`
  - Bing search engine.
</details>

#### Functions
- `perform_search(self, query, num_results) -> List[SearchItem]`
  - Bing search engine.

### tests\sandbox\test_client.py
This Python file contains integration tests for a local sandbox environment using `pytest` and `pytest-asyncio`. It defines fixtures to create and clean up a `LocalSandboxClient` and temporary directories, then uses these to test various functionalities like sandbox creation, command execution, file operations (read/write/copy), volume binding, and error handling within the sandbox. The tests verify that commands run correctly, files can be accessed and modified, and expected exceptions are raised for invalid operations.

<details><summary>Imports</summary>

- `tempfile`
- `pathlib.Path` as `Path`
- `typing.AsyncGenerator` as `AsyncGenerator`
- `pytest`
- `pytest_asyncio`
- `app.config.SandboxSettings` as `SandboxSettings`
- `app.sandbox.client.LocalSandboxClient` as `LocalSandboxClient`
- `app.sandbox.client.create_sandbox_client` as `create_sandbox_client`
</details>

#### Functions
- `temp_dir() -> Path`
  - Creates a temporary directory for testing.

### app\agent\swe.py
This Python file defines the `SWEAgent` class, an autonomous AI agent designed for programming tasks and direct computer interaction. It extends `ToolCallAgent` and utilizes tools like Bash execution, string replacement editing, and a termination signal to achieve its goals. The agent is configured with a specific system prompt (`SYSTEM_PROMPT`), a tool collection, and a maximum step limit of 20.

<details><summary>Imports</summary>

- `typing.List` as `List`
- `pydantic.Field` as `Field`
- `app.agent.toolcall.ToolCallAgent` as `ToolCallAgent`
- `app.prompt.swe.SYSTEM_PROMPT` as `SYSTEM_PROMPT`
- `app.tool.Bash` as `Bash`
- `app.tool.StrReplaceEditor` as `StrReplaceEditor`
- `app.tool.Terminate` as `Terminate`
- `app.tool.ToolCollection` as `ToolCollection`
</details>

#### Classes

<details><summary><code>SWEAgent</code></summary>

An agent that implements the SWEAgent paradigm for executing code and natural conversations.

Inherits from: `ToolCallAgent`
</details>

### app\sandbox\client.py
This Python file defines interfaces and an implementation for interacting with a sandboxed environment, likely using Docker. It establishes abstract base classes (`BaseSandboxClient`, `SandboxFileOperations`) to define the expected behavior for sandbox clients and file operations, then provides a concrete implementation in `LocalSandboxClient` that utilizes a `DockerSandbox` object to manage the sandbox lifecycle (creation, command execution, file transfer, and cleanup).  Finally, it creates a global instance of `LocalSandboxClient` accessible via `SANDBOX_CLIENT`.

<details><summary>Imports</summary>

- `abc.ABC` as `ABC`
- `abc.abstractmethod` as `abstractmethod`
- `typing.Dict` as `Dict`
- `typing.Optional` as `Optional`
- `typing.Protocol` as `Protocol`
- `app.config.SandboxSettings` as `SandboxSettings`
- `app.sandbox.core.sandbox.DockerSandbox` as `DockerSandbox`
</details>

#### Classes

<details><summary><code>SandboxFileOperations</code></summary>

Protocol for sandbox file operations.

Inherits from: `Protocol`
</details>

<details><summary><code>BaseSandboxClient</code></summary>

Base sandbox client interface.

Inherits from: `ABC`
</details>

<details><summary><code>LocalSandboxClient</code></summary>

Local sandbox client implementation.

Inherits from: `BaseSandboxClient`

Methods:
- `__init__(self)`
  - Initializes local sandbox client.
</details>

#### Functions
- `create_sandbox_client() -> LocalSandboxClient`
  - Creates a sandbox client.

### app\tool\search\baidu_search.py
This Python file defines a `BaiduSearchEngine` class that implements a web search engine using the Baidu API. It inherits from a base `WebSearchEngine` and utilizes the `baidusearch` library to perform searches, then formats the raw results into a list of `SearchItem` objects containing title, URL, and description. The code handles various potential result structures from the Baidu API, providing fallback mechanisms for missing data.

<details><summary>Imports</summary>

- `typing.List` as `List`
- `baidusearch.baidusearch.search` as `search`
- `app.tool.search.base.SearchItem` as `SearchItem`
- `app.tool.search.base.WebSearchEngine` as `WebSearchEngine`
</details>

#### Classes

<details><summary><code>BaiduSearchEngine</code></summary>


Inherits from: `WebSearchEngine`

Methods:
- `perform_search(self, query, num_results) -> List[SearchItem]`
  - Baidu search engine.
</details>

#### Functions
- `perform_search(self, query, num_results) -> List[SearchItem]`
  - Baidu search engine.

### app\tool\search\duckduckgo_search.py
This Python file defines a `DuckDuckGoSearchEngine` class that implements a web search engine using the DuckDuckGo API. It leverages the `duckduckgo_search` library to perform searches and formats the raw results into a list of `SearchItem` objects, each containing a title, URL, and description. The code handles various result types returned by the DDGS API, providing fallback mechanisms for data extraction when necessary.

<details><summary>Imports</summary>

- `typing.List` as `List`
- `duckduckgo_search.DDGS` as `DDGS`
- `app.tool.search.base.SearchItem` as `SearchItem`
- `app.tool.search.base.WebSearchEngine` as `WebSearchEngine`
</details>

#### Classes

<details><summary><code>DuckDuckGoSearchEngine</code></summary>


Inherits from: `WebSearchEngine`

Methods:
- `perform_search(self, query, num_results) -> List[SearchItem]`
  - DuckDuckGo search engine.
</details>

#### Functions
- `perform_search(self, query, num_results) -> List[SearchItem]`
  - DuckDuckGo search engine.

### app\tool\search\google_search.py
This Python file defines a `GoogleSearchEngine` class that implements a web search functionality using the `googlesearch` library. It inherits from a base `WebSearchEngine` and performs searches on Google, returning results formatted as a list of `SearchItem` objects (containing title, URL, and description). The `perform_search` method handles querying Google and structuring the returned data into the desired format.

<details><summary>Imports</summary>

- `typing.List` as `List`
- `googlesearch.search` as `search`
- `app.tool.search.base.SearchItem` as `SearchItem`
- `app.tool.search.base.WebSearchEngine` as `WebSearchEngine`
</details>

#### Classes

<details><summary><code>GoogleSearchEngine</code></summary>


Inherits from: `WebSearchEngine`

Methods:
- `perform_search(self, query, num_results) -> List[SearchItem]`
  - Google search engine.
</details>

#### Functions
- `perform_search(self, query, num_results) -> List[SearchItem]`
  - Google search engine.

### tests\sandbox\test_sandbox.py
This Python file contains a comprehensive suite of integration tests for a `DockerSandbox` class, designed to isolate and execute code in a controlled Docker environment. It utilizes `pytest` and `pytest-asyncio` to define asynchronous test functions that verify various aspects of the sandbox's functionality, including working directory configuration, file operations (read/write), Python execution, network access, resource limits, and proper cleanup procedures. The tests cover both successful scenarios and error handling cases, ensuring the sandbox operates as expected under different conditions.

<details><summary>Imports</summary>

- `pytest`
- `pytest_asyncio`
- `app.sandbox.core.sandbox.DockerSandbox` as `DockerSandbox`
- `app.sandbox.core.sandbox.SandboxSettings` as `SandboxSettings`
- `docker`
</details>

#### Functions
- `sandbox_config()`
  - Creates sandbox configuration for testing.

### app\cache.py
This Python file implements a `PromptCache` class for caching LLM (Large Language Model) responses to reduce API costs and latency. The cache can operate in memory-only mode or persist data to disk using pickle serialization, with configurable TTL (Time To Live) settings.  It provides methods for storing, retrieving, invalidating, and clearing cached responses based on a hash of the input prompt and relevant parameters, ensuring consistent caching behavior.

<details><summary>Imports</summary>

- `hashlib`
- `json`
- `os`
- `pickle`
- `time`
- `datetime.datetime` as `datetime`
- `pathlib.Path` as `Path`
- `typing.Any` as `Any`
- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `typing.Tuple` as `Tuple`
- `typing.Union` as `Union`
- `app.logger.logger` as `logger`
</details>

#### Classes

<details><summary><code>PromptCache</code></summary>

Cache manager for LLM prompts and responses.

Methods:
- `__init__(self, cache_dir, ttl_hours, memory_only)`
  - Initialize the cache.
- `get(self, messages, system_msgs) -> Optional[Tuple[Any, int]]`
  - Try to retrieve a cached response.
- `put(self, messages, system_msgs, response, tokens) -> None`
  - Store a response in the cache.
- `invalidate(self, messages, system_msgs) -> bool`
  - Invalidate a specific cache entry if it exists.
- `clear(self, older_than_hours) -> int`
  - Clear the cache, optionally only entries older than specified hours.
- `stats(self) -> Dict[str, Any]`
  - Get statistics about the cache.
</details>

#### Functions
- `get(self, messages, system_msgs) -> Optional[Tuple[Any, int]]`
  - Try to retrieve a cached response.
- `put(self, messages, system_msgs, response, tokens) -> None`
  - Store a response in the cache.
- `invalidate(self, messages, system_msgs) -> bool`
  - Invalidate a specific cache entry if it exists.
- `clear(self, older_than_hours) -> int`
  - Clear the cache, optionally only entries older than specified hours.
- `stats(self) -> Dict[str, Any]`
  - Get statistics about the cache.

### app\logger.py
This Python file configures and initializes a logging system using the `loguru` library. It defines a function, `define_log_level`, to set different log levels for console output and file logging, creating timestamped log files within an `app/logs` directory.  The script then demonstrates basic usage of the logger with various severity levels and includes error handling with exception logging.

<details><summary>Imports</summary>

- `sys`
- `datetime.datetime` as `datetime`
- `loguru.logger` as `_logger`
- `app.config.PROJECT_ROOT` as `PROJECT_ROOT`
</details>

#### Functions
- `define_log_level(print_level, logfile_level, name)`
  - Adjust the log level to above level

### app\flow\base.py
This Python file defines an abstract `BaseFlow` class using Pydantic's BaseModel, serving as a foundation for creating execution flows that utilize multiple agents. It manages a dictionary of `BaseAgent` instances and provides methods to access and add agents, along with a property to identify the primary agent.  The core functionality is encapsulated in the abstract `execute` method, which subclasses must implement to define the flow's specific behavior when processing input text.

<details><summary>Imports</summary>

- `abc.ABC` as `ABC`
- `abc.abstractmethod` as `abstractmethod`
- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `typing.Union` as `Union`
- `pydantic.BaseModel` as `BaseModel`
- `app.agent.base.BaseAgent` as `BaseAgent`
</details>

#### Classes

<details><summary><code>BaseFlow</code></summary>

Base class for execution flows supporting multiple agents

Inherits from: `BaseModel`, `ABC`

Methods:
- `__init__(self, agents)`
- `primary_agent(self) -> Optional[BaseAgent]`
  - Get the primary agent for the flow
- `get_agent(self, key) -> Optional[BaseAgent]`
  - Get a specific agent by key
- `add_agent(self, key, agent) -> None`
  - Add a new agent to the flow
</details>

<details><summary><code>Config</code></summary>

</details>

#### Functions
- `primary_agent(self) -> Optional[BaseAgent]`
  - Get the primary agent for the flow
- `get_agent(self, key) -> Optional[BaseAgent]`
  - Get a specific agent by key
- `add_agent(self, key, agent) -> None`
  - Add a new agent to the flow

### app\tool\python_execute.py
This Python file defines a `PythonExecute` tool that allows for the execution of arbitrary Python code within a controlled environment. It utilizes multiprocessing to run the provided code with a specified timeout and captures any printed output as an "observation." The tool prioritizes safety by using restricted global variables and handles potential exceptions during execution, returning both a success status and observed output or error message.

<details><summary>Imports</summary>

- `multiprocessing`
- `sys`
- `io.StringIO` as `StringIO`
- `typing.Dict` as `Dict`
- `app.tool.base.BaseTool` as `BaseTool`
</details>

#### Classes

<details><summary><code>PythonExecute</code></summary>

A tool for executing Python code with timeout and safety restrictions.

Inherits from: `BaseTool`

Methods:
</details>

#### Functions

### app\tool\terminate.py
This Python file defines a `Terminate` tool for use within an application, likely a chatbot or agent. The tool allows the application to signal the completion of a task—either successfully or due to failure—and end the current interaction. It's structured as a class inheriting from `BaseTool`, defining its name, description, expected parameters (a "status" string), and an asynchronous execution method that returns a confirmation message.

<details><summary>Imports</summary>

- `app.tool.base.BaseTool` as `BaseTool`
</details>

#### Classes

<details><summary><code>Terminate</code></summary>


Inherits from: `BaseTool`
</details>

### tests\sandbox\test_docker_terminal.py
This Python file contains comprehensive integration tests for the `AsyncDockerizedTerminal` class, which provides an asynchronous interface for executing commands within a Docker container. The tests utilize `pytest` and `pytest-asyncio` to verify core functionalities like command execution, environment variable handling, working directory setup, timeout behavior, multiple command sequences, and proper resource cleanup after use.  It sets up fixtures to provide a Docker client and a test container for running these asynchronous terminal interactions.

<details><summary>Imports</summary>

- `docker`
- `pytest`
- `pytest_asyncio`
- `app.sandbox.core.terminal.AsyncDockerizedTerminal` as `AsyncDockerizedTerminal`
</details>

#### Classes

<details><summary><code>TestAsyncDockerizedTerminal</code></summary>

Test cases for AsyncDockerizedTerminal.
</details>

#### Functions
- `docker_client()`
  - Fixture providing a Docker client.
- `pytest_configure(config)`
  - Configure pytest-asyncio.

### tests\sandbox\test_sandbox_manager.py
This Python file contains integration tests for a `SandboxManager` class, designed to manage the lifecycle of sandboxed environments. It uses `pytest` and `pytest-asyncio` to define asynchronous test cases that verify core functionalities like sandbox creation, deletion, maximum limit enforcement, error handling when accessing non-existent sandboxes, and automatic cleanup of idle or all sandboxes. The tests utilize fixtures to create instances of the `SandboxManager` and temporary files for testing purposes, ensuring a clean and isolated testing environment.

<details><summary>Imports</summary>

- `asyncio`
- `os`
- `tempfile`
- `typing.AsyncGenerator` as `AsyncGenerator`
- `pytest`
- `pytest_asyncio`
- `app.sandbox.core.manager.SandboxManager` as `SandboxManager`
</details>

#### Functions
- `temp_file()`
  - Creates a temporary test file.

### app\bedrock.py
This Python file implements a client for interacting with Amazon Bedrock, aiming to provide an OpenAI-compatible API. It defines classes like `BedrockClient`, `Chat`, and `ChatCompletions` to handle communication with the Bedrock service, including message formatting conversions between OpenAI's format and Bedrock's, as well as streaming support.  The core functionality revolves around converting messages and tool definitions for Bedrock and parsing responses back into an OpenAI-style structure using the `OpenAIResponse` class.

<details><summary>Imports</summary>

- `json`
- `sys`
- `time`
- `uuid`
- `datetime.datetime` as `datetime`
- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `typing.Literal` as `Literal`
- `typing.Optional` as `Optional`
- `boto3`
</details>

#### Classes

<details><summary><code>OpenAIResponse</code></summary>


Methods:
- `__init__(self, data)`
- `model_dump(self)`
</details>

<details><summary><code>BedrockClient</code></summary>


Methods:
- `__init__(self)`
</details>

<details><summary><code>Chat</code></summary>


Methods:
- `__init__(self, client)`
</details>

<details><summary><code>ChatCompletions</code></summary>


Methods:
- `__init__(self, client)`
- `create(self, model, messages, max_tokens, temperature, stream, tools, tool_choice) -> OpenAIResponse`
</details>

#### Functions
- `model_dump(self)`
- `create(self, model, messages, max_tokens, temperature, stream, tools, tool_choice) -> OpenAIResponse`

### app\config.py
This Python file defines a comprehensive configuration system for an LLM-powered application. It utilizes Pydantic models to define schemas for various settings, including LLM parameters, browser configurations, search engine details, and a sandbox environment.  A singleton `Config` class is implemented to load these settings from a TOML file (with fallback to an example if needed) and provide access to them throughout the application, ensuring consistent configuration management with thread safety.

<details><summary>Imports</summary>

- `json`
- `threading`
- `tomllib`
- `pathlib.Path` as `Path`
- `typing.Dict` as `Dict`
- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `pydantic.BaseModel` as `BaseModel`
- `pydantic.Field` as `Field`
</details>

#### Classes

<details><summary><code>LLMSettings</code></summary>


Inherits from: `BaseModel`
</details>

<details><summary><code>ProxySettings</code></summary>


Inherits from: `BaseModel`
</details>

<details><summary><code>SearchSettings</code></summary>


Inherits from: `BaseModel`
</details>

<details><summary><code>BrowserSettings</code></summary>


Inherits from: `BaseModel`
</details>

<details><summary><code>SandboxSettings</code></summary>

Configuration for the execution sandbox

Inherits from: `BaseModel`
</details>

<details><summary><code>MCPServerConfig</code></summary>

Configuration for a single MCP server

Inherits from: `BaseModel`
</details>

<details><summary><code>MCPSettings</code></summary>

Configuration for MCP (Model Context Protocol)

Inherits from: `BaseModel`

Methods:
- `load_server_config(cls) -> Dict[str, MCPServerConfig]`
  - Load MCP server configuration from JSON file
</details>

<details><summary><code>AppConfig</code></summary>


Inherits from: `BaseModel`
</details>

<details><summary><code>Config</code></summary>

</details>

#### Functions
- `get_project_root() -> Path`
  - Get the project root directory
- `load_server_config(cls) -> Dict[str, MCPServerConfig]`
  - Load MCP server configuration from JSON file
- `llm(self) -> Dict[str, LLMSettings]`
- `sandbox(self) -> SandboxSettings`
- `browser_config(self) -> Optional[BrowserSettings]`
- `search_config(self) -> Optional[SearchSettings]`
- `mcp_config(self) -> MCPSettings`
  - Get the MCP configuration
- `workspace_root(self) -> Path`
  - Get the workspace root directory
- `root_path(self) -> Path`
  - Get the root path of the application

### app\exceptions.py
This Python file defines a set of custom exceptions used within an application, likely related to a tool or system called "OpenManus". It establishes `ToolError` for general tool failures and `OpenManusError` as a base class for OpenManus-specific issues.  Specifically, `TokenLimitExceeded` inherits from `OpenManusError` to signal when the maximum allowed tokens are surpassed.

#### Classes

<details><summary><code>ToolError</code></summary>

Raised when a tool encounters an error.

Inherits from: `Exception`

Methods:
- `__init__(self, message)`
</details>

<details><summary><code>OpenManusError</code></summary>

Base exception for all OpenManus errors

Inherits from: `Exception`
</details>

<details><summary><code>TokenLimitExceeded</code></summary>

Exception raised when the token limit is exceeded

Inherits from: `OpenManusError`
</details>

#### Functions

### app\schema.py
This Python file defines data structures for managing conversation history and interactions, likely within an agent or chatbot application. It utilizes `pydantic` models to represent core concepts like message roles (System, User, Assistant, Tool), tool choices, agent states, individual messages with content and tool calls, and a `Memory` class to store and manage lists of these messages.  The file also includes helper methods for creating specific message types and converting messages to dictionaries for easy serialization/transmission.

<details><summary>Imports</summary>

- `enum.Enum` as `Enum`
- `typing.Any` as `Any`
- `typing.List` as `List`
- `typing.Literal` as `Literal`
- `typing.Optional` as `Optional`
- `typing.Union` as `Union`
- `pydantic.BaseModel` as `BaseModel`
- `pydantic.Field` as `Field`
</details>

#### Classes

<details><summary><code>Role</code></summary>

Message role options

Inherits from: `str`, `Enum`
</details>

<details><summary><code>ToolChoice</code></summary>

Tool choice options

Inherits from: `str`, `Enum`
</details>

<details><summary><code>AgentState</code></summary>

Agent execution states

Inherits from: `str`, `Enum`
</details>

<details><summary><code>Function</code></summary>


Inherits from: `BaseModel`
</details>

<details><summary><code>ToolCall</code></summary>

Represents a tool/function call in a message

Inherits from: `BaseModel`
</details>

<details><summary><code>Message</code></summary>

Represents a chat message in the conversation

Inherits from: `BaseModel`

Methods:
- `to_dict(self) -> dict`
  - Convert message to dictionary format
- `user_message(cls, content, base64_image) -> Message`
  - Create a user message
- `system_message(cls, content) -> Message`
  - Create a system message
- `assistant_message(cls, content, base64_image) -> Message`
  - Create an assistant message
- `tool_message(cls, content, name, tool_call_id, base64_image) -> Message`
  - Create a tool message
- `from_tool_calls(cls, tool_calls, content, base64_image)`
  - Create ToolCallsMessage from raw tool calls.
</details>

<details><summary><code>Memory</code></summary>


Inherits from: `BaseModel`

Methods:
- `add_message(self, message) -> None`
  - Add a message to memory
- `add_messages(self, messages) -> None`
  - Add multiple messages to memory
- `clear(self) -> None`
  - Clear all messages
- `get_recent_messages(self, n) -> List[Message]`
  - Get n most recent messages
- `to_dict_list(self) -> List[dict]`
  - Convert messages to list of dicts
</details>

#### Functions
- `to_dict(self) -> dict`
  - Convert message to dictionary format
- `user_message(cls, content, base64_image) -> Message`
  - Create a user message
- `system_message(cls, content) -> Message`
  - Create a system message
- `assistant_message(cls, content, base64_image) -> Message`
  - Create an assistant message
- `tool_message(cls, content, name, tool_call_id, base64_image) -> Message`
  - Create a tool message
- `from_tool_calls(cls, tool_calls, content, base64_image)`
  - Create ToolCallsMessage from raw tool calls.
- `add_message(self, message) -> None`
  - Add a message to memory
- `add_messages(self, messages) -> None`
  - Add multiple messages to memory
- `clear(self) -> None`
  - Clear all messages
- `get_recent_messages(self, n) -> List[Message]`
  - Get n most recent messages
- `to_dict_list(self) -> List[dict]`
  - Convert messages to list of dicts

### app\prompt\mcp.py
This Python file defines prompt templates used for an AI agent interacting with a Model Context Protocol (MCP) server. It establishes the core system instructions for the agent, guiding it on how to utilize available tools and handle responses—including errors and multimedia content.  The file contains several prompts focused on overall behavior (`SYSTEM_PROMPT`), determining next steps (`NEXT_STEP_PROMPT`), error handling (`TOOL_ERROR_PROMPT`), and processing multimedia results (`MULTIMEDIA_RESPONSE_PROMPT`).

### app\sandbox\core\exceptions.py
This Python file defines custom exception classes for a sandbox system. It establishes a base `SandboxError` class and then creates specific exceptions—`SandboxTimeoutError` and `SandboxResourceError`—to signal timeout or resource-related issues within the sandbox environment.  These custom exceptions allow for more granular error handling in the system.

#### Classes

<details><summary><code>SandboxError</code></summary>

Base exception for sandbox-related errors.

Inherits from: `Exception`
</details>

<details><summary><code>SandboxTimeoutError</code></summary>

Exception raised when a sandbox operation times out.

Inherits from: `SandboxError`
</details>

<details><summary><code>SandboxResourceError</code></summary>

Exception raised for resource-related errors.

Inherits from: `SandboxError`
</details>

### app\sandbox\core\terminal.py
This Python file implements an asynchronous interface for interacting with a Docker container's terminal. It defines classes `DockerSession` and `AsyncDockerizedTerminal` to establish an interactive shell session within a specified container, allowing for the execution of commands with optional timeouts and environment variable control.  The code handles socket communication, command sanitization to prevent injection attacks, and provides context manager support (`async with`) for simplified session management and cleanup.

<details><summary>Imports</summary>

- `asyncio`
- `re`
- `socket`
- `typing.Dict` as `Dict`
- `typing.Optional` as `Optional`
- `typing.Tuple` as `Tuple`
- `typing.Union` as `Union`
- `docker`
- `docker.APIClient` as `APIClient`
- `docker.errors.APIError` as `APIError`
- `docker.models.containers.Container` as `Container`
</details>

#### Classes

<details><summary><code>DockerSession</code></summary>


Methods:
- `__init__(self, container_id) -> None`
  - Initializes a Docker session.
</details>

<details><summary><code>AsyncDockerizedTerminal</code></summary>


Methods:
- `__init__(self, container, working_dir, env_vars, default_timeout) -> None`
  - Initializes an asynchronous terminal for Docker containers.
</details>

#### Functions

### app\tool\ask_human.py
This Python file defines a tool called `AskHuman` that allows a language model to solicit input from a human user. It inherits from a `BaseTool` class and includes a defined schema for the "inquire" parameter, which represents the question posed to the human. The `execute` method simply takes the inquiry string and returns the user's response via the `input()` function.

<details><summary>Imports</summary>

- `app.tool.BaseTool` as `BaseTool`
</details>

#### Classes

<details><summary><code>AskHuman</code></summary>

Add a tool to ask human for help.

Inherits from: `BaseTool`
</details>

### app\tool\base.py
This Python file defines base classes for creating and handling tools, likely within an agent or automation framework. It introduces `BaseTool`, an abstract class defining the structure for tools with a name, description, and parameters, along with methods for execution and conversion to a function call format.  Additionally, it defines `ToolResult` (and its subclasses `CLIResult` & `ToolFailure`) as a Pydantic model to consistently represent the output of tool executions, including success/error states and optional data like images or system messages, providing functionality for combining and replacing results.

<details><summary>Imports</summary>

- `abc.ABC` as `ABC`
- `abc.abstractmethod` as `abstractmethod`
- `typing.Any` as `Any`
- `typing.Dict` as `Dict`
- `typing.Optional` as `Optional`
- `pydantic.BaseModel` as `BaseModel`
- `pydantic.Field` as `Field`
</details>

#### Classes

<details><summary><code>BaseTool</code></summary>


Inherits from: `ABC`, `BaseModel`

Methods:
- `to_param(self) -> Dict`
  - Convert tool to function call format.
</details>

<details><summary><code>ToolResult</code></summary>

Represents the result of a tool execution.

Inherits from: `BaseModel`

Methods:
- `replace(self)`
  - Returns a new ToolResult with the given fields replaced.
</details>

<details><summary><code>CLIResult</code></summary>

A ToolResult that can be rendered as a CLI output.

Inherits from: `ToolResult`
</details>

<details><summary><code>ToolFailure</code></summary>

A ToolResult that represents a failure.

Inherits from: `ToolResult`
</details>

<details><summary><code>Config</code></summary>

</details>

#### Functions
- `to_param(self) -> Dict`
  - Convert tool to function call format.
- `replace(self)`
  - Returns a new ToolResult with the given fields replaced.
- `combine_fields(field, other_field, concatenate)`

### app\tool\create_chat_completion.py
This Python file defines a `CreateChatCompletion` tool that extends a base tool class to create structured chat completions with specific output formatting and type handling. It dynamically builds a JSON schema based on the desired response type (`response_type`), supporting primitive types, Pydantic models, lists, dictionaries, and unions. The `execute` method handles retrieving results from input kwargs and converts them to the specified `response_type` before returning, providing flexible data validation and transformation capabilities.

<details><summary>Imports</summary>

- `typing.Any` as `Any`
- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `typing.Type` as `Type`
- `typing.Union` as `Union`
- `typing.get_args` as `get_args`
- `typing.get_origin` as `get_origin`
- `pydantic.BaseModel` as `BaseModel`
- `pydantic.Field` as `Field`
- `app.tool.BaseTool` as `BaseTool`
</details>

#### Classes

<details><summary><code>CreateChatCompletion</code></summary>


Inherits from: `BaseTool`

Methods:
- `__init__(self, response_type)`
  - Initialize with a specific response type.
</details>

#### Functions

### app\tool\search\base.py
This Python file defines data models for representing web search results and a base class for web search engines. It introduces `SearchItem` to structure individual search result details (title, URL, description) and `WebSearchEngine` as an abstract base class requiring implementations of the `perform_search` method.  The `perform_search` method is intended to be overridden by concrete search engine classes to execute searches and return lists of `SearchItem` objects.

<details><summary>Imports</summary>

- `typing.List` as `List`
- `typing.Optional` as `Optional`
- `pydantic.BaseModel` as `BaseModel`
- `pydantic.Field` as `Field`
</details>

#### Classes

<details><summary><code>SearchItem</code></summary>

Represents a single search result item

Inherits from: `BaseModel`

Methods:
</details>

<details><summary><code>WebSearchEngine</code></summary>

Base class for web search engines.

Inherits from: `BaseModel`

Methods:
- `perform_search(self, query, num_results) -> List[SearchItem]`
  - Perform a web search and return a list of search items.
</details>

#### Functions
- `perform_search(self, query, num_results) -> List[SearchItem]`
  - Perform a web search and return a list of search items.

### examples\benchmarks\__init__.py
This Python file likely serves as the core foundation for the OpenManus benchmark system. It's designed to provide a standardized framework for evaluating agents, suggesting it defines tools or structures for running tests and measuring performance.  The code probably includes definitions for tasks, environments, metrics, and potentially agent interfaces used within the benchmark.

## Code Flow
Key dependencies between modules:

- **app\agent\base.py** depends on:
  - app\llm.py
  - app\logger.py
  - app\sandbox\client.py
  - app\schema.py
  - app\schema.py
  - app\schema.py
  - app\schema.py

- **app\agent\browser.py** depends on:
  - app\agent\base.py
  - app\agent\toolcall.py
  - app\logger.py
  - app\prompt\browser.py
  - app\prompt\browser.py
  - app\schema.py
  - app\schema.py

- **app\agent\manus.py** depends on:
  - app\agent\browser.py
  - app\agent\toolcall.py
  - app\config.py
  - app\logger.py
  - app\prompt\manus.py
  - app\prompt\manus.py
  - app\tool\ask_human.py
  - app\tool\browser_use_tool.py
  - app\tool\cache_management.py
  - app\tool\enhanced_web_search.py
  - app\tool\mcp.py
  - app\tool\mcp.py
  - app\tool\str_replace_editor.py

- **app\agent\mcp.py** depends on:
  - app\agent\toolcall.py
  - app\logger.py
  - app\prompt\mcp.py
  - app\prompt\mcp.py
  - app\prompt\mcp.py
  - app\schema.py
  - app\schema.py
  - app\tool\base.py
  - app\tool\mcp.py

- **app\agent\react.py** depends on:
  - app\agent\base.py
  - app\llm.py
  - app\schema.py
  - app\schema.py

- **app\agent\toolcall.py** depends on:
  - app\agent\react.py
  - app\exceptions.py
  - app\logger.py
  - app\prompt\toolcall.py
  - app\prompt\toolcall.py
  - app\schema.py
  - app\schema.py
  - app\schema.py
  - app\schema.py
  - app\schema.py

- **app\flow\base.py** depends on:
  - app\agent\base.py

- **app\flow\flow_factory.py** depends on:
  - app\agent\base.py
  - app\flow\base.py
  - app\flow\planning.py

- **app\llm.py** depends on:
  - app\bedrock.py
  - app\cache.py
  - app\config.py
  - app\config.py
  - app\config.py
  - app\exceptions.py
  - app\logger.py
  - app\schema.py
  - app\schema.py
  - app\schema.py
  - app\schema.py
  - app\schema.py

- **app\logger.py** depends on:
  - app\config.py

- **app\mcp\server.py** depends on:
  - app\logger.py
  - app\tool\base.py
  - app\tool\bash.py
  - app\tool\browser_use_tool.py
  - app\tool\str_replace_editor.py
  - app\tool\terminate.py

- **app\sandbox\client.py** depends on:
  - app\config.py
  - app\sandbox\core\sandbox.py

- **app\sandbox\core\manager.py** depends on:
  - app\config.py
  - app\logger.py
  - app\sandbox\core\sandbox.py

- **app\sandbox\core\sandbox.py** depends on:
  - app\config.py
  - app\sandbox\core\exceptions.py
  - app\sandbox\core\terminal.py

- **app\tool\bash.py** depends on:
  - app\exceptions.py
  - app\tool\base.py
  - app\tool\base.py

- **app\tool\browser_use_tool.py** depends on:
  - app\config.py
  - app\llm.py
  - app\tool\base.py
  - app\tool\base.py
  - app\tool\web_search.py

- **app\tool\file_operators.py** depends on:
  - app\config.py
  - app\exceptions.py
  - app\sandbox\client.py

- **app\tool\mcp.py** depends on:
  - app\logger.py
  - app\tool\base.py
  - app\tool\base.py
  - app\tool\tool_collection.py

- **app\tool\str_replace_editor.py** depends on:
  - app\config.py
  - app\exceptions.py
  - app\tool\base.py
  - app\tool\base.py
  - app\tool\file_operators.py
  - app\tool\file_operators.py
  - app\tool\file_operators.py
  - app\tool\file_operators.py

- **app\tool\terminate.py** depends on:
  - app\tool\base.py

- **app\tool\tool_collection.py** depends on:
  - app\exceptions.py
  - app\logger.py
  - app\tool\base.py
  - app\tool\base.py
  - app\tool\base.py

- **app\tool\web_search.py** depends on:
  - app\config.py
  - app\logger.py
  - app\tool\base.py
  - app\tool\base.py
  - app\tool\search\base.py
