import os
import dotenv
import datetime
import logging

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from functools import reduce
from semantic_kernel import Kernel
from semantic_kernel.agents.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.core_plugins.sessions_python_tool.sessions_python_plugin import SessionsPythonTool
from semantic_kernel.exceptions.function_exceptions import FunctionExecutionException

app = FastAPI()

# Config
dotenv.load_dotenv()
streaming = False # To toggle streaming or non-streaming mode, change the following boolean
pool_management_endpoint = os.getenv("POOL_MANAGEMENT_ENDPOINT")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_message(message):
    COLORS = {
        'MESSAGE': '\033[95m',
        'ENDC': '\033[0m'          # Reset
    }
    print(f"{COLORS['MESSAGE']}{message}{COLORS['ENDC']}")


def log_flow(from_agent, to_agent):
    COLORS = {
        'FROM_AGENT': '\033[94m',  # Blue
        'TO_AGENT': '\033[92m',    # Green
        'ENDC': '\033[0m'          # Reset
    }
    print(f"{COLORS['FROM_AGENT']}{from_agent.capitalize()}{COLORS['ENDC']} (to {COLORS['TO_AGENT']}{to_agent.capitalize() or '*'}{COLORS['ENDC']}): \n")


def log_separator():
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    print(f"{YELLOW}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{ENDC}\n")


def auth_callback_factory(scope):
    auth_token = None
    async def auth_callback() -> str:
        """Auth callback for the SessionsPythonTool.
        This is a sample auth callback that shows how to use Azure's DefaultAzureCredential
        to get an access token.
        """
        nonlocal auth_token
        current_utc_timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())

        if not auth_token or auth_token.expires_on < current_utc_timestamp:
            credential = DefaultAzureCredential()

            try:
                auth_token = credential.get_token(scope)
            except ClientAuthenticationError as cae:
                err_messages = getattr(cae, "messages", [])
                raise FunctionExecutionException(
                    f"Failed to retrieve the client auth token with messages: {' '.join(err_messages)}"
                ) from cae

        return auth_token.token
    
    return auth_callback


async def invoke_agent(agent: ChatCompletionAgent, to_agent: str, input: str, history: ChatHistory):
    """Invoke the agent with the user input."""
    history.add_user_message(input)

    if streaming:
        contents = []
        content_name = ""
        async for content in agent.invoke_stream(history):
            content_name = content.name
            contents.append(content)
        streaming_chat_message = reduce(lambda first, second: first + second, contents)
        # print(f"# {content.role} - {content_name.capitalize() or '*'}: \n'{streaming_chat_message}'\n")
        log_flow(content_name, to_agent)
        print(f"\033[94m{streaming_chat_message}'\n")
        history.add_message(content)
    else:
        async for content in agent.invoke(history):
            # print(f"\033[94m# {content.name.capitalize() or '*'}:\033[0m \n'{content.content}'\n")
            log_flow(content.name, to_agent)
            print(f"\033[94m{content.content}'\n")
            history.add_message(content)

    if history.messages:
        last_message = history.messages[-1]
    return last_message


@app.get("/")
async def root():
    return RedirectResponse("/docs")


@app.get("/chat")
async def chat(message: str):
    # Instantiate the Kernel
    kernel = Kernel()

    # Add AzureChatCompletion services for each agent.
    services = ["code_writer", "code_reviewer", "code_executor"]
    for service_id in services:
        kernel.add_service(AzureChatCompletion(
            service_id=service_id,
            ad_token_provider=auth_callback_factory("https://cognitiveservices.azure.com/.default"),
            endpoint=azure_openai_endpoint,
            deployment_name=azure_deployment_name,
        ))

    # Add the code interpreter sessions pool to the Kernel
    kernel.add_plugin(
        plugin_name="CodeInterpreterSessionsTool",
        plugin=SessionsPythonTool(
            auth_callback=auth_callback_factory("https://dynamicsessions.io/.default"),
            pool_management_endpoint=pool_management_endpoint,
        )
    )

    # Create multiple agents with specific sets of concerns and instructions
    code_writer = ChatCompletionAgent(
        kernel=kernel, 
        service_id="code_writer", 
        name="code_writer", 
        instructions="""
            You are a CodeWriter agent. 
            You use your coding skill to solve problems. 
            You output only valid python code. 
            This valid code will be executed in a sandbox, resulting in result, stdout, or stderr. 
            All necessary libraries have already been installed.
            You are entering a work session with other agents: CodeReviewer, CodeExecutor.
            Do NOT execute code. Only return the code you write for it to be reviewed by the CodeReviewer agent.
        """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id="code_writer",
            temperature=0.0,
            max_tokens=1000,
            function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
        ),
    )

    code_reviewer = ChatCompletionAgent(
        kernel=kernel, 
        service_id="code_reviewer", 
        name="code_reviewer", 
        instructions="""
            You are a CodeReviewer agent.
            You use your code reviewing skills to provide worded feedback to improve code.
            Be as critical as possible in your review.
            Review the Python code given to you. 
            You are entering a work session with other agents: CodeWriter, CodeExecutor.
            Do NOT write code, do not return any code blocks of any sort, do not provide an improved or revised version.
            Do NOT execute code. Any code you approve will be executed by the CodeExecutor agent.
            1. Return 'approved' if it looks good.
            2. Return expert feedback using words to improve the code, 
            start off your response to the CodeWriter with: 'Revise the code based on the following review:', 
            and state at the end 'This review is for the CodeWriter to use'.
            Ensure the code is as simple as possible as the code just needs to be able to run. No human is ever going to read this code.
        """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id="code_reviewer",
            temperature=0.0,
            max_tokens=1000,
            function_choice_behavior=FunctionChoiceBehavior.NoneInvoke(),
        ),
    )

    code_executor = ChatCompletionAgent(
        kernel=kernel, 
        service_id="code_executor", 
        name="code_executor", 
        instructions="""
            You are a CodeExecutor agent.
            You have access to an IPython kernel to execute Python code. 
            Your output should be the result from executing valid python code. 
            This valid code will be executed in a sandbox, resulting in result, stdout, or stderr. 
            All necessary libraries have already been installed.
            You are entering a work session with other agents: CodeWriter, CodeReviewer.
            Execute the code given to you, using the output, return a chat response to the user.
            Ensure the response to the user is readable to a human and there is not any code.
            If you do not call a function, do not hallucinate the response of a code execution, 
            instead if you cannot run code simply say you cannot run code.
        """,
        execution_settings=AzureChatPromptExecutionSettings(
            service_id="code_executor",
            temperature=0.0,
            max_tokens=1000,
            # function_choice_behavior=FunctionChoiceBehavior.Auto(),
            function_choice_behavior=FunctionChoiceBehavior.Auto(filters={"included_plugins": ["CodeInterpreterSessionsTool"]}),
        ),
    )

    chat_history = ChatHistory()

    # Main logical flow to invoke agents for this multi-agent code execution demo
    try:
        max_iterations = 10  # Set the maximum number of iterations
        iteration_count = 0

        log_separator()
        log_message("Received chat message")
        log_flow("User", "")
        print(f"{message}\n")

        while iteration_count < max_iterations:
            # Invoke code writer agent
            log_separator()
            log_message("Invoking code writer agent")
            code = await invoke_agent(code_writer, code_reviewer.name, message, chat_history)

            # Invoke code reviewer agent
            log_separator()
            log_message("Invoking code reviewer agent")
            review = await invoke_agent(code_reviewer, "", str(code), chat_history)

            # Check if the review is satisfactory
            if "approved" in review.content.lower():
                logger.info("Code review approved")
                break
            else:
                logger.info("Code review not approved, sending back to code writer")
                message = review.content
            iteration_count += 1

        if iteration_count == max_iterations:
            logger.error("Maximum iterations reached without approval")
            return {"error": "Maximum iterations reached without approval"}

        # Invoke code executor agent
        log_separator()
        log_message("Invoking code executor agent")
        execution_result = await invoke_agent(code_executor, "User", str(code), chat_history)

        response = {
            "code": str(code),
            "final_review": str(review),
            "execution_result": str(execution_result),
        }

        log_separator()
        logger.info(f"Returning response: {str(response)}")

    except Exception as e:
        logger.error(f"Kernel invocation failed: {e}")
        return {"error": str(e)}

    return response
