import os
import dotenv
import datetime
import logging

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent, FunctionCallContent, FunctionResultContent, AuthorRole
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.core_plugins.sessions_python_tool.sessions_python_plugin import SessionsPythonTool
from semantic_kernel.exceptions.function_exceptions import FunctionExecutionException
from semantic_kernel.functions.kernel_arguments import KernelArguments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

app = FastAPI()

pool_management_endpoint = os.getenv("POOL_MANAGEMENT_ENDPOINT")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

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


@app.get("/")
async def root():
    return RedirectResponse("/docs")

@app.get("/chat")
async def chat(message: str):
    logger.info(f"Received chat message: {message}")
    kernel = Kernel()

    service_id = "sessions-tool"
    chat_service = AzureChatCompletion(
        service_id=service_id,
        ad_token_provider=auth_callback_factory("https://cognitiveservices.azure.com/.default"),
        endpoint=azure_openai_endpoint,
        deployment_name=azure_deployment_name,
    )
    kernel.add_service(chat_service)
    logger.info("Added AzureChatCompletion service to kernel")

    sessions_tool = SessionsPythonTool(
        auth_callback=auth_callback_factory("https://dynamicsessions.io/.default"),
        pool_management_endpoint=pool_management_endpoint,
    )
    kernel.add_plugin(sessions_tool, "SessionsTool")
    logger.info("Added SessionsPythonTool plugin to kernel")

    # Define functions for each agent
    code_writer_function = kernel.add_function(
        prompt="""
                    You are a CodeWriter agent. 
                    You use your coding skill to solve problems. 
                    You output only valid python code. 
                    This valid code will be executed in a sandbox, resulting in result, stdout, or stderr. 
                    All necessary libraries have already been installed.
                    You are entering a work session with other agents: CodeExecutor.
                    Do NOT execute code. Only return the code you write for it to be executed by the CodeExecutor agent.
                    
                    Write Python code for the following task: {{$user_input}}

                    Chat History: {{$chat_history}}
                """,
        plugin_name="CodeWriter",
        function_name="WriteCode",
    )
    logger.info("Added code writer function to kernel")

    code_executor_function = kernel.add_function(
        prompt="""
                    Execute the following Python code: {{$code}} and return the output.
                """,
        plugin_name="CodeExecutor",
        function_name="ExecuteCode",
    )
    logger.info("Added code executor function to kernel")

    req_settings = AzureChatPromptExecutionSettings(
        service_id=service_id, 
        temperature=0.0,
        tool_choice="auto"
    )
    logger.info("Created AzureChatPromptExecutionSettings")

    filter = {"excluded_plugins": ["ChatBot", "CodeWriter", "CodeExecutor"]}
    req_settings.function_call_behavior = FunctionCallBehavior.EnableFunctions(auto_invoke=True, filters=filter)
    logger.info("Configured function call behavior")

    arguments = KernelArguments(settings=req_settings)
    logger.info("Created KernelArguments with request settings")

    history = ChatHistory()
    logger.info("Initialized ChatHistory")

    # Main logic to invoke agents
    try:
        # Invoke code writer agent
        logger.info("Invoking code writer agent")
        arguments["chat_history"] = history
        arguments["user_input"] = message
        code = await kernel.invoke(
            function=code_writer_function,
            arguments=arguments,
        )
        logger.info(f"Code writer output: {code}")

        # Invoke code executor agent
        logger.info("Invoking code executor agent")
        arguments["code"] = code
        execution_result = await kernel.invoke(
            function=code_executor_function,
            arguments=arguments,
        )
        logger.info(f"Code executor output: {execution_result}")

        response = {
            "code": str(code),
            "execution_result": str(execution_result),
        }
        logger.info(f"Returning response: {response}")

    except Exception as e:
        logger.error(f"Kernel invocation failed: {e}")
        return {"error": str(e)}

    return response