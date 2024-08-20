import datetime
import os
import logging

import dotenv
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import \
    AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import \
    AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.core_plugins.sessions_python_tool.sessions_python_plugin import \
    SessionsPythonTool
from semantic_kernel.exceptions.function_exceptions import \
    FunctionExecutionException
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
    kernel = Kernel()

    service_id = "sessions-tool"
    chat_service = AzureChatCompletion(
        service_id=service_id,
        ad_token_provider=auth_callback_factory("https://cognitiveservices.azure.com/.default"),
        endpoint=azure_openai_endpoint,
        deployment_name=azure_deployment_name,
    )
    kernel.add_service(chat_service)

    sessions_tool = SessionsPythonTool(
        auth_callback=auth_callback_factory("https://dynamicsessions.io/.default"),
        pool_management_endpoint=pool_management_endpoint,
    )
    kernel.add_plugin(sessions_tool, "SessionsTool")

    chat_function = kernel.add_function(
        prompt="{{$chat_history}}{{$user_input}}",
        plugin_name="ChatBot",
        function_name="Chat",
    )

    req_settings = AzureChatPromptExecutionSettings(service_id=service_id, tool_choice="auto")

    req_settings.function_choice_behavior = FunctionChoiceBehavior.Auto(filters={"excluded_plugins": ["ChatBot"]})

    arguments = KernelArguments(settings=req_settings)

    history = ChatHistory()

    system_message = """
    You are a helpful AI assistant. 
    You use your coding skill to solve problems. 
    You have access to an IPython kernel to execute Python code. 
    You output only valid python code. 
    This valid code will be executed in a sandbox, resulting in result, stdout, or stderr. 
    All necessary libraries have already been installed. 
    Execute the code and return the result to the user.
"""
    
    history.add_system_message(system_message)

    arguments["chat_history"] = history
    arguments["user_input"] = message
    answer = await kernel.invoke(
        function=chat_function,
        arguments=arguments,
    )

    response = {
        "output": str(answer),
    }

    return response
