import asyncio
import datetime
import dotenv
import os

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential
from functools import reduce
from semantic_kernel.agents.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.core_plugins.sessions_python_tool.sessions_python_plugin import SessionsPythonTool
from semantic_kernel.exceptions.function_exceptions import FunctionExecutionException
from semantic_kernel.kernel import Kernel


dotenv.load_dotenv()

# Env Config
streaming = True
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
base_url = os.environ.get("AZURE_OPENAI_BASE_URL")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
pool_management_endpoint = os.getenv("POOL_MANAGEMENT_ENDPOINT")


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


async def invoke_agent(agent: ChatCompletionAgent, input: str, chat: ChatHistory):
    """Invoke the agent with the user input."""
    chat.add_user_message(input)

    print(f"# {AuthorRole.USER}: '{input}'")

    if streaming:
        contents = []
        content_name = ""
        async for content in agent.invoke_stream(chat):
            content_name = content.name
            contents.append(content)
        streaming_chat_message = reduce(lambda first, second: first + second, contents)
        print(f"# {content.role} - {content_name or '*'}: '{streaming_chat_message}'")
        chat.add_message(content)
    else:
        async for content in agent.invoke(chat):
            print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
            chat.add_message(content)


async def main():
    # Create the instance of the Kernel
    kernel = Kernel()

    # Add the OpenAIChatCompletion AI Service to the Kernel
    kernel.add_service(AzureChatCompletion(
            service_id="agent", 
            api_key=api_key,
            deployment_name=deployment_name,
            base_url=base_url,
            api_version=api_version,
        )
    )

    # Add the chat function to the Kernel
    kernel.add_function(
        prompt="{{$chat_history}}{{$user_input}}",
        execution_settings=OpenAIChatPromptExecutionSettings(
            service_id="agent",
            temperature=0.0,
            max_tokens=1000,
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
        ),
        plugin_name="ChatBot",
        function_name="Chat",
        description="Chat with the assistant",
    )

    # Add the code interpreter sessions pool to the Kernel
    kernel.add_plugin(
        plugin_name="SessionsTool",
        plugin=SessionsPythonTool(
            auth_callback=auth_callback_factory("https://dynamicsessions.io/.default"),
            pool_management_endpoint=pool_management_endpoint,
        )
    )

    # Create the agent
    agent = ChatCompletionAgent(
        service_id="agent", 
        kernel=kernel, 
        name="Agent",
        instructions="""
            You are a helpful AI assistant. 
            You use your coding skill to solve problems. 
            You have access to an IPython kernel to execute Python code. 
            You output only valid python code. 
            This valid code will be executed in a sandbox, resulting in result, stdout, or stderr. 
            All necessary libraries have already been installed. 
            Execute the code and return the result to the user.
        """)

    # Define the chat history
    chat = ChatHistory()
    system_message = """
        You are a helpful AI assistant. 
        You use your coding skill to solve problems. 
        You have access to an IPython kernel to execute Python code. 
        You output only valid python code. 
        This valid code will be executed in a sandbox, resulting in result, stdout, or stderr. 
        All necessary libraries have already been installed. 
        Execute the code and return the result to the user.
    """
    
    chat.add_system_message(system_message)

    # Respond to user input
    await invoke_agent(agent, "What time is it?", chat)

if __name__ == "__main__":
    asyncio.run(main())
