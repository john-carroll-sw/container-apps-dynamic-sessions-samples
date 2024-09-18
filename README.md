# Extended Version: Azure Container Apps dynamic sessions samples

This project is based on the original repository: [Azure-Samples/container-apps-dynamic-sessions-samples](https://github.com/Azure-Samples/container-apps-dynamic-sessions-samples/tree/main).

See tutorials:

* [LangChain](https://learn.microsoft.com/azure/container-apps/sessions-tutorial-langchain)
* [LlamaIndex](https://learn.microsoft.com/azure/container-apps/sessions-tutorial-llamaindex)
* [Semantic Kernel](https://learn.microsoft.com/azure/container-apps/sessions-tutorial-semantic-kernel)
* [AutoGen](https://learn.microsoft.com/azure/container-apps/sessions-tutorial-autogen)


## Overview

This project provides a single and multi-agent interaction system running code in sandboxed environments within Azure Container Apps (ACA) dynamic sessions. This fork has a few added examples in the Semantic Kernel and Autogen folders.

## Features

- Sandboxed code execution
- Integration with Azure Container Apps
- Agentic interactions examples(Semantic Kernel, Autogen)
  - Semantic Kernel
    - Try out [fastapi_multi_agent.py](./semantic-kernel-python-webapi/fastapi_multi_agent.py)! - 3 agents (code writer, reviewer, executor)
      - Run `fastapi dev fastapi_multi_agent.py`, go to http://127.0.0.1:8000/docs#/default/chat_chat_get, enter an example that would need code execution such as, i.e: 'What is today's date and time? Show in Eastern time.'
    - [main.py](./semantic-kernel-python-webapi/main.py) - basic, single agent code execution example
  - AutoGen
    - [main.py](./autogen-python-webapi/main.py) - basic, two agent code execution example (code writer, code executor)


## Setup

1. Clone the repository:

   ```sh
   git clone https://github.com/john-carroll-sw/container-apps-dynamic-sessions-samples.git
   cd container-apps-dynamic-sessions-samples
   ```

2. Install dependencies in each framework folder. Semantic Kernel:

   ```sh
   pip install -r requirements.txt
   ```

   For Semantic Kernel directly install:
   ```sh
   pip install fastapi==0.111.0 azure-identity==1.16.0 python-dotenv==1.0.1 semantic-kernel==1.5.1
   ```

3. Run the application:

   ```sh
   cd <directory of framework>
   fastapi dev <filename (i.e main.py, fastapi_multi_agent.py)>
   ```

### Usage

Example usage:

1. To use the chat functionality, send a GET request to /chat with the message parameter. Example from curl: 

    ```sh
    curl "http://127.0.0.1:8000/chat?message=Hello"

2. Navigate to API and use the UI

    ```sh
    http://127.0.0.1:8000/

## Contributing

Contributions to this project are welcome. Please follow the standard fork-and-pull request workflow. If you plan to introduce a major change, it's best to open an issue first to discuss it.

## License

This project is licensed under the [MIT License](LICENSE).