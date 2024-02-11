In this library, we create the llm and chat model, for infering using the together api.
In the api key things to note:

=> The language model needs prompt formatting to be done prior to sending the prompt. Please style your prompt in the requires template.

=> The chat model accepts the messages list, as traditional {'role', 'content'} objects.


Here, we do not use the inbuilt together llm in langchain, but build the code for the inference by extending the LLM and BaseChatModel classes. This is because the langchain_together library offers no native support for chat models, and also uses the deprecated inference end point as opposed to the newer completions and chat_completions endpoint.