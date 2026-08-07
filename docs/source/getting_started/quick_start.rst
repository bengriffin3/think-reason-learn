Quick Start
-----------

GPTree
~~~~~~

.. code-block:: python

   # Setup imports

   from IPython.display import Image
   import pandas as pd
   import numpy as np
   from think_reason_learn.gptree import GPTree
   from think_reason_learn.core.llms import GoogleChoice, OpenAIChoice
   from think_reason_learn.core.llms import XAIChoice, AnthropicChoice

   import asyncio

   # Sample data: Predict startup founder success

   X = pd.DataFrame(
       {
           "founder_info": [
               "Alex is a serial entrepreneur with two successful exits, strong network in Silicon Valley, and expertise in AI.",
               "Jordan graduated top of class from Oxford but has no prior business experience and limited funding.",
               "Taylor has 10 years in finance, secured seed funding quickly, and built a talented team.",
               "Casey started a company right out of high school, faced multiple failures, but persists with innovative ideas.",
               "Morgan is a former Google engineer with patents in machine learning and venture capital backing."
               "Seraphine graduated from UMaT with a first class degree in Minerals Engineering and has a strong network in the mining industry."
           ]
       }
   )
   y = np.array(["successful", "failed", "successful", "failed", "successful", "successful"])

   # Initialize GPTree with LLM choices
   tree = GPTree(
       qgen_llmc=[
           GoogleChoice(model="gemini-1.5-flash-latest"),
           OpenAIChoice(model="gpt-5"),
           XAIChoice(model="grok-3-mini"),
       ],
       critic_llmc=[
           XAIChoice(model="grok-3-mini"),
           OpenAIChoice(model="gpt-4o-mini"),
           AnthropicChoice(model="claude-3-5-sonnet-20240620"),
       ],
       qgen_instr_llmc=[
           GoogleChoice(model="gemini-1.5-flash-latest"),
           XAIChoice(model="grok-3-mini"),
       ],
   )

   # Generate question generation instructions template
   qgit = await tree.set_tasks(
       task_description="Predict if a startup founder will be successful or fail based on their background.",
   )

   # Print the generated instructions template
   print(qgit)

   # Fit the tree
   fitter = tree.fit(X, y, reset=True)

   # Build the root node / Could do it in a loop or manually
   root = await anext(fitter)

   # Visualize the tree
   display(Image(tree.view_tree()))

   # Get training data with generated features
   tree.get_training_data()

   # Get all generated questions
   tree.get_questions()

   # Predict on the training data
   # NOTE: ensure the fitter is fully consumed
   predictions = await tree.predict(X)
   for pred in predictions:
       print(pred)

Using a local model (Ollama, vLLM, or any OpenAI-compatible server)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``openai`` provider works against any server that implements the OpenAI
chat-completions API. Point it at your local endpoint and pick your local
model — no code changes needed:

.. code-block:: bash

   export OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama's default port
   export OPENAI_API_KEY=ollama                       # any non-empty value works locally

.. code-block:: python

   from think_reason_learn.core.llms import OpenAIChoice

   qgen_llmc = [OpenAIChoice(model="qwen3:8b")]  # any model served locally

When ``OPENAI_BASE_URL`` is set, requests are routed through
``/v1/chat/completions`` with token logprobs enabled (Reasoned Rule Mining
depends on ``response.logprobs``) instead of the ``/v1/responses`` endpoint,
which local servers generally do not implement. The routing is controlled by
the ``endpoint_style`` parameter of the OpenAI provider
(``"responses"`` or ``"chat_completions"``) and auto-detected from the base
URL when unset. Override it with the ``OPENAI_ENDPOINT_STYLE`` environment
variable — for example, ``OPENAI_ENDPOINT_STYLE=chat_completions`` also gets
token logprobs from OpenAI's hosted API.

For more examples and comprehensive use cases, visit the `examples directory <https://github.com/vela-research/think-reason-learn/tree/main/examples>`_ on our GitHub repository.
