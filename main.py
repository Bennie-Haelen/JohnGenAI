import json
import tiktoken  
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.schema import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from modules.llm_utils import get_llm
from prompts.read_prompt_template import read_prompt_template
from prompts import prompt_names
from logger_setup import logger, log_entry_exit

LLM_MODEL = "gpt-4o"
NAME      = "John"

def main():

    # Initialize the Language Model (LLM)
    llm = get_llm(LLM_MODEL)

    prompt_name = prompt_names.TEST_PROMPT
    prompt_template_str = read_prompt_template(prompt_name)
    logger.info(f"Using prompt template: {prompt_name}")

    # Set up the prompt template with the expected input variable
    prompt_template = PromptTemplate(
        input_variables=["name"], template=prompt_template_str)

    # Format the prompt by injecting the FHIR resource name
    prompt = prompt_template.format(name=NAME)
    logger.info(f"Prompt after formatting: {prompt}")

    # Create a message array containing the formatted prompt
    messages = [HumanMessage(content=prompt)]

    # Invoke the LLM model to generate a description
    result_ouput = llm.invoke(input=messages).content

    logger.info(f"Result output from the LLM: {result_ouput}")

# -------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    main()
# -------------------------------------------------------------------------