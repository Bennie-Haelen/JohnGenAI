from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from logger_setup import logger, log_entry_exit


# Function to initialize the LangChain LLM (Language Learning Model)
@log_entry_exit  # Decorator for logging function entry and exit
def get_llm(model_name):
    """
    Initializes and returns an appropriate LangChain LLM model based on the provided model name.
    
    Parameters:
    - model_name (str): The name of the LLM model to be used.
    
    Returns:
    - An instance of either ChatGoogleGenerativeAI or ChatOpenAI, depending on the model name.
    
    Behavior:
    - If the model name contains "gemini" (case-insensitive), it initializes a Google Gemini model.
    - Otherwise, it defaults to an OpenAI Chat model.
    - Both models are initialized with:
      - `temperature=0.0` (ensuring deterministic output)
      - `max_tokens=15000` (limiting response length)
    """

    # Check if the model name contains "gemini" (case insensitive)
    if "gemini" in model_name.lower():
        
        # Initialize Google Gemini LLM
        logger.info(f"Using Google Generative AI model: {model_name}")
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0) 

        if llm is None:
            raise ValueError(f"Google Generative AI model '{model_name}' not found.")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    else:
    
        # Default to OpenAI Chat model
        logger.info(f"Using OpenAI Chat model: {model_name}")
        return ChatOpenAI(model=model_name, temperature=0.0)