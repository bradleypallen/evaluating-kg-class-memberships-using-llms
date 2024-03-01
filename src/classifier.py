import os
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint

class Classifier:
    """Represents a zero-shot chain-of-thought classification procedure."""

    RATIONALE_GENERATION_PROMPT = """Concept: {concept} 
Definition: {definition}
Entity: {entity} 
Description: {description}
  
Using the above definition, and only the information in the above definition, 
provide an argument for the assertion that {entity} is a(n) {concept}. 
Let's reason step by step.
    
Rationale:
"""

    ANSWER_GENERATION_PROMPT = """Concept: {concept} 
Definition: {definition}
Entity: {entity} 
Description: {description}
Rationale: {rationale}

Now using the argument provided in the above rationale, answer the question: is {entity} a(n) {concept}? 
Answer 'positive' or 'negative', and only the one word 'positive' or 'negative'. Use lower case letters in your answer. 
If there is not enough information to be sure of an answer, answer 'negative'.
  
Answer:
"""
    
    def __init__(self, id, term, definition, reference=None, model_name="gpt-4-0125-preview", temperature=0.1):
        """
        Initializes a classification procedure for a concept, given a unique identifier, a term, and a definition.
        
        Parameters:
            id: The identifier for the concept.
            term: The term or name of the concept.
            definition: The definition of the concept.
            reference: A URL containing the source of the definition.
            model_name: The name of the model to be used for zero shot CoT classification (default "gpt-4").
            temperature: The temperature parameter for the model (default 0.1).
         """
        self.id = id
        self.term = term
        self.definition = definition
        self.reference = reference
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._llm(model_name, temperature)
        self.chain = self._chain()

    def _llm(self, model_name, temperature):
        if model_name in [
            "gpt-3.5-turbo",
            "gpt-4-1106-preview",
            "gpt-4-0125-preview"
            ]:
            return ChatOpenAI(model_name=model_name, temperature=temperature)
        elif model_name in [
            "meta-llama/Llama-2-70b-chat-hf", 
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "google/gemma-7b-it",
            "google/gemma-2b-it"
            ]:
            return HuggingFaceEndpoint(
                repo_id=model_name, 
                temperature=temperature, 
                huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
            )
        else:
            raise Exception(f'Model {model_name} not supported')

    def _chain(self):
        """
        Creates a langchain.SequentialChain that implements a zero-shot
        chain of thought (CoT) using a specification. 
        """
        rationale_generation = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["concept", "definition", "entity", "description"], 
                template=self.RATIONALE_GENERATION_PROMPT
            ), 
            output_key="rationale"
        )
        answer_generation = LLMChain(
            llm=self.llm, 
            prompt=PromptTemplate(
                input_variables=["concept", "definition", "entity", "description", "rationale"], 
                template=self.ANSWER_GENERATION_PROMPT
            ), 
            output_key="answer"
        )
        return SequentialChain(
            chains=[rationale_generation, answer_generation],
            input_variables=["concept", "definition", "entity", "description"],
            output_variables=["rationale", "answer"]
        )
    
    def classify(self, name, description):
        """
        Determines whether or not an entity is in the extension of the classification procedure's concept.
        
        Parameters:
            name: The name of the entity to be classified.
            description: The description of the entity to be classified.
        
        Returns:
            A JSON object containing a classification of the entity based on the concept's definition.
        """
        return self.chain.invoke(
            {
                "concept": self.term, 
                "definition": self.definition, 
                "entity": name,
                "description": description
            }
        )