
from dotenv import load_dotenv
from pyprojroot import here
import os
from uuid import uuid4
from langsmith import Client
from langchain.docstore.document import Document

import csv
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_xml_agent
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_anthropic import ChatAnthropic


from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.utils.function_calling import convert_to_openai_function

import dill as pickle

## INICIAR LANGSMITH Y API KEYS
dotenv_path = here() / ".env"
load_dotenv(dotenv_path=dotenv_path)


client = Client()

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = f"API - {unique_id}"


# LEVANTAR DATOS

crear_dataset = True
with open('metodos_obj_str.pkl', 'rb') as archivo:
    metodos_lista = pickle.load(archivo)


if crear_dataset:
    db = Chroma(persist_directory="db_RAG", embedding_function=OpenAIEmbeddings())
    db.delete_collection()
    docs = []

    for metod in metodos_lista:
        docs.append(Document(page_content=metod.descripcion, metadata={"nombre":metod.nombre,"sistema":metod.sistema}))

    embedding_function = OpenAIEmbeddings()
    db_ret = Chroma.from_documents(docs, embedding_function, persist_directory="db_RAG")
else:
    db_ret = Chroma(persist_directory="db_RAG", embedding_function=OpenAIEmbeddings())

# DEFINIR TOOLS

class method_decription(BaseModel):
    method_description: str = Field(description="Description of what the method should do")

class method_name(BaseModel):
    method: str = Field(description="Method to search information of")
    sistem: str = Field(description="Sistem to which the method belongs to", default="all")

@tool("get_methods_from_description", args_schema=method_decription)
def get_method(method_description:str):
    """ Returns a list of possible methods provided a description. """
    filter={}
    ret_metods = db_ret.similarity_search(method_description,k=5,filter=filter)
    ret_metods_names = [met.metadata["nombre"] for met in ret_metods]
    ret_metods_sistems = [met.metadata["sistema"] for met in ret_metods]
    ret_metodos_obj = []

    for met in metodos_lista:
        if met.nombre in ret_metods_names:
            indices = [indice for indice, elemento in enumerate(ret_metods_names) if elemento == met.nombre]
            if met.sistema in [ret_metods_sistems[i] for i in indices]:
                ret_metodos_obj.append(met)

    resp = ""
    for metod in ret_metodos_obj:
        resp +=  method_info_as_string(metod) + "\n"

    return resp

@tool("get_method_info_from_name", args_schema=method_name)
def get_method_info(method:str, sistem: str="all"):
    """ Returns information (input-ouput squeema, possible errors and calling example using Soap or JSON) of a method.  """

    # Search for the method exactly
    possible_methods = []
    for met in metodos_lista:
        if met.nombre.lower().strip() == method.lower().strip():
            possible_methods.append(met)
    
    if len(possible_methods) > 1 and sistem != "all":
        ret_metodos_obj = []
        for met in possible_methods:
            if met.sistema.lower().strip() == sistem.lower().strip():
                ret_metodos_obj.append(met)
    else:
        ret_metodos_obj = possible_methods

    # In case exact match doesnt work
    if len(ret_metodos_obj) == 0:
        ret_metods = db_ret.similarity_search(method,k=2)
        ret_metods_names = [met.metadata["nombre"] for met in ret_metods]
        ret_metods_sistems = [met.metadata["sistema"] for met in ret_metods]
        for met in metodos_lista:
            if met.nombre in ret_metods_names:
                indices = [indice for indice, elemento in enumerate(ret_metods_names) if elemento == met.nombre]
                if met.sistema in [ret_metods_sistems[i] for i in indices]:
                    ret_metodos_obj.append(met)

    resp = ""
    for metod in ret_metodos_obj:
        resp +=  method_info_as_string(metod,params_info=True) + "\n"
    return resp


def method_info_as_string(metod,params_info=False):
    strng  = ""
    strng += f"Metodo: {metod.nombre} \n Sistema: {metod.sistema} \n Descripción: {metod.descripcion} \n "
    if params_info:
        strng  += f"{metod.entrada} \n {metod.salida} \n {metod.error} \n"
        strng  += f"{metod.ej_in} \n {metod.ej_out}"
    return strng

tools = [get_method, get_method_info]


# DEFINIR AGENTE

openai = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.0,streaming=True)
antrhopic = ChatAnthropic(temperature=0, model_name="claude-3-haiku-20240307")
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", """Task: You are a helpful assistant. You must answer users question IN SPANISH. To obtain the information needed, use the tools provided or rely on the chat history. Under no circunstances invent information. Be detailed but stay relevant to the question on your answers, following the instructions the user demands."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
chat_anthropic_template = PromptTemplate.from_template("""
You are a helpful assistant. You must answer users question IN SPANISH. Be detailed but stay relevant to the question on your answers, following the instructions the user demands. 

To answer the user question, USE THE TOOLS PROVIDED or rely on the Chat History. YOU MUST NOT MAKE UP information. 


You have access to the following tools:

{tools}

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>


For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:

<tool>search</tool><tool_input>weather in SF</tool_input>

<observation>64 degrees</observation>

                                                       
When you are done, respond with a final answer between <final_answer></final_answer>. You must NOT use the tags <tool></tool>, <tool_input></tool_input> or <observation></observation> inside the final answer. For example:

<final_answer>The weather in SF is 64 degrees</final_answer>

Begin!

<ChatHistory>
{chat_history}
</ChatHistory>

                                                       
Question: {input}

{agent_scratchpad}
""")

agent = create_openai_tools_agent(openai, tools, chat_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


agent_anth = create_xml_agent(antrhopic, tools, chat_anthropic_template)
agent_executor_anth = AgentExecutor(agent=agent_anth, tools=tools, verbose=False)