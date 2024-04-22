
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
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_xml_agent, create_tool_calling_agent
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI


from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.utils.function_calling import convert_to_openai_function

import unicodedata
import dill as pickle

from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch

## INICIAR LANGSMITH Y API KEYS
dotenv_path = here() / ".env"
load_dotenv(dotenv_path=dotenv_path)


client = Client()

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = f"API - {unique_id}"


# LEVANTAR DATOS
def remover_tildes(input_str):
    # Normalizar la cadena de texto a 'NFD' para descomponer los acentos
    normalized_str = unicodedata.normalize('NFD', input_str)
    # Filtrar para quitar los caracteres de combinación (diacríticos)
    return ''.join(c for c in normalized_str if unicodedata.category(c) != 'Mn')

with open('metodos_obj_str.pkl', 'rb') as archivo:
    metodos_lista = pickle.load(archivo)

# crear_dataset = True
# if crear_dataset:
#     db = Chroma(persist_directory="db_RAG", embedding_function=OpenAIEmbeddings())
#     db.delete_collection()
#     docs = []

#     for metod in metodos_lista:
#         docs.append(Document(page_content=metod.descripcion, metadata={"nombre":metod.nombre,"sistema":metod.sistema}))

#     db_ret = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory="db_RAG")
# else:

local = int(os.environ["LOCAL"])

if local:
    db_ret = Chroma(persist_directory="db_RAG", embedding_function=OpenAIEmbeddings())
else:
    password = os.environ["DB_PASSWORD"]
    CONN_STR = f"mongodb+srv://panda:{password}@asistentes.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
    DB_NAME = "APIasistente"
    COLLECTION_NAME = "asistente_api_2"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "index_name"
    NAMESPACE = DB_NAME + '.' + COLLECTION_NAME
    db_ret = AzureCosmosDBVectorSearch.from_connection_string(
        CONN_STR, NAMESPACE, OpenAIEmbeddings(chunk_size=1), index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )

# DEFINIR TOOLS

class sistem(BaseModel):
    sistem: str = Field(description="Sistem to which get the method")

class method_decription(BaseModel):
    method_description: str = Field(description="Description of what the method should do")
    sistem: str = Field(description="Sistem to which the method belongs to", default="all")

class method_name(BaseModel):
    method: str = Field(description="Method to search information of")
    sistem: str = Field(description="Sistem to which the method belongs to", default="all")

@tool("get_methods_from_description", args_schema=method_decription)
def get_method(method_description:str, sistem: str="all"):
    """ Returns a list of possible methods provided a description. The parameter sistem is used to filter the search of the methods to a given sistem"""
    filter={}
    if sistem != "all":
        filter = {"sistema":remover_tildes(sistem.replace(" ", ""))}
    ret_metods = db_ret.similarity_search(method_description,k=10,filter=filter)
    ret_metods_names = [met.metadata["nombre"] for met in ret_metods]
    ret_metods_sistems = [met.metadata["sistema"] for met in ret_metods]
    ret_metodos_obj = []
    indices_list = []
    for met in metodos_lista:
        if met.nombre in ret_metods_names:
            indices = [indice for indice, elemento in enumerate(ret_metods_names) if elemento == met.nombre]
            for i in indices:
                if met.sistema in ret_metods_sistems[i]: 
                    ret_metodos_obj.append(met)
                    indices_list.append(i)
    ret_metodos_obj_orden = sorted(zip(indices_list, ret_metodos_obj))
    ret_metodos_obj = [elemento for valor, elemento in ret_metodos_obj_orden]
    resp = ""
    for metod in ret_metodos_obj:
        resp +=  method_info_as_string(metod) + "\n"
    return resp

@tool("get_method_info_from_name", args_schema=method_name)
def get_method_info(method:str, sistem: str="all"):
    """ Returns information (input-ouput squeema, possible errors and calling example using Soap or JSON) of a method. The parameter sistem is used to filter the search of the method to a given sistem """

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


@tool("get_all_method_from_sistem", args_schema=sistem)
def get_all_method_from_sistem(sistem: str="all"):
    """ Returns all methods for a given sistem. """

    ret_metodos_obj = []
    for met in metodos_lista:
         if met.sistema.lower().strip() == remover_tildes(sistem.lower().strip()):
            ret_metodos_obj.append(met)


    if len(ret_metodos_obj) == 0:
        resp = "El sistema {sistem} no es parte de Bantotal"
    else:
        resp = ""
        for metod in ret_metodos_obj:
            resp +=  method_info_as_string(metod) + "\n"
            
    return resp

def method_info_as_string(metod,params_info=False):
    strng  = ""
    strng += f"Metodo: {metod.nombre} \n Sistema: {metod.sistema} \n Descripción: {metod.descripcion} \n "
    if params_info:
        strng  += f"{metod.entrada} \n {metod.salida} \n {metod.error} \n"
        strng  += f"{metod.ej_in} \n {metod.ej_out}"
    return strng

tools = [get_method, get_method_info,get_all_method_from_sistem]


# DEFINIR AGENTE

# openai = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.0,streaming=True)
openai  = AzureChatOpenAI(
    deployment_name="gpt-35-turbo-16k",
    temperature=0.0
)
antrhopic = ChatAnthropic(temperature=0, model_name="claude-3-haiku-20240307")

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", """Task: You are a helpful assistant, expert on the API documentation of Bantotal. You must answer users question IN SPANISH. 

Instructions: All information in your answers must be retrieved from the use of the tools provided or based on previous information from the chat history. In case the question can´t be answered  using the tools provided (It is not relevant to the API documentation) honestly say that you can not answer that question.

Be detailed in your answers but stay focused to the question. Add all details that are useful to provide a complete answer, but do not add details beyond the scope of the question.

When using tools, filter by sistem when possible to enhace performance. The list of possible sistems is
<sistemas>
AhorroProgramado
CadenadeCierre
Calendarios
CASHManagement
Clientes
ConfiguracionBantotal
Contabilidad
CuentasCorrientes
CuentasdeAhorro
CuentasVista
DepositosaPlazo
DescuentodeDocumentos
Indicadores
Microfinanzas
ModeladordePrestamos
PAE
ParametrosBase
Personas
Precios
Prestamos
ReglasdeNegocio
Seguridad
TarjetasdeDebito
Usuarios
Workflow
</sistemas>
        
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
) 
chat_template_v1 = ChatPromptTemplate.from_messages(
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