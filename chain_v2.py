
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
from langchain_openai import ChatOpenAI, OpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_xml_agent, create_tool_calling_agent
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.output_parsers  import  JsonOutputParser
from typing import List, Tuple, Dict, Type, Any

from langchain.pydantic_v1 import BaseModel, Field,create_model
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.utils.function_calling import convert_to_openai_function

import unicodedata
import dill
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch

from azure.search.documents.models import VectorFilterMode
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

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
    metodos_lista = dill.load(archivo)


embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002")



index_name = "api5"
credential = AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY"))
endpoint = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")

embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002")
search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)


# DEFINIR TOOLS

class sistem(BaseModel):
    sistem: str = Field(description="Sistem to which get the method")

class method_decription(BaseModel):
    method_description: str = Field(description="Description of what the method should do")

class method_name(BaseModel):
    method: str = Field(description="Method to search information of")
    sistems: List[str] = Field(description="Sistems raleted to the method")




## TOOL BUSCAR METODO
@tool("get_methods_from_description", args_schema=method_decription)
def get_method(method_description:str):
    """ Returns a list of relevant methods provided a description. The parameter sistem is used to filter the search of the methods to a given sistem"""

    # Pure Vector Search
    vector_query = VectorizedQuery(vector=embeddings.embed_query(method_description), k_nearest_neighbors=15, fields="content_vector", exhaustive=True)
    results = search_client.search(vector_queries= [vector_query],select=["sistem", "content","id"])

    ret_metods_names = []
    ret_metods_sistems = []
    for res in results:
        ret_metods_names.append(res["id"].split("_")[1])
        ret_metods_sistems.append(res["id"].split("_")[0])
        
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


## TOOL INFORMACION DE UN METODO
@tool("get_method_info_from_name", args_schema=method_name)
def get_method_info(method:str, sistem: str="all"):
    """ Returns information (input-ouput squeema, possible errors and calling example using Soap or JSON) of BTServices methods. The parameter sistem is used to filter the search of the method to a given sistem """

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
        filter=""
        if sistem != "all":
            filter = f"sistem eq '{sistem}'"
        # Pure Vector Search
        vector_query = VectorizedQuery(vector=embeddings.embed_query(method), k_nearest_neighbors=10, fields="content_vector", exhaustive=True)
        results = search_client.search(vector_queries= [vector_query],filter=filter,select=["sistem", "content","id"])

        ret_metods_names = []
        ret_metods_sistems = []
        for res in results:
            ret_metods_names.append(res["id"].split("_")[1])
            ret_metods_sistems.append(res["id"].split("_")[0])
        for met in metodos_lista:
            if met.nombre in ret_metods_names:
                indices = [indice for indice, elemento in enumerate(ret_metods_names) if elemento == met.nombre]
                if met.sistema in [ret_metods_sistems[i] for i in indices]:
                    ret_metodos_obj.append(met)
    resp = ""
    for metod in ret_metodos_obj:
        resp +=  method_info_as_string(metod,params_info=True) + "\n"
    return resp



## TOOL METODOS DE UN SISTEMA
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




## CREAR TOOLS DE MANUALS

path = "./Manuales/"
manuales = []
for filename in os.listdir(path):
    with open(path + filename, 'rb') as file:
        manuales.append(dill.load(file))


def class_from_section(sections: List[Tuple[int, str]]) -> Type[BaseModel]:
    level_1_sections = {name.replace(' ', '_').lower(): (bool, Field(default=False,description="Whether to return the content of this section"))
                        for level, name in sections if level == 1}
    class_dict = {}
    for name in level_1_sections.keys():
         field = (bool, Field(default=False,description="Whether to return the content of this section"))
         class_dict[name] = field
    ManualSections = create_model('ManualSections', **class_dict)    
    return ManualSections



def create_manual_tool_2(description, manual_pkl,args_schema):
    def decorator(func):
        func.__doc__ = description
        return tool("retrieve_information_from_" + manual_pkl.title.replace(" ", "_").lower(), args_schema=args_schema)(func)

    @decorator
    def retrieve_information(**args_schema):
        # Elijo el manual específico
        selected_manual = manual_pkl
        sections = []
        for idx, ret_sec in enumerate(args_schema.values()):
            if ret_sec:
                sections.append(str(idx+1))
        # Recupero el contenido de las secciones
        content = ""
        for sec in sections:
            content += selected_manual.get_section(sec) + '\n\n'
        return content

    return retrieve_information
# Example of creating a tool for a specific manual

manual_tools = []
for manual in manuales:
    manual_name = manual.title
    description = f'Retrieves relevant information from the manual using semantic search. The information stored in the manual is: "{manual.description}'
    manual_tools.append(create_manual_tool_2(description, manual,class_from_section(manual.structure)))
    # manual_tools.append(create_manual_tool(description, manual,manual_search_args))



def method_info_as_string(metod,params_info=False):
    strng  = ""
    strng += f"Metodo: {metod.nombre} \n Sistema: {metod.sistema} \n Descripción: {metod.descripcion} \n "
    if params_info:
        strng  += f"{metod.entrada} \n {metod.salida} \n {metod.error} \n"
        strng  += f"{metod.ej_in} \n {metod.ej_out}"
    return strng

tools = [get_method, get_method_info,get_all_method_from_sistem] + manual_tools




# DEFINIR AGENTE

openai = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.0,streaming=True)



chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", """Task: 

You are a helpful assistant, expert on the BTServices of Bantotal. BTServices provides an API to access the Bantotal Core data tables and programs. You must answer users question IN SPANISH. 

Instructions: 

1) All information in your answers MUST BE RETRIEVED from the use of the tools. DO NOT MAKE INFORMATION UP. 

2) In case the question can´t be answered  using the tools provided (It is not relevant to BTservices) honestly say that you can not answer that question.

3) The user must be transparent to the tools you are using to retrieve the information. If a tool needs to be used, use it without consulting the user. 

4) Be detailed in your answers but stay focused to the question. Add all details that are useful to provide a complete answer, but do not add details beyond the scope of the question.

5) All tools can be used as many times as needed. If you are not able to provide a complete answer with the output of one tool, try using the same tool with different parameters or a new tool.

6) When using tools, consider filteing by sistem when possible to enhace performance. The list of possible sistems is
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




agent = create_openai_tools_agent(openai, tools, chat_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
