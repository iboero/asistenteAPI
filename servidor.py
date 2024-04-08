#IMPORTS 

from flask import Flask, request, render_template_string, jsonify,Response, stream_with_context, request
from langchain_core.output_parsers import  StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import dill as pickle

from dotenv import load_dotenv
from pyprojroot import here
import os
from uuid import uuid4
from langsmith import Client

import csv
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from chain import agent_executor,agent_executor_anth

import asyncio


# LEVANTAR SERVIDOR Y HTML
app = Flask(__name__)

def importar_html_como_string(archivo_html, codificacion='utf-8'):
    with open(archivo_html, 'r', encoding=codificacion) as archivo:
        contenido = archivo.read()
    return contenido

HTML = importar_html_como_string("pagina_web.html"  )


# VARIABLES GLOBALES
selected_option = None
ultima_respuesta = ""
ultima_pregunta = ""
chat_history = []

# FUNCIONES ASOCIADAS A JS

# Levantar servidor
@app.route('/')
def home():
    global chat_history
    chat_history = []
    return render_template_string(HTML)

@app.route('/health')
def health_check():
    return 'OK', 200

# ## Manejar Respuesta
def generate_data(message):
    global ultima_respuesta
    resp = ""
    chat_history_string = ""
    for m in chat_history:
        chat_history_string += f"{m.type}: {m.content} \n"
    # resp = agent_executor.invoke({"input":message,"chat_history":chat_history})["output"]
    resp = agent_executor.invoke({"input":message,"chat_history":chat_history})["output"]

    resp_html = resp.replace("\n", "||")
    yield f"data: {resp_html}\n\n"
    ultima_respuesta = resp
    chat_history.append(HumanMessage(content=message))
    chat_history.append(AIMessage(content=resp))
    # resp = resp.replace("\n", "||")
    # yield f"data: {resp}\n\n"
    yield "data: done\n\n"

# def generate_data(message):
#     # Esta es la función que realmente queremos ejecutar de forma asíncrona
#     async def async_generator():
#         global chat_history
#         global ultima_respuesta
#         resp = ""
#         print("Genrando chunks")
#         async for chunk in agent_executor.astream_events({"input":message,"chat_history":chat_history}, version="v1"):
#             print(chunk)
#             if chunk["event"] == "on_chat_model_stream":
#                 content = chunk["data"]["chunk"].content.replace("\n", "||")
#                 resp += chunk["data"]["chunk"].content
#                 yield f"data: {content}\n\n"
#         chat_history.append(HumanMessage(content=message))
#         chat_history.append(AIMessage(content=resp))
#         ultima_respuesta = resp
#         yield "data: done\n\n"

#     # Crear un nuevo evento loop en el hilo actual
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

#     # Ejecutar la corutina en el hilo y recoger los resultados
#     try:
#         async_gen = async_generator()
#         while True:
#             try:
#                 data = loop.run_until_complete(async_gen.__anext__())
#                 yield data
#             except StopAsyncIteration:
#                 break
#     finally:
#         loop.close()

@app.route('/send', methods=['POST'])
def send():
    global ultima_pregunta
    global ultima_respuesta
    data = request.json
    ultima_pregunta = data['message']
    return jsonify({'status': 'success'}), 200


@app.route('/stream')
def stream():
    print("Respondiendo pregunta")
    global ultima_pregunta
    return Response(stream_with_context(generate_data(ultima_pregunta)), mimetype='text/event-stream')


## Cambiar Sistema
@app.route('/update_option', methods=['POST'])
def update_option():
    global selected_option
    data = request.json
    selected_option = data['option']
    print("Opción actualizada a:", selected_option)
    return jsonify({'status': 'success'})

@app.route('/new_chat', methods=['POST'])
def handle_new_chat():
    global chat_history
    chat_history = []
    return jsonify({'status': 'new chat started'})  # Respuesta opcional


## Manejar Feedback
@app.route('/feedback', methods=['POST'])
def feedback():
    global ultima_pregunta
    global ultima_respuesta
    global ultima_seccion
    data = request.json
    feedback = data['feedback']
    positive = int(data["positive"])*2 - 1
    with open('feedback.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Pregunta","Respuesta","Seccion","Positivo","Comentario"])
        writer.writerow([ultima_pregunta,ultima_respuesta,selected_option, positive, feedback])
    print([ultima_pregunta,ultima_respuesta, feedback])
    return jsonify({'status': 'Feedback recibido'})


# CORRER SERVIDOR
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


