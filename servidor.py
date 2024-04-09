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
# selected_option = None
ultima_respuesta = {}
ultima_pregunta = {}
chat_histories = {}

# FUNCIONES ASOCIADAS A JS

# Levantar servidor
@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/health')
def health_check():
    return 'OK', 200

# ## Manejar Respuesta
def generate_data(message,tab_id):
    global ultima_respuesta
    global chat_histories
    resp = ""
    chat_history = chat_histories[tab_id]
    # chat_history_string = ""
    # for m in chat_history:
    #     chat_history_string += f"{m.type}: {m.content} \n"
    # resp = agent_executor.invoke({"input":message,"chat_history":chat_history})["output"]
    resp = agent_executor.invoke({"input":message,"chat_history":chat_history})["output"]

    resp_html = resp.replace("\n", "||")
    yield f"data: {resp_html}\n\n"
    ultima_respuesta[tab_id] = resp
    chat_history.append(HumanMessage(content=message))
    chat_history.append(AIMessage(content=resp))
    chat_histories[tab_id] = chat_history
    yield "data: done\n\n"


@app.route('/send', methods=['POST'])
def send():
    global ultima_pregunta
    global ultima_respuesta
    data = request.json
    tab_id = data['tabId']
    if tab_id not in chat_histories:
        chat_histories[tab_id] = []
    ultima_pregunta[tab_id] = data['message']
    return jsonify({'status': 'success'}), 200


@app.route('/stream')
def stream():
    tab_id = request.args.get('tabId')
    print(F"Respondiendo pregunta de {tab_id}")
    global ultima_pregunta
    return Response(stream_with_context(generate_data(ultima_pregunta[tab_id],tab_id)), mimetype='text/event-stream')


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
    global chat_histories
    data = request.json
    tab_id = data['tabId']
    chat_histories[tab_id] = []
    return jsonify({'status': 'new chat for {tabId}'})  # Respuesta opcional


## Manejar Feedback
@app.route('/feedback', methods=['POST'])
def feedback():
    global ultima_pregunta
    global ultima_respuesta
    data = request.json
    feedback = data['feedback']
    tab_id = data['tabId']
    positive = int(data["positive"])*2 - 1
    with open('feedback.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Pregunta","Respuesta","Seccion","Positivo","Comentario"])
        writer.writerow([ultima_pregunta[tab_id],ultima_respuesta[tab_id],selected_option, positive, feedback])
    print([ultima_pregunta,ultima_respuesta, feedback])
    return jsonify({'status': 'Feedback recibido'})


# CORRER SERVIDOR
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


