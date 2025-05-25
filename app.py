import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
import os
from nltk.util import ngrams
from nltk.metrics import jaccard_distance
import nltk
from supabase import create_client, Client

url = ''#<-- URL do Supabase
key = ""#<-- Chave do Supabase
supabase: Client = create_client(url, key)

app = Flask(__name__)
app.secret_key = ''#<-- Chave secreta para o Flask

#nltk.download('punkt')

def find_most_similar_index(df, user_description):
    max_similarity = -1
    most_similar_index = -1
    
    user_ngrams = set(ngrams(nltk.word_tokenize(user_description), 2))
    
    for index, row in df.iterrows():
        description = row['Desc']
        
        description_ngrams = set(ngrams(nltk.word_tokenize(description), 2))
        
        jaccard_sim = 1 - jaccard_distance(user_ngrams, description_ngrams)
        
        if jaccard_sim > max_similarity:
            max_similarity = jaccard_sim
            most_similar_index = index
    
    return most_similar_index

OPENAI_API_KEY = ''#<-- Chave da OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct", api_key= OPENAI_API_KEY)

#response = supabase.table('data').select('*').execute()
#data = response.data
#df = pd.DataFrame(data)

df = pd.read_csv(filepath_or_buffer='arquivos/Italian_house_price.csv', sep=',', decimal='.')
df['Desc'] = df['Desc'].astype(str)

@app.route('/', methods=['GET', 'POST'])
def homepage(name=None, df=df):
    num = len(df)
    random_num = np.random.randint(0, num)
    try:
        linkcasa1 = None
        hou_price=df['Price(€)'][random_num]
        area=df['mq'][random_num]
        area_price=None
        bedrooms=df['Rooms'][random_num]
        bathrooms=None
        description=df['Desc'][random_num]
        est_price=None
        address=df['Street'][random_num]
    except:
        linkcasa1 = None
        hou_price=None
        area=None
        area_price=None
        bedrooms=None
        bathrooms=None
        description=None
        est_price=None
        address=None

    if request.method == 'POST':
        descricao_usuario = request.form.get('user_input')
        templateLingua = PromptTemplate.from_template('''
        Responda na lingua {lingua}.
        ''')
        templatePergunta = PromptTemplate.from_template ('''
        Extraia as informacoes principais da descricao, excluindo opinioes, como cor, se possui ou nao varanda: {descricao}
        ''')
        templateFinal = (
            templateLingua
            + 'Faca um texto simples tentabndo usar o minimo de palavras possiveis'
            + 'Responda a seguinte questao: '
            + templatePergunta
        )
        pergunta = templateFinal.format(lingua = 'italiano', descricao = descricao_usuario)
        descricao_formatada = llm.invoke(pergunta)
        print(descricao_formatada)
        num_descricao = find_most_similar_index(df = df, user_description = descricao_formatada)
        return redirect(url_for('localizacao_exata', index = num_descricao))
    return render_template('homepage.html', person=name, linkcasa1=linkcasa1, description=description, hou_price=hou_price, area=area, area_price=area_price, bedrooms=bedrooms, bathrooms=bathrooms, est_price=est_price, address= address)

@app.route('/sell', methods=['GET', 'POST'])
def sell(df=df, name=None, llm=llm):
    if request.method == 'POST':
        user_input_image = request.files['user_input_image']
        user_input_address = request.form.get('user_input_address')
        user_input_price = request.form.get('user_input_price')
        user_input_area = request.form.get('user_input_area')
        user_input_bedrooms = request.form.get('user_input_bedrooms')

        if not (user_input_image and user_input_address and user_input_price and user_input_area and user_input_bedrooms ):
            flash('All fields must be filled out.')
            return redirect(url_for('sell'))


        num = len(df)+1
        image_path = f'static/img/{num}{user_input_image.filename}'
        user_input_image.save(image_path)

        templateCaracteristicasTexto = PromptTemplate.from_template('''
        Responda a pergunta em ate {num_paragrafos} paragrafos.
        ''')
        templateLingua = PromptTemplate.from_template('''
        Responda na lingua {lingua}.
        ''')
        templateCaracteristicasRuas = PromptTemplate.from_template('''
        Quais sao as caracteristicas do endereco {endereco}
        ''')
        templateCaracteristicasCasa = PromptTemplate.from_template('''
        As caracteristicas da casa sao as seguintes: 
        Quartos = {quantidadeQuartos}                                                        
        ''')
        templateFinal = (
            templateCaracteristicasTexto
            + templateLingua
            + 'Responda a seguinte questao: Faca uma descricao com as informacoes:'
            + templateCaracteristicasRuas
            + templateCaracteristicasCasa
        )
        prompt = templateFinal.format(num_paragrafos=2,lingua='italiano',endereco=user_input_address,quantidadeQuartos=user_input_bedrooms)
        description = llm.invoke(prompt)

        #supabase.table('data').insert({'id':len(df), 'rua': user_input_address, 'hou_price': user_input_price, 'area': user_input_area, 'area_price': user_input_price_per_area, 'bedroom': user_input_bedrooms, 'bathroom': user_input_bathrooms, 'description': final_answer, 'est_price': 0, 'past': image_path }).execute()
        nova_linha = {'Price(€)':user_input_price, 'Rooms': user_input_bedrooms, 'Desc': description, 'Street': user_input_address}
        df.loc[len(df)] = nova_linha

        print('informations saved')
        flash('Information saved successfully.')

    return render_template('sell.html')

@app.route('/localizacao/<path:index>')
def localizacao_exata(index, df=df, name=None):
    
    try:
        index = int(index)
        
        if index < 0 or index >= len(df):
            raise IndexError

        linkcasa1 = None
        hou_price = df['Price(€)'][index]
        area = df['mq'][index]
        bedrooms = df['Rooms'][index]
        description = df['Desc'][index]
        address = df['Street'][index]

        return render_template('page.html', person=name, linkcasa1=linkcasa1, description=description, hou_price=hou_price, area=area, bedrooms=bedrooms, address=address)
    
    except ValueError:
        return "Índice inválido, deve ser um número inteiro.", 400
    
    except IndexError:
        return "Índice fora do intervalo.", 404

def get_user_ip():
    return request.remote_addr

@app.route('/user_ip')
def user_ip():
    ip = get_user_ip()
    return f'Seu IP: {ip}'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
