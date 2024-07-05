from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from googleapiclient.discovery import build
import requests
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.schema import Document
from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
import os
from nltk import ngrams
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams as nltk_ngrams
from supabase import create_client, Client

url = 'https://qbnpppizztfscaswmhbe.supabase.co'
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFibnBwcGl6enRmc2Nhc3dtaGJlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjAyMTU5MzIsImV4cCI6MjAzNTc5MTkzMn0.oTgoozE1RpBLHvVqYj4E5mo_kQv0po0PVfGPFN8D_Aw"
supabase: Client = create_client(url, key)

app = Flask(__name__)
app.secret_key = '001'

def google_search(query, api_key, cse_id):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': api_key,
        'cx': cse_id,
    }
    response = requests.get(url, params=params)
    return response.json()

def perguntas(localizacao, llm):
    localizacao_Chave = llm.predict('What is the region and city of ' + localizacao + '? I just need the name of the region and the city separated with commas')
    language = llm.predict('What is the language spoken in the country of ' + localizacao + '? I just need the name.')
    return language, localizacao_Chave

def link_site(language, search_api_key, search_engine_id, query, llm):
    try:
        query_translated = llm.predict(f'Translate the phrase: {query}. For the language: {language}')
        results = google_search(query_translated, search_api_key, search_engine_id)
        if 'items' in results:
            for item in results['items']:
                link = item['link']
                try:
                    response = requests.get(link, timeout=10)
                    if response.headers.get('content-type', '').startswith('application/pdf'):
                        print(f"Found PDF: {link}")
                        continue
                    elif response.status_code == 200:
                        loader = WebBaseLoader(link)
                        documentos = loader.load()
                        for doc in documentos:
                            page_content = doc.page_content
                            if '403 Forbidden' not in page_content:
                                return link, documentos
                            else:
                                print(f"Link com conteúdo proibido (403 Forbidden): {link}")
                    else:
                        print(f"Link inacessível (status code: {response.status_code}): {link}")
                except requests.RequestException as e:
                    print(f"Erro ao acessar o link: {link}, erro: {e}")

        print("Nenhum link acessível encontrado ou tentativa fora do alcance.")
        return None, None
    except Exception as e:
        print(f"Erro ao obter o link do site: {e}")
        return None, None

def web_processing(language, SEARCH_API_KEY, id, pergunta, llm):
    url, documentos = link_site(language, SEARCH_API_KEY, id, pergunta, llm)
    print(url)

    if documentos is None:
        return None

    chunk_size = 50
    chunk_overlap = 10
    char_split = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['.', '\n\n', '\n']
    )

    splits = char_split.split_documents(documentos)

    embedding_model = GooglePalmEmbeddings(model_name="models/embedding-gecko-001", google_api_key=SEARCH_API_KEY)

    path = 'arquivos/chroma_retrival_bd'

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=path
    )

    metadata_info = [
        AttributeInfo(
            name='source',
            description='Qualquer apostila pode ser retirada',
            type='string'
        )
    ]

    document_description = 'WebSite'
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_description,
        metadata_info,
        verbose=True
    )
    return retriever

def resposta(query, localizacao, SEARCH_API_KEY, id, llm):
    language, localizacao_Chave = perguntas(localizacao, llm)
    query_certa = f'{query} {localizacao_Chave}'
    print(query_certa)
    retriever = web_processing(language, SEARCH_API_KEY, id, query_certa, llm)
    if retriever:
        str_list = [query, localizacao, '?']
        pergunta = ' '.join(str_list)
        docs = retriever.get_relevant_documents(pergunta)
        for doc in docs:
            text = llm.predict(f'Answer the question: {query_certa}. Based on the text: {doc.page_content}')
            print(text)
            return text
    return 'No relevant documents found.'

def resumo(llm, criminal, restaurants, schools, supermarkets, traffic, parks, localizacao, price, area, bedrooms_quantity, bathrooms_quantity):
    pergunta = f'Make a summary in English with the information: Localization: {localizacao}, Price{price}, Area{area}, Bedrooms quantity {bedrooms_quantity}, Bathrooms quantity {bathrooms_quantity}, Criminal rate {criminal} in {localizacao}, Restaurants {restaurants} in {localizacao}, Schools {schools} in {localizacao}, Supermarkets {supermarkets} in {localizacao}, Car Traffic {traffic} in {localizacao}, Parks {parks} in {localizacao}'
    resposta = llm.predict(pergunta)
    print(resposta)
    return resposta

def find_most_similar_index(df, user_description):
    max_similarity = -1
    most_similar_index = -1
    user_ngrams = set(nltk_ngrams(user_description.split(), 2))  
    
    for index, row in df.iterrows():
        description = row['description']
        description_ngrams = set(nltk_ngrams(description.split(), 2))  
        
       
        jaccard_sim = 1 - jaccard_distance(user_ngrams, description_ngrams)
        
        
        if jaccard_sim > max_similarity:
            max_similarity = jaccard_sim
            most_similar_index = index
    
    return most_similar_index

GOOGLE_API_KEY = 'AIzaSyCP52s5khB3lCjyowgnYESbNwWG6rMiHXA'
SEARCH_API_KEY = 'AIzaSyAsKwnYKs4oCZ9dsz24rWGtk8QTkZ9ZCXI'

id = 'a3826ea98d9ca4435'

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

response = supabase.table('data').select('*').execute()
data = response.data
df = pd.DataFrame(data)

@app.route('/', methods=['GET', 'POST'])
def homepage(name=None, df=df):
    num = len(df)
    random_num = np.random.randint(0, num)
    try:
        image_dir = f'./static/img/{random_num}/'

        if os.path.exists(image_dir):
            print("Directory exists.")
            try:
                image_files = os.listdir(image_dir)
                print(f"Files in directory: {image_files}")
            except PermissionError:
                print(f"Permission denied for directory: {image_dir}")
                image_files = []
            except Exception as e:
                print(f"An error occurred: {e}")
                image_files = []
        else:
            print(f"Directory not found: {image_dir}")
            image_files = []

        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

        image_files = [file for file in image_files if any(file.endswith(ext) for ext in image_extensions)]

        if image_files:
            first_image = image_files[0]
            linkcasa1 = f'{image_dir}{first_image}'

        else:
            linkcasa1 = None
        
        hou_price=df.hou_price[random_num]
        area=df.area[random_num]
        area_price=df.area_price[random_num]
        bedrooms=df.bedroom[random_num]
        bathrooms=df.bathroom[random_num]
        description=df.description[random_num]
        est_price=df.est_price[random_num]
        address=df.rua[random_num]
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
        descricao = request.form.get('user_input')
        num_descricao = find_most_similar_index(df = df, user_description = descricao)
        try:
            localizacao = df.rua[num_descricao]
        except:
            localizacao = None
        return redirect(url_for('localizacao_exata', localizacao=localizacao))
    return render_template('homepage.html', person=name, linkcasa1=linkcasa1, description=description, hou_price=hou_price, area=area, area_price=area_price, bedrooms=bedrooms, bathrooms=bathrooms, est_price=est_price, address= address)

@app.route('/sell', methods=['GET', 'POST'])
def sell(df=df, name=None):
    if request.method == 'POST':
        user_input_image = request.files['user_input_image']
        user_input_address = request.form.get('user_input_address')
        user_input_price = request.form.get('user_input_price')
        user_input_area = request.form.get('user_input_area')
        user_input_bedrooms = request.form.get('user_input_bedrooms')
        user_input_bathrooms = request.form.get('user_input_bathrooms')
        user_input_price_per_area = request.form.get('price_per_area')

        if not (user_input_image and user_input_address and user_input_price and user_input_area and user_input_bedrooms and user_input_bathrooms):
            flash('All fields must be filled out.')
            return redirect(url_for('sell'))


        num = len(df)+1
        image_path = f'static/img/{num}{user_input_image.filename}'
        user_input_image.save(image_path)

        criminal = resposta(
            query='Criminal rate ',
            localizacao=user_input_address,
            SEARCH_API_KEY=SEARCH_API_KEY,
            id=id,
            llm=llm
        )
        restaurants = resposta(
            query='Restaurants in ',
            localizacao=user_input_address,
            SEARCH_API_KEY=SEARCH_API_KEY,
            id=id,
            llm=llm
        )
        schools = resposta(
            query='Schools in ',
            localizacao=user_input_address,
            SEARCH_API_KEY=SEARCH_API_KEY,
            id=id,
            llm=llm
        )
        supermarkets = resposta(
            query='Supermarkets in ',
            localizacao=user_input_address,
            SEARCH_API_KEY=SEARCH_API_KEY,
            id=id,
            llm=llm
        )
        traffic = resposta(
            query='Car traffic in ',
            localizacao=user_input_address,
            SEARCH_API_KEY=SEARCH_API_KEY,
            id=id,
            llm=llm
        )

        parks = resposta(
            query='Parks in ',
            localizacao=user_input_address,
            SEARCH_API_KEY=SEARCH_API_KEY,
            id=id,
            llm=llm
        )

        final_answer = resumo(
            llm=llm,
            criminal=criminal,
            restaurants=restaurants,
            schools=schools,
            supermarkets=supermarkets,
            traffic=traffic,
            parks=parks,
            localizacao = user_input_address,
            price=user_input_price,
            area=user_input_area,
            bedrooms_quantity=user_input_bedrooms,
            bathrooms_quantity=user_input_bathrooms,
        )
        
        supabase.table('data').insert({'id':len(df), 'rua': user_input_address, 'hou_price': user_input_price, 'area': user_input_area, 'area_price': user_input_price_per_area, 'bedroom': user_input_bedrooms, 'bathroom': user_input_bathrooms, 'description': final_answer, 'est_price': 0, 'past': image_path }).execute()

        print('informations saved')
        flash('Information saved successfully.')

    return render_template('sell.html')

@app.route('/localizacao/<path:localizacao>')
def localizacao_exata(localizacao, df=df, name=None):
    for num, address in enumerate(df.rua):
        if address == localizacao:
            try:
                image_dir = f'./static/img/{num}/'

                if os.path.exists(image_dir):
                    print("Directory exists.")
                    try:
                        image_files = os.listdir(image_dir)
                        print(f"Files in directory: {image_files}")
                    except PermissionError:
                        print(f"Permission denied for directory: {image_dir}")
                        image_files = []
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        image_files = []
                else:
                    print(f"Directory not found: {image_dir}")
                    image_files = []

                
                first_image = image_files[0]
                linkcasa1 = f'{image_dir}{first_image}'
                print(linkcasa1)

                hou_price=df.hou_price[num]
                area=df.area[num]
                area_price=df.area_price[num]
                bedrooms=df.bedroom[num]
                bathrooms=df.bathroom[num]
                description=df.description[num]
                est_price=df.est_price[num]
                break
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
        else:
            linkcasa1 = None
            hou_price=None
            area=None
            area_price=None
            bedrooms=None
            bathrooms=None
            description=None
            est_price=None
            address=None
    return render_template('page.html', person=name, linkcasa1=linkcasa1, description=description, hou_price=hou_price, area=area, area_price=area_price, bedrooms=bedrooms, bathrooms=bathrooms, est_price=est_price, address=address)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
