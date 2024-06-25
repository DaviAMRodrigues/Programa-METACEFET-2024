from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
import requests
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

'''
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path='arquivos/langchanin_cache_db.sqlite'))

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
                        try:
                            loader = PyPDFDirectoryLoader(link)
                            documentos = loader.load()
                            return link, documentos
                        except:
                            print('Erro ao abrir o pdf')
                            return None, None
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
            text = llm.predict(f'Answer the question: {query_certa}. Based on the text: {doc.page_content} or in {doc.metadata["description"]}')
            print(text)
            return text
    return 'No relevant documents found.'

def resumo(llm, criminal, restaurants, schools, supermarkets, traffic, parks, localizacao):
    pergunta = f'Make a summary in English with the information: Criminal rate {criminal} in {localizacao}, Restaurants {restaurants} in {localizacao}, Schools {schools} in {localizacao}, Supermarkets {supermarkets} in {localizacao}, Car Traffic {traffic} in {localizacao}, Parks {parks} in {localizacao}'
    resposta = llm.predict(pergunta)
    return resposta

GOOGLE_API_KEY = 'AIzaSyCP52s5khB3lCjyowgnYESbNwWG6rMiHXA'
SEARCH_API_KEY = 'AIzaSyAsKwnYKs4oCZ9dsz24rWGtk8QTkZ9ZCXI'

id = 'a3826ea98d9ca4435'

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

@app.route('/')
def homepage(name=None):
    return render_template('homepage.html', person=name)

localizacao = '242 Bay Ridge Ave, Brooklyn, NY 11220, EUA'


criminal = resposta(
    query='Criminal rate ',
    localizacao=localizacao,
    SEARCH_API_KEY=SEARCH_API_KEY,
    id=id,
    llm=llm
)
restaurants = resposta(
    query='Restaurants in ',
    localizacao=localizacao,
    SEARCH_API_KEY=SEARCH_API_KEY,
    id=id,
    llm=llm
)
schools = resposta(
    query='Schools in ',
    localizacao=localizacao,
    SEARCH_API_KEY=SEARCH_API_KEY,
    id=id,
    llm=llm
)
supermarkets = resposta(
    query='Supermarkets in ',
    localizacao=localizacao,
    SEARCH_API_KEY=SEARCH_API_KEY,
    id=id,
    llm=llm
)
traffic = resposta(
    query='Car traffic in ',
    localizacao=localizacao,
    SEARCH_API_KEY=SEARCH_API_KEY,
    id=id,
    llm=llm
)

parks = resposta(
    query='Parks in ',
    localizacao=localizacao,
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
    localizacao = localizacao,
)

print(final_answer)
'''