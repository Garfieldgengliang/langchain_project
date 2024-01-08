import re, wordninja
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

openai.api_base = "https://api.fe8.cn/v1"
openai.api_key = os.getenv('OPENAI_API_KEY')


import numpy as np
from numpy import dot
from numpy.linalg import norm
from langchain.embeddings import OpenAIEmbeddings


def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))


def l2(a, b):
    x = np.asarray(a)-np.asarray(b)
    return norm(x)


model = OpenAIEmbeddings(model='text-embedding-ada-002')

query = "国际争端"
documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]

query_vec = model.embed_query(query)
doc_vecs = model.embed_documents(documents)

print("Cosine distance:")  # 越大越相似
print(cos_sim(query_vec, query_vec))
for vec in doc_vecs:
    print(cos_sim(query_vec, vec))

print("\nEuclidean distance:")  # 越小越相似
print(l2(query_vec, query_vec))
for vec in doc_vecs:
    print(l2(query_vec, vec))


from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

query = "国际争端"
documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]

query_vec = model.embed_query(query)
doc_vecs = model.embed_documents(documents)

print("Cosine distance:")  # 越大越相似
print(cos_sim(query_vec, query_vec))
for vec in doc_vecs:
    print(cos_sim(query_vec, vec))



