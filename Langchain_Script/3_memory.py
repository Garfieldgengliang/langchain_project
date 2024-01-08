
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

openai.api_base = "https://api.fe8.cn/v1"
openai.api_key = os.getenv('OPENAI_API_KEY')

# 对话上下文 save_context in BufferMemory
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

history = ConversationBufferMemory()
history.save_context({"input": "你好啊"}, {"output": "你也好啊"})

print(history.load_memory_variables({}))

history.save_context({"input": "你再好啊"}, {"output": "你又好啊"})

print(history.load_memory_variables({}))

# Message 格式
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()

history.add_user_message("你好!")

history.add_ai_message("有什么可以帮您?")

print(history)

# 只保留一个上下文窗口
from langchain.memory import ConversationBufferWindowMemory

window = ConversationBufferWindowMemory(k=2)
window.save_context({"input": "第一轮问"}, {"output": "第一轮答"})
window.save_context({"input": "第二轮问"}, {"output": "第二轮答"})
window.save_context({"input": "第三轮问"}, {"output": "第三轮答"})
print(window.load_memory_variables({}))

# conversation summary
from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI

memory = ConversationSummaryMemory(
    llm=OpenAI(temperature=0),
    # buffer="The conversation is between a customer and a sales."
    buffer="以中文表示"
)
memory.save_context(
    inputs={"input": "你好"}, outputs={"output": "你好，我是你的AI助手。我能为你回答有关AGIClass的各种问题。"})

print(memory.load_memory_variables({}))


# vector database
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import faiss

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS


embedding_size = 1536 # OpenAIEmbeddings的维度
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

# 实际应用中k可以稍大一些，这里k=1演示方便
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

# 把记忆存在向量数据库中
memory.save_context({"input": "我喜欢看电影"}, {"output": "不错啊"})
memory.save_context({"input": "我喜欢踢足球"}, {"output": "下次一起啊"})
memory.save_context({"input": "我不喜欢喝咖啡"}, {"output": "ok"})

# 聊到相关话题，检索之前的记忆
print(memory.load_memory_variables({"prompt": "周末做点体育运动?"})["history"])

