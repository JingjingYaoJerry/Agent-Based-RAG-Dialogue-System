# Implemented based on "Handy Multi-Agent Tutorial"
# and https://docs.camel-ai.org/
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.societies import RolePlaying
from camel.toolkits import FunctionTool
from camel.utils import print_text_animated

from qdrant import QdrantDB
from vector_retriever import VecRetriever

from colorama import Fore
from dotenv import load_dotenv
import os
import json
import shutil


# Load environment variables (i.e., api key) from .env file
load_dotenv()
api_key = os.getenv("QWEN_API_KEY")

def clean_up_db(db: QdrantDB) -> None:
    """Clean up the database at the end"""
    db.storage_instance.close_client() # delete the metadata
    db_path = os.path.join(os.path.dirname(__file__), 'collection')
    # Remove the collection and its metadata files "physically"
    shutil.rmtree(db_path)
    os.remove(os.path.join(os.path.dirname(__file__), '.lock'))
    os.remove(os.path.join(os.path.dirname(__file__), 'meta.json'))


# Initialize the Qdrant database
db = QdrantDB()
doc_num = 0
if str(db.storage_instance.client.count(collection_name='collection0')) != 'count=32':
    # print(str(db.storage_instance.client.count(collection_name='collection0'))) # for debugging
    # Load the pre-processed texts (i.e., docs) from the json file
    with open(r'.\files\small_ocr_content_list.json', 'rb') as file:
        docs = json.load(file)
        # print(docs)
        for doc in docs:
            # Save page indices as file names (for reference)
            file_name = doc.get('page_idx', '')
            content = doc.get('text', '')
            # Use the completed save_text method to save the content to the database
            db.save_text(content, file_name)
            # Demonstrate the saving process
            print(Fore.RED + f"一内容已转换成向量并存入数据库...")
            doc_num += 1  # track the number of documents saved
# print(db.storage_instance.client.count(collection_name='collection0'))  # for debugging
print(Fore.MAGENTA + f"向量数据库已完成存入{doc_num}条数据。" + Fore.RESET)

# Initialize the vector retriever with the pre-defined database and embedding model
retriever = VecRetriever(db)

# Create the LLM model for answer generation (for the agent)
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="qwen-plus-2025-04-28", # model name
    url="https://dashscope.aliyuncs.com/compatible-mode/v1", # url for retrieving the configs
    api_key=api_key
)


# Define the tools functions for the agent
def query_preprocess(query: str) -> str:
    """Preprocess the query for the agent"""
    description = '预处理('
    processed = query.strip()  # simple preprocessing, can be extended
    description += 'str.strip()'
    return f"{description})结果: {processed}"

def answer_type_detection(query: str) -> str:
    """Detect the type of the expected answer based on the query"""
    agent = ChatAgent(
        model=model,
        system_message="""
        你是一个帮助判定答案类型的助手，我会给你原始查询，
        根据下方的答案类型列表，使用其中一种类型判断原始查询的答案类型。\n
        答案类型列表:
        [
            "定义", "原因", "描述",
            "国家", "城市", "地区", "位置",
            "缩写", "表达",
            "群体", "头衔", "个人", "人物",
            "数据", "距离", "比例", "尺寸", "金额", "数值",
            "货币", "食物", "动物", "实体", "时间"
        ]
        """, # set the assistant/system message
        output_language='zh' # set the output language to Chinese
    )
    response = agent.step(input_message=f"请判断原始查询的答案类型: {query}。")
    return f"判定的原始查询答案类型: {response.msg.content}。"

def display_answer(answer: str) -> None:
    """Display the final answer in a formatted way"""
    print(Fore.MAGENTA + f"最终生成答案: {answer}\n" + Fore.RESET)


# Wrap the functions as tools for the agent as instructed in the tutorial
query_preprocess_tool = FunctionTool(query_preprocess)
query_preprocess_tool.set_parameter_description('query', '原始查询字符串')
query_preprocess_tool.set_function_description("""
预处理原始查询，去除多余空格和换行符。
输入: 原始查询字符串
输出: 预处理后的查询字符串
""")
answer_type_detection_tool = FunctionTool(answer_type_detection)
answer_type_detection_tool.set_parameter_description('query', '原始查询字符串')
answer_type_detection_tool.set_function_description("""
判定原始查询的答案类型，返回一个字符串，表示答案类型。
输入: 原始查询字符串
输出: 判定的答案类型字符串
""")
display_answer_tool = FunctionTool(display_answer)
display_answer_tool.set_parameter_description('answer', '达到任务要求的最终答案、总结')
display_answer_tool.set_function_description("""
显示最终答案，格式化输出。
输入: 最终答案字符串
输出: 无（直接打印到控制台）
""")

# Define RolePlaying arguments and the AI-Society
# Based on the tutorial's content (3.2 创建你的第一个Agent Society)
task_kwargs = {
    'task_prompt': '基于“检索到的相关文本片段”，以RAG的方式用一到两句话回答“原始查询” 。',
    'with_task_specify': False, # somehow the prompt is going to be specified automatically
}
user_role_kwargs = {
    'user_role_name': '内容分析师 (ContentAnalyst): 负责分析信息、提取关键点',
    'user_agent_kwargs': {
        'model': model, 
        'tools': [query_preprocess_tool, answer_type_detection_tool, display_answer_tool], 
        # 'system_message': """
        # 你是一个内容分析师，负责分析信息、提取关键点。
        # 你将接收原始查询和相关文本片段，提取关键信息并生成用户消息。
        # 请确保你的回答简洁明了，并包含必要的上下文信息。
        # """
    },   
}
assistant_role_kwargs = {
    'assistant_role_name': '知识解读师 (KnowledgeReader): 负责理解和解释知识内容',
    'assistant_agent_kwargs': {
        'model': model, 
        'tools': [query_preprocess_tool, answer_type_detection_tool, display_answer_tool], 
        # 'system_message': """
        # 你是一个知识解读师，负责理解和解释知识内容。
        # 你将接收用户消息和相关文本片段，生成对原始查询的回答。
        # 请确保你的回答准确、清晰，并包含必要的上下文信息。
        # """
    }, 
}

society = RolePlaying(
    **task_kwargs,
    **user_role_kwargs,
    **assistant_role_kwargs,
    output_language='zh'
)

# Conversation memory: keep recent turns (sliding window) for multi-turn context
conversation_history = []
HISTORY_TURNS = 6 # number of previous turns to include when building the prompt (pairs/entries)


def is_terminated(response):
    """Return whether the conversation is terminated with info"""
    if response.terminated:
        role = response.msg.role_type.name
        reason = response.info['termination_reasons']
        print(f'AI {role} 因为 {reason} 而终止')

    return response.terminated

def run(society, init_msg, round_limit=20):
    """Run the multi-agent conversation loop for a single session.

    The function now prepends a short conversation history (last N turns)
    to the provided `init_msg` so agents can use prior QA context. It also
    appends each round's user/assistant messages into `conversation_history`.
    """
    global conversation_history

    # Inject recent conversation history into the initial message so agents
    # will be aware of prior turns. Keep it short (HISTORY_TURNS entries).
    if conversation_history:
        history_context = "\n## Conversation History (recent turns) ##\n"
        for item in conversation_history[-HISTORY_TURNS:]:
            role_label = item.get('role', 'user')
            if role_label == '用户' or role_label.lower().startswith('user'):
                prefix = 'User: '
            else:
                prefix = 'Assistant: '
            history_context += prefix + item.get('content', '') + "\n"
        init_msg = history_context + "\n" + init_msg

    # Use the retrieved docs as the input message to start the conversation
    input_msg = society.init_chat(init_msg_content=init_msg)

    society.user_sys_msg = f"""
        你是一个内容分析师，负责分析信息、提取关键点。
        你将接收原始查询和相关文本片段，提取关键信息并生成用户消息。
        请确保你的回答简洁明了，并包含必要的相关文本片段。
        请参考消息开头处的“Conversation History”，在可能的情况下使用历史对话作为上下文。
        """
    
    society.assistant_sys_msg = f"""
        你是一个知识解读师，负责理解和解释知识内容。
        你将接收用户消息和相关文本片段，生成对原始查询的回答。
        请确保你的回答准确、清晰，并包含必要的上下文信息。
        在生成回答时，请参考消息开头处注入的历史对话片段。
        """

    # print(society.__dict__) # for debugging

    # 开始互动会话
    for _ in range(round_limit):

        # 获取这一轮的两个响应
        assistant_response, user_response = society.step(assistant_msg=input_msg)

        # Append the round to conversation history so it can be reused later
        try:
            conversation_history.append({"role": "用户", "content": user_response.msg.content})
            conversation_history.append({"role": "助手", "content": assistant_response.msg.content})
        except Exception:
            # If responses are missing, skip recording but continue
            pass

        # 检查终止条件
        if is_terminated(assistant_response) or is_terminated(user_response):
            break

        # 获取结果
        print(Fore.GREEN + f'[AI 用户] {user_response.msg.content}.\n')
        # 检查任务是否结束
        if 'CAMEL_TASK_DONE' in user_response.msg.content:
            break
        print(Fore.BLUE + f'[AI 助手] {assistant_response.msg.content}.\n')

        # 打印助手的输出，包括任何函数执行信息
        tool_calls = user_response.info.get('tool_calls')
        if tool_calls:
            print_text_animated(Fore.GREEN + "Tool Calls (AI用户):")
            for func_record in tool_calls:
                print_text_animated(f"{func_record}")

        tool_calls = assistant_response.info.get('tool_calls')
        if tool_calls:
            print_text_animated(Fore.BLUE + "Tool Calls (AI助手):")
            for func_record in tool_calls:
                print_text_animated(f"{func_record}")

        # 获取下一轮的输入消息
        input_msg = assistant_response.msg

    return None

def test():
    """Test the system with sample queries"""
    tests = [
        "什么是商品的使用价值？",
        "交换价值是什么？",
        "商品的价值是由什么决定的？",
        "什么是社会必要劳动时间？",
        "劳动的⼆重性指什么？"
    ] # use questions from the assignment sheet
    for test in tests:
        print(Fore.RED + f"原始查询问题：{test}")
        results = retriever.search(test, top_k=5) # no need to specify threshold
        # First present the retrieved top-k docs with 'page_idx' references
        # Also, format the user message for the agent
        usr_msg = f"""
        ##原始查询问题##
        {test}

        ##检索到的相关文本片段##
        """
        for idx, item in enumerate(results): 
            print(Fore.CYAN + f"召回文本片段(top{idx+1}): {item['content']}")
            print(f"页面：{item['file_name']}")
            print("-" * 50)
            usr_msg += f"Top {idx+1}: {item['content']} (页面：{item['file_name']})\n"
        # Then pass both the question and retrieved docs to the society for answer generation
        run(society=society, init_msg=usr_msg) # where str is automatically converted to BaseMessage


query = input(Fore.YELLOW + "请输入问题：")
while query:
    if query.strip().lower() == 'exit':
        break
    elif query.strip().lower() == 'test':
        test()
    else:
        print(Fore.RED + f"原始查询问题：{query}")
        results = retriever.search(query, top_k=5) # no need to specify threshold
        # First present the retrieved top-k docs with 'page_idx' references
        # Also, format the user message for the agent
        usr_msg = f"""
        ##原始查询问题##
        {query}

        ##检索到的相关文本片段##
        """
        for idx, item in enumerate(results): 
            print(Fore.CYAN + f"召回文本片段(top{idx+1}): {item['content']}")
            print(f"页面：{item['file_name']}")
            print("-" * 50)
            usr_msg += f"Top {idx+1}: {item['content']} (页面：{item['file_name']})\n"
        # Then pass both the question and retrieved docs to the society for answer generation
        run(society=society, init_msg=usr_msg) # where str is automatically converted to BaseMessage

    query = input(Fore.YELLOW + "请输入问题：")

print(Fore.GREEN + "会话结束, 感谢使用！")
clean_up_db(db=db) # clean up at the end