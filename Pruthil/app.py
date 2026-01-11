from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil
from datetime import datetime, timezone, timedelta
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_cloud_services import LlamaCloudIndex
import uvicorn
import firebase_admin
from firebase_admin import credentials, firestore, storage
import uuid
import mimetypes
import time
import random


from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated

from langchain_groq import ChatGroq
try:
    import groq as _groq_module
    GroqRateLimitError = getattr(_groq_module, "RateLimitError", None)
except Exception:
    GroqRateLimitError = None
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool

app = FastAPI(title="Multi-Format RAG API with ReAct Agent", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "https://huggingface.co",
        "https://*.hf.space",
        "http://localhost:*",
        "http://127.0.0.1:*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

UPLOAD_DIR = "./uploaded_docs"

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
LLAMA_CLOUD_INDEX_NAME = os.getenv("LLAMA_CLOUD_INDEX_NAME", "howlingunicron")
LLAMA_CLOUD_PROJECT_NAME = os.getenv("LLAMA_CLOUD_PROJECT_NAME", "Default")
LLAMA_CLOUD_ORGANIZATION_ID = os.getenv("LLAMA_CLOUD_ORGANIZATION_ID", "7a42adaa-6a50-4dd5-aea9-fe9205202488")
LLAMA_CLOUD_PIPELINE_ID = "cfcc72a2-8fc0-4f01-acce-a0d91c955c89"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not LLAMA_CLOUD_API_KEY:
    print("⚠️ WARNING: LLAMA_CLOUD_API_KEY not found in environment!")
if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not found in environment!")
if not SERPAPI_API_KEY:
    print("⚠️ WARNING: SERPAPI_API_KEY not found - Web search will be disabled")
else:
    print("✅ SERPAPI_API_KEY loaded successfully")

SUPPORTED_FILE_TYPES = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.doc': 'application/msword',
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.rtf': 'application/rtf',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.xls': 'application/vnd.ms-excel',
    '.csv': 'text/csv',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.ppt': 'application/vnd.ms-powerpoint',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
    '.tiff': 'image/tiff',
    '.html': 'text/html',
    '.xml': 'text/xml',
    '.json': 'application/json',
    '.py': 'text/x-python',
    '.js': 'text/javascript',
    '.java': 'text/x-java',
    '.cpp': 'text/x-c++',
    '.c': 'text/x-c',
}

FIREBASE_CONFIG = {
    "type": "service_account",
    "project_id": os.getenv("FIREBASE_PROJECT_ID", "chat-app-backend-a94c5"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.getenv("FIREBASE_CERT_URL")
}

FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "chat-app-backend-a94c5.appspot.com")

if os.getenv("FIREBASE_PRIVATE_KEY"):
    print("✅ Firebase credentials found in environment")
else:
    print("⚠️ Firebase credentials not found - running in local mode without persistence")

os.makedirs(UPLOAD_DIR, exist_ok=True)

llama_index = None
retriever = None
llm = None
db_firestore = None
bucket = None
web_search_tool = None
agent_apps = {}  

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    conversation_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"
    enable_search: Optional[bool] = False  
    temperature: Optional[float] = 0.0  

class QueryResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: Optional[List[dict]] = None
    timestamp: str
    used_web_search: Optional[bool] = False
    agent_steps: Optional[List[dict]] = []  

class UploadResponse(BaseModel):
    message: str
    file_id: str
    filename: str
    file_type: str
    file_size: int
    user_id: str
    firebase_url: Optional[str] = None
    timestamp: str
    status: str

class ConversationMessage(BaseModel):
    question: str
    answer: str
    timestamp: str
    sources: Optional[List[dict]] = None
    used_web_search: Optional[bool] = False

class Conversation(BaseModel):
    conversation_id: str
    user_id: str
    messages: List[ConversationMessage]
    created_at: str
    updated_at: str

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    context: str
    conversation_history: List[dict]


def initialize_llm(temperature: float = 0.0):
    
    try:
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        print(f"✅ Groq LLM initialized with temperature={temperature}")
        return llm
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        return None


def initialize_web_search_tool():
    
    global web_search_tool
    
    if SERPAPI_API_KEY:
        try:
            serp_search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
            web_search_tool = Tool(
                name="web_search",
                func=serp_search.run,
                description="Useful for when you need to answer questions about current events or data not in the documents."
            )
            print("✅ SerpAPI web search tool initialized")
            return web_search_tool
        except Exception as e:
            print(f"⚠️ Error initializing SerpAPI: {e}")
            return None
    else:
        print("⚠️ SERPAPI_API_KEY not found - web search will be disabled")
        return None


def initialize_llamacloud_index():
    
    global llama_index, retriever
    
    try:
        llama_index = LlamaCloudIndex(
            name=LLAMA_CLOUD_INDEX_NAME,
            project_name=LLAMA_CLOUD_PROJECT_NAME,
            organization_id=LLAMA_CLOUD_ORGANIZATION_ID,
            api_key=LLAMA_CLOUD_API_KEY,
        )
        
        retriever = llama_index.as_retriever(similarity_top_k=5)
        
        print(f"✅ Connected to LlamaCloud index: {LLAMA_CLOUD_INDEX_NAME}")
        return llama_index
    except Exception as e:
        print(f"⚠️ Error connecting to LlamaCloud: {e}")
        return None


def invoke_with_retries(llm_inst, msgs, max_retries=5, base_delay=1.0, max_delay=60.0):
    attempt = 0
    while True:
        try:
            return llm_inst.invoke(msgs)
        except Exception as e:
            attempt += 1
            is_rate = False
            if GroqRateLimitError is not None and isinstance(e, GroqRateLimitError):
                is_rate = True
            else:
                emsg = str(e).lower()
                if "rate limit" in emsg or "rate_limit" in emsg or "429" in emsg:
                    is_rate = True

            if not is_rate or attempt > max_retries:
                raise

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)) * (1 + random.random()))
            print(f"⏳ Rate limit hit (attempt {attempt}/{max_retries}). Sleeping {delay:.1f}s...")
            time.sleep(delay)


def create_agent_graph(enable_search: bool = False, conversation_history: List[dict] = None, temperature: float = 0.0):
    
    if conversation_history is None:
        conversation_history = []
    
    llm_instance = initialize_llm(temperature)
    
    agent_steps = []
    
    def retrieve_node(state: AgentState):
        print("--- 1. RETRIEVING FROM LLAMA CLOUD ---")
        query = state["messages"][-1].content
        nodes = retriever.retrieve(query)
        context_text = "\n\n".join([n.get_content() for n in nodes])
        
        step_info = {
            "type": "retrieve",
            "title": "Retrieving from LlamaCloud",
            "content": f"Searching vector database for relevant documents...\nQuery: {query}\nFound {len(nodes)} relevant chunks",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "num_chunks": len(nodes),
                "query": query
            }
        }
        agent_steps.append(step_info)
        
        return {"context": context_text}
    
    def agent_node(state: AgentState):
        print("--- 2. AGENT (GROQ) DECIDING ---")
        messages = state["messages"]
        context = state["context"]
        conv_history = state.get("conversation_history", [])
        
        
        history_text = ""
        if conv_history:
            history_text = "\n\nPREVIOUS CONVERSATION:\n"
            for msg in conv_history[-5:]:  # Last 5 messages
                history_text += f"User: {msg['question']}\nAssistant: {msg['answer']}\n\n"
        
        
        system_msg = SystemMessage(content=f"""
You are a helpful assistant with access to a document knowledge base.

{history_text}

RETRIEVED CONTEXT FROM DOCUMENTS:
{context}

INSTRUCTIONS:
1. FIRST, check if the retrieved context answers the user's question.
2. If the context provides a good answer, use it to respond.
3. If the context is empty or doesn't answer the question:
   {"- You MUST use the 'web_search' tool to find current information." if enable_search else "- Politely inform the user that you don't have that information in the documents."}
4. Always cite sources when using document context.
5. Be conversational and remember the context from previous messages.
""")
        
        
        step_info = {
            "type": "agent",
            "title": "Agent Reasoning",
            "content": f"Analyzing context and deciding on response strategy...\nTemperature: {temperature}\nWeb Search: {'Enabled' if enable_search else 'Disabled'}\nContext length: {len(context)} characters",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "temperature": temperature,
                "enable_search": enable_search,
                "context_length": len(context)
            }
        }
        agent_steps.append(step_info)
        
        
        response = invoke_with_retries(llm_with_tools, [system_msg] + messages)
        
        return {"messages": [response]}
    
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("agent", agent_node)
    
    if enable_search and web_search_tool:
        tools = [web_search_tool]
        llm_with_tools = llm_instance.bind_tools(tools)
        
        
        original_tool_node = ToolNode(tools)
        
        def tracked_tool_node(state):
            print("--- 3. EXECUTING TOOL ---")
            result = original_tool_node(state)
            
            
            step_info = {
                "type": "tool",
                "title": "Web Search Executed",
                "content": "Searching the web for current information via SerpAPI...",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "tool": "web_search"
                }
            }
            agent_steps.append(step_info)
            
            return result
        
        workflow.add_node("tools", tracked_tool_node)
    else:
        llm_with_tools = llm_instance
    
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "agent")
    
    if enable_search and web_search_tool:
        workflow.add_conditional_edges(
            "agent",
            tools_condition
        )
        workflow.add_edge("tools", "agent")
    
    compiled_graph = workflow.compile()
    
    
    return compiled_graph, agent_steps


def get_or_create_agent(conversation_id: str, enable_search: bool = False, conversation_history: List[dict] = None, temperature: float = 0.0):
    
    key = f"{conversation_id}_{enable_search}_{temperature}"
    
    if key not in agent_apps:
        agent_graph, steps_tracker = create_agent_graph(enable_search, conversation_history, temperature)
        agent_apps[key] = {
            "graph": agent_graph,
            "enable_search": enable_search,
            "history": conversation_history or [],
            "temperature": temperature,
            "steps": steps_tracker
        }
        print(f"✅ Created new agent for conversation: {conversation_id} (search: {enable_search}, temp: {temperature})")
    
    return agent_apps[key]


def get_file_type(filename: str) -> tuple:
    
    ext = os.path.splitext(filename)[1].lower()
    mime_type = SUPPORTED_FILE_TYPES.get(ext)
    
    if mime_type:
        return ext, mime_type
    
    mime_type, _ = mimetypes.guess_type(filename)
    return ext, mime_type or "application/octet-stream"


def validate_file(filename: str) -> bool:
    
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_FILE_TYPES


def upload_to_llamacloud(file_path: str, file_id: str, filename: str, user_id: str):
    
    global llama_index
    
    if llama_index is None:
        raise Exception("LlamaCloud index not initialized")
    
    try:
        ext, mime_type = get_file_type(filename)
        print(f"📤 Uploading {filename} ({mime_type}) to LlamaCloud...")
        
        result = llama_index.upload_file(
            file_path=file_path,
            verbose=True
        )
        
        print(f"✅ File indexed in LlamaCloud: {filename}")
        return result
        
    except Exception as e:
        print(f"❌ Error uploading to LlamaCloud: {e}")
        raise


def initialize_firebase():
    
    global db_firestore, bucket
    
    if not os.getenv("FIREBASE_PRIVATE_KEY"):
        print("⚠️ Firebase credentials not found - running in local mode")
        return False
    
    try:
        if not firebase_admin._apps:
            firebase_json_path = "firebase-credentials.json"
            
            if os.path.exists(firebase_json_path):
                cred = credentials.Certificate(firebase_json_path)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': FIREBASE_STORAGE_BUCKET
                })
                print("✅ Firebase initialized with JSON file")
            else:
                cred = credentials.Certificate(FIREBASE_CONFIG)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': FIREBASE_STORAGE_BUCKET
                })
                print("✅ Firebase initialized with environment variables")
        
        db_firestore = firestore.client()
        bucket = storage.bucket()
        print("✅ Firestore and Firebase Storage ready")
        return True
    except Exception as e:
        print(f"⚠️ Firebase initialization failed: {e}")
        db_firestore = None
        bucket = None
        return False


def upload_file_to_firebase_storage(file_path: str, user_id: str, filename: str):
    
    if bucket is None:
        return None
    
    try:
        file_id = str(uuid.uuid4())
        blob_name = f"user_uploads/{user_id}/{file_id}/{filename}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        url = blob.generate_signed_url(expiration=timedelta(days=7))
        
        print(f"✅ Backed up to Firebase Storage: {blob_name}")
        return {"file_id": file_id, "blob_name": blob_name, "url": url}
    except Exception as e:
        print(f"❌ Error uploading to Firebase Storage: {e}")
        return None


def save_conversation_to_firestore(
    conversation_id: str,
    user_id: str,
    question: str,
    answer: str,
    sources: List[dict] = None,
    used_web_search: bool = False
):
    
    if db_firestore is None:
        return None

    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        conv_ref = db_firestore.collection('conversations').document(conversation_id)
        conv_doc = conv_ref.get()

        message_data = {
            'question': question,
            'answer': answer,
            'timestamp': timestamp,
            'sources': sources or [],
            'used_web_search': used_web_search
        }

        if conv_doc.exists:
            conv_ref.update({
                'messages': firestore.ArrayUnion([message_data]),
                'updated_at': timestamp
            })
        else:
            conv_ref.set({
                'conversation_id': conversation_id,
                'user_id': user_id,
                'messages': [message_data],
                'created_at': timestamp,
                'updated_at': timestamp
            })

        return conversation_id
    except Exception as e:
        print(f"❌ Error saving conversation: {e}")
        return None


def get_conversation_history(conversation_id: str) -> List[dict]:
    
    if db_firestore is None:
        return []
    
    try:
        conv_ref = db_firestore.collection('conversations').document(conversation_id)
        conv_doc = conv_ref.get()
        
        if conv_doc.exists:
            data = conv_doc.to_dict()
            return data.get('messages', [])
        return []
    except Exception as e:
        print(f"❌ Error fetching conversation history: {e}")
        return []


@app.on_event("startup")
async def startup_event():
    
    print("🚀 Starting Multi-Format RAG API with ReAct Agent...")
    
    initialize_llm()
    initialize_firebase()
    initialize_llamacloud_index()
    initialize_web_search_tool()
    
    if llm and retriever:
        print("✅ Ready to create ReAct agents!")
    else:
        print("⚠️ Missing LLM or retriever")
    
    print("✅ Startup complete!")


@app.get("/")
async def root():
    
    return {
        "message": "Multi-Format RAG API with ReAct Agent",
        "version": "3.0.0",
        "backend": "LlamaCloud + Groq + LangGraph",
        "features": ["RAG", "ReAct Agent", "Optional Web Search", "Multi-turn Conversations"],
        "supported_formats": list(SUPPORTED_FILE_TYPES.keys()),
        "endpoints": {
            "POST /upload": "Upload any supported document type",
            "POST /query": "Query with ReAct agent (enable_search parameter)",
            "GET /health": "Health check",
            "GET /supported-types": "List all supported file types",
            "GET /conversations/user/{user_id}": "Get user conversations",
            "DELETE /conversations/{conversation_id}": "Delete conversation",
        }
    }


@app.get("/supported-types")
async def get_supported_types():
    
    return {
        "supported_types": SUPPORTED_FILE_TYPES,
        "total_types": len(SUPPORTED_FILE_TYPES),
        "categories": {
            "documents": [".pdf", ".docx", ".doc", ".txt", ".md", ".rtf"],
            "spreadsheets": [".xlsx", ".xls", ".csv"],
            "presentations": [".pptx", ".ppt"],
            "images": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"],
            "code": [".html", ".xml", ".json", ".py", ".js", ".java", ".cpp", ".c"]
        }
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), user_id: str = "anonymous"):
    
    
    if not validate_file(file.filename):
        ext = os.path.splitext(file.filename)[1]
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Supported types: {list(SUPPORTED_FILE_TYPES.keys())}"
        )
    
    file_path = None
    try:
        file_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        ext, mime_type = get_file_type(file.filename)
        
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(file_path)
        print(f"📥 Uploaded: {file.filename} ({ext}, {file_size} bytes)")
        
        firebase_url = None
        if bucket:
            firebase_data = upload_file_to_firebase_storage(file_path, user_id, file.filename)
            firebase_url = firebase_data['url'] if firebase_data else None
        
        llama_result = upload_to_llamacloud(file_path, file_id, file.filename, user_id)
        
        if db_firestore:
            file_metadata = {
                'file_id': file_id,
                'filename': file.filename,
                'file_type': ext,
                'mime_type': mime_type,
                'user_id': user_id,
                'firebase_url': firebase_url,
                'file_size': file_size,
                'upload_timestamp': timestamp,
                'status': 'indexed',
                'llama_cloud': True
            }
            db_firestore.collection('uploaded_files').document(file_id).set(file_metadata)
        
        os.remove(file_path)
        
        return UploadResponse(
            message="Document uploaded and indexed successfully",
            file_id=file_id,
            filename=file.filename,
            file_type=ext,
            file_size=file_size,
            user_id=user_id,
            firebase_url=firebase_url,
            timestamp=timestamp,
            status="success"
        )
    
    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    
    
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please check LlamaCloud and Groq API configuration."
        )
    
    
    if not (0.0 <= request.temperature <= 2.0):
        raise HTTPException(
            status_code=400,
            detail="Temperature must be between 0.0 and 2.0"
        )
    
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        
        conversation_history = get_conversation_history(conversation_id)
        
        
        agent_data = get_or_create_agent(
            conversation_id, 
            request.enable_search,
            conversation_history,
            request.temperature
        )
        agent_graph = agent_data["graph"]
        agent_steps = agent_data["steps"]
        
        
        agent_steps.clear()
        
        print(f"💬 Processing query with ReAct agent (search: {request.enable_search}, temp: {request.temperature}): {request.question}")
        
        
        inputs = {
            "messages": [HumanMessage(content=request.question)],
            "context": "",
            "conversation_history": conversation_history
        }
        
        result = None
        used_web_search = False
        
        for event in agent_graph.stream(inputs):
            for key, value in event.items():
                if key == "agent":
                    result = value
                elif key == "tools":
                    used_web_search = True
                    print(f"🌍 Web search executed")
        
        
        if result and "messages" in result:
            answer = result["messages"][-1].content
        else:
            answer = "I apologize, but I couldn't generate a response."
        
        
        sources = []
        nodes = retriever.retrieve(request.question)
        if nodes:
            sources = [
                {
                    "text": node.get_content()[:300] + "...",
                    "score": node.score if hasattr(node, 'score') else None,
                    "metadata": node.metadata if hasattr(node, 'metadata') else {}
                }
                for node in nodes[:request.top_k]
            ]
        
        
        agent_data["history"].append({
            "question": request.question,
            "answer": answer
        })
        
        
        save_conversation_to_firestore(
            conversation_id=conversation_id,
            user_id=request.user_id,
            question=request.question,
            answer=answer,
            sources=sources,
            used_web_search=used_web_search
        )
        
        return QueryResponse(
            answer=answer,
            conversation_id=conversation_id,
            sources=sources,
            timestamp=timestamp,
            used_web_search=used_web_search,
            agent_steps=agent_steps
        )
    
    except Exception as e:
        print(f"❌ Error in query: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")


@app.get("/health")
async def health_check():
    
    llm_info = "Not initialized"
    if llm:
        llm_info = f"{llm.__class__.__name__} - {getattr(llm, 'model', 'unknown model')}"
    
    return {
        "status": "healthy",
        "llamacloud_connected": llama_index is not None,
        "retriever_ready": retriever is not None,
        "llm_ready": llm is not None,
        "web_search_ready": web_search_tool is not None,
        "firebase_connected": db_firestore is not None,
        "active_agents": len(agent_apps),
        "backend": {
            "vector_store": "LlamaCloud",
            "llm": llm_info,
            "provider": "Groq",
            "agent": "LangGraph ReAct Agent",
            "web_search": "SerpAPI (optional)"
        },
        "supported_file_types": len(SUPPORTED_FILE_TYPES)
    }


@app.get("/conversations/user/{user_id}")
async def get_user_conversations(user_id: str, limit: int = 10):
    
    if db_firestore is None:
        return {
            "user_id": user_id,
            "total": 0,
            "conversations": [],
            "note": "Firebase not initialized"
        }
    
    try:
        conversations = db_firestore.collection('conversations')\
            .where('user_id', '==', user_id)\
            .limit(limit)\
            .stream()
        
        conv_list = [conv.to_dict() for conv in conversations]
        conv_list.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        
        return {
            "user_id": user_id,
            "total": len(conv_list),
            "conversations": conv_list[:limit]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    
    keys_to_delete = [k for k in agent_apps.keys() if k.startswith(conversation_id)]
    for key in keys_to_delete:
        del agent_apps[key]
        print(f"🗑️ Cleared agent: {key}")
    
    if db_firestore:
        try:
            db_firestore.collection('conversations').document(conversation_id).delete()
            print(f"🗑️ Deleted Firestore record: {conversation_id}")
        except Exception as e:
            print(f"⚠️ Error deleting from Firestore: {e}")
    
    return {
        "message": f"Conversation {conversation_id} deleted successfully",
        "agents_cleared": len(keys_to_delete)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)