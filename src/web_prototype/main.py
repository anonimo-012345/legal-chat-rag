from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

import auth
import rag_pipeline
from database import init_db, get_db, User, ChatMessage
from schemas import UserCreate, UserLogin, ChatRequest, ChatResponse

app = FastAPI(title="Assistente Jurídico RAG")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    user = auth.get_current_user_optional(request, next(get_db()))
    if user:
        return RedirectResponse(url="/chat")
    return RedirectResponse(url="/login")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse(request, "register.html")


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request, db: Session = Depends(get_db)):
    try:
        user = auth.get_current_user(request, db)
    except HTTPException:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse(request, "chat.html", {"username": user.username})

# AUTENTICAÇÃO
@app.post("/api/register")
def register(data: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Nome de usuário já existe")

    user = User(username=data.username, hashed_password=auth.hash_password(data.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = auth.create_access_token({"sub": user.username})
    response = JSONResponse({"message": "Cadastro realizado com sucesso"})
    response.set_cookie(
        key="access_token", value=token, httponly=True, max_age=60 * 60 * 24 * 7, samesite="lax"
    )
    return response


@app.post("/api/login")
def login(data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()
    if not user or not auth.verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Usuário ou senha incorretos")

    token = auth.create_access_token({"sub": user.username})
    response = JSONResponse({"message": "Login realizado com sucesso"})
    response.set_cookie(
        key="access_token", value=token, httponly=True, max_age=60 * 60 * 24 * 7, samesite="lax"
    )
    return response


@app.post("/api/logout")
def logout():
    response = JSONResponse({"message": "Logout realizado"})
    response.delete_cookie("access_token")
    return response

@app.get("/api/chat/history")
def get_history(current_user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == current_user.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in messages]


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    data: ChatRequest,
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    # 1. Carrega o histórico anterior deste usuário do banco
    previous_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == current_user.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    chat_history = [(m.role, m.content) for m in previous_messages]

    # 2. Chama o pipeline RAG (reescrita -> retrieval -> geração)
    try:
        result = rag_pipeline.ask_question(data.question, chat_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a pergunta: {e}")

    # 3. Persiste a pergunta e a resposta no histórico do usuário
    db.add(ChatMessage(user_id=current_user.id, role="human", content=data.question))
    db.add(ChatMessage(user_id=current_user.id, role="ai", content=result["answer"]))
    db.commit()

    return ChatResponse(answer=result["answer"], search_question=result["search_question"])


@app.delete("/api/chat/history")
def clear_history(current_user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    db.query(ChatMessage).filter(ChatMessage.user_id == current_user.id).delete()
    db.commit()
    return {"message": "Histórico apagado"}


@app.post("/api/ingest")
def ingest(current_user: User = Depends(auth.get_current_user)):
    try:
        n_chunks = rag_pipeline.run_ingestion()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na ingestão: {e}")
    return {"message": "Ingestão concluída", "chunks_criados": n_chunks}
