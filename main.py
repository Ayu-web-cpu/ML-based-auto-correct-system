

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
import joblib, numpy as np, torch
from rapidfuzz import process, fuzz, distance
from rapidfuzz.distance import Levenshtein
from wordfreq import word_frequency, top_n_list
from transformers import T5Tokenizer, T5ForConditionalGeneration
from spellchecker import SpellChecker

# Database Config

DATABASE_URL = "postgresql://postgres:ayush@localhost:5432/autocorrect_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


# Models

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    password = Column(String)
    corrections = relationship("CorrectionHistory", back_populates="user")

class CorrectionHistory(Base):
    __tablename__ = "corrections"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    input_text = Column(String)
    spelling_fixed = Column(String)
    final_corrected = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="corrections")

Base.metadata.create_all(bind=engine)


# Auth Setup

SECRET_KEY = "super_secret_key_change_this"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta=None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token. Please log in again."
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# ML Model Loading

spell_model = joblib.load("spell_model_xgb.pkl")
vocab = joblib.load("vocab.pkl")
vectorizer = joblib.load("tfidf.pkl")
spell = SpellChecker()
common_vocab = set(top_n_list("en", 5000))
vocab = list(set(vocab) & common_vocab)
tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")
grammar_model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")

# Correction Logic

def features(wrong, candidate, vectorizer):
    prefix = len([1 for a, b in zip(wrong, candidate) if a == b])
    suffix = len([1 for a, b in zip(wrong[::-1], candidate[::-1]) if a == b])
    v1 = vectorizer.transform([wrong])
    v2 = vectorizer.transform([candidate])
    tfidf_sim = float((v1 @ v2.T).toarray()[0][0])
    set1, set2 = set(wrong), set(candidate)
    jaccard_sim = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
    return [
        fuzz.ratio(wrong, candidate) / 100,
        distance.Levenshtein.distance(wrong, candidate),
        abs(len(wrong) - len(candidate)),
        prefix, suffix,
        word_frequency(candidate, "en"),
        tfidf_sim, jaccard_sim
    ]

def autocorrect_word(word):
    if word not in spell.unknown([word]):
        return word
    candidates = [c for c, _, _ in process.extract(word, vocab, limit=10)]
    candidates = [c for c in candidates if distance.Levenshtein.distance(word, c) <= 3]
    if not candidates:
        return word
    X_test = [features(word, c, vectorizer) for c in candidates]
    probs = spell_model.predict_proba(np.array(X_test))[:, 1]
    best = candidates[probs.argmax()]
    if probs.max() < 0.3:
        return word
    return best

def autocorrect_sentence_spelling(sentence):
    return " ".join(autocorrect_word(w) for w in sentence.split())

def correct_grammar(text: str) -> str:
    input_text = "gec: " + text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = grammar_model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# FastAPI App Setup

app = FastAPI(title="Hybrid Autocorrect API with Auth & Postgres")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# Schemas

class Register(BaseModel):
    username: str
    password: str

class CorrectionInput(BaseModel):
    text: str


# Routes

@app.post("/register", tags=["Auth"])
def register_user(user: Register, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    new_user = User(username=user.username, password=get_password_hash(user.password))
    db.add(new_user)
    db.commit()
    return {"message": "User registered successfully"}

@app.post("/login", tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}


# Protected Routes

@app.post("/correct", tags=["Correction"])
def correct_text(data: CorrectionInput, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    spelling_fixed = autocorrect_sentence_spelling(data.text)
    final_corrected = correct_grammar(spelling_fixed)
    record = CorrectionHistory(
        user_id=current_user.id,
        input_text=data.text,
        spelling_fixed=spelling_fixed,
        final_corrected=final_corrected
    )
    db.add(record)
    db.commit()
    return {
        "input": data.text,
        "spelling_fixed": spelling_fixed,
        "final_corrected": final_corrected
    }

@app.get("/history")
def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    history = (
        db.query(CorrectionHistory)
        .filter(CorrectionHistory.user_id == current_user.id)
        .order_by(CorrectionHistory.timestamp.desc())
        .all()
    )
    return [
        {
            "id": h.id,
            "input_text": h.input_text,
            "spelling_fixed": h.spelling_fixed,
            "final_corrected": h.final_corrected,
            "timestamp": h.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        for h in history
    ]
