"""Espírito Santo AI - Backend API
CRM + Chatbot IA + Voz (Whisper STT + OpenAI TTS) + WhatsApp via Baileys (QR Code) auto-hospedado.
Inclui também Z-API / Evolution / Meta Cloud como provedores opcionais.
"""
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Literal, Any, Dict
import uuid
from datetime import datetime, timezone, timedelta
import bcrypt
import jwt as pyjwt

from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
from emergentintegrations.llm.openai import OpenAISpeechToText, OpenAITextToSpeech
from whatsapp_providers import (
    build_provider_from_config,
    default_zapi_from_env,
    normalize_phone,
    ZAPIProvider,
    BaileysProvider,
)
import httpx
import base64
import asyncio
import io

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

JWT_SECRET = os.environ.get('JWT_SECRET', 'legalflow-secret')
JWT_ALG = "HS256"
JWT_EXP_HOURS = 24 * 7
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')

app = FastAPI(title="Espírito Santo AI")
api_router = APIRouter(prefix="/api")
security = HTTPBearer(auto_error=False)
log = logging.getLogger("espirito-santo")

# ==================== UTILS ====================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False

def create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXP_HOURS),
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

async def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not creds:
        raise HTTPException(401, "Token não fornecido")
    try:
        payload = pyjwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALG])
        user = await db.users.find_one({"id": payload["sub"]}, {"_id": 0, "password": 0})
        if not user:
            raise HTTPException(401, "Usuário não encontrado")
        return user
    except pyjwt.PyJWTError:
        raise HTTPException(401, "Token inválido")

# ==================== MODELS ====================

class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    oab: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class LeadCreate(BaseModel):
    name: str
    phone: str
    email: Optional[str] = None
    case_type: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = "site"

class LeadUpdate(BaseModel):
    stage: Optional[str] = None
    score: Optional[int] = None
    notes: Optional[str] = None
    case_type: Optional[str] = None

class ChatMessageIn(BaseModel):
    session_id: Optional[str] = None
    message: str
    visitor_name: Optional[str] = None
    visitor_phone: Optional[str] = None
    voice: Optional[str] = "nova"  # OpenAI TTS voice
    want_audio: Optional[bool] = True
    return_analysis: Optional[bool] = True

class WhatsAppMessageIn(BaseModel):
    contact_id: str
    text: str

class WhatsAppSendDirect(BaseModel):
    phone: str
    text: str

class ProcessCreate(BaseModel):
    client_name: str
    client_id: Optional[str] = None
    process_number: str
    case_type: str
    court: Optional[str] = None
    status: str = "Em Andamento"
    description: Optional[str] = None
    next_hearing: Optional[str] = None

class TransactionCreate(BaseModel):
    client_name: str
    description: str
    amount: float
    type: Literal["receita", "despesa"]
    status: Literal["pago", "pendente", "atrasado"] = "pendente"
    due_date: Optional[str] = None
    process_id: Optional[str] = None

class CreativeGenerate(BaseModel):
    title: str
    network: Literal["instagram", "facebook", "linkedin"]
    format: Literal["post", "story", "carousel"] = "post"
    topic: str
    tone: Literal["profissional", "informativo", "amigavel", "urgente"] = "profissional"
    case_type: Optional[str] = None

class CaptionRequest(BaseModel):
    topic: str
    network: str
    tone: str = "profissional"

class WhatsAppConfig(BaseModel):
    provider: Literal["zapi", "evolution", "meta", "baileys"] = "zapi"
    zapi_instance_id: Optional[str] = None
    zapi_instance_token: Optional[str] = None
    zapi_client_token: Optional[str] = None
    evo_base_url: Optional[str] = None
    evo_api_key: Optional[str] = None
    evo_instance: Optional[str] = None
    meta_access_token: Optional[str] = None
    meta_phone_number_id: Optional[str] = None
    bot_enabled: bool = False
    bot_prompt: Optional[str] = None
    # Modo de voz do bot na resposta WhatsApp
    bot_voice_mode: Literal["text_only", "text_and_audio", "audio_only", "auto"] = "text_and_audio"
    # Provedor LLM para gerar respostas do bot:
    #   "emergent" -> Emergent LLM Key (gpt-5.2 / gpt-4o-mini) [padrao, melhor qualidade]
    #   "ollama"   -> Ollama local na maquina do usuario via tunnel (gratis)
    llm_provider: Literal["emergent", "ollama"] = "emergent"
    ollama_base_url: Optional[str] = None  # ex: https://meu-tunnel.trycloudflare.com
    ollama_model: Optional[str] = None     # ex: llama3.2:3b, qwen2.5:7b, mistral:7b
    # Provedor de TTS
    voice_provider: Literal["openai", "elevenlabs"] = "openai"
    bot_voice: str = "nova"
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: Optional[str] = None
    elevenlabs_voice_name: Optional[str] = None

class DebugInstruction(BaseModel):
    instruction: str
    context: Optional[str] = None

# ==================== AUTH ROUTES ====================

@api_router.post("/auth/register")
async def register(payload: UserRegister):
    existing = await db.users.find_one({"email": payload.email})
    if existing:
        raise HTTPException(400, "E-mail já cadastrado")
    user_id = str(uuid.uuid4())
    user_doc = {
        "id": user_id,
        "name": payload.name,
        "email": payload.email,
        "password": hash_password(payload.password),
        "oab": payload.oab,
        "created_at": now_iso(),
    }
    await db.users.insert_one(user_doc)
    token = create_token(user_id, payload.email)
    return {"token": token, "user": {"id": user_id, "name": payload.name, "email": payload.email, "oab": payload.oab}}

@api_router.post("/auth/login")
async def login(payload: UserLogin):
    user = await db.users.find_one({"email": payload.email})
    if not user:
        # auto-create demo account for convenience
        demo_accounts = {
            "demo@espirito-santo.com.br": ("Dr. Demo", "000000/SP"),
            "demo@legalflow.ai": ("Dr. Demo", "000000/SP"),
            "demo@kenia-garcia.com.br": ("Dr. Demo", "000000/SP"),
        }
        if payload.email in demo_accounts and payload.password == "demo123":
            name, oab = demo_accounts[payload.email]
            uid = str(uuid.uuid4())
            await db.users.insert_one({
                "id": uid, "name": name, "email": payload.email,
                "password": hash_password(payload.password), "oab": oab,
                "created_at": now_iso(),
            })
            token = create_token(uid, payload.email)
            return {"token": token, "user": {"id": uid, "name": name, "email": payload.email, "oab": oab}}
        raise HTTPException(401, "Credenciais inválidas")
    if not verify_password(payload.password, user["password"]):
        raise HTTPException(401, "Credenciais inválidas")
    token = create_token(user["id"], user["email"])
    return {"token": token, "user": {"id": user["id"], "name": user["name"], "email": user["email"], "oab": user.get("oab")}}

@api_router.get("/auth/me")
async def me(current_user=Depends(get_current_user)):
    return current_user

# ==================== LEADS / CRM ====================

CRM_STAGES = ["novos_leads", "em_contato", "interessado", "qualificado", "em_negociacao", "convertido", "nao_interessado"]

# Mapeamento amigável para exibição
STAGE_META = {
    "novos_leads": {"label": "Novos Leads", "color": "blue"},
    "em_contato": {"label": "Em Contato", "color": "yellow"},
    "interessado": {"label": "Interessado", "color": "green"},
    "qualificado": {"label": "Qualificado", "color": "purple"},
    "em_negociacao": {"label": "Em Negociação", "color": "orange"},
    "convertido": {"label": "Convertido", "color": "emerald"},
    "nao_interessado": {"label": "Não Interessado", "color": "red"},
}

LEGAL_AREAS = [
    "Trabalhista", "Família", "Previdenciário/INSS", "Cível",
    "Criminal", "Empresarial", "Tributário", "Bancário", "Consumidor", "Outro",
]

@api_router.post("/leads")
async def create_lead(payload: LeadCreate, current_user=Depends(get_current_user)):
    lead_id = str(uuid.uuid4())
    score = 50
    if payload.case_type:
        urgent = ["criminal", "trabalhista", "inss"]
        if any(k in payload.case_type.lower() for k in urgent):
            score = 75
    doc = {
        "id": lead_id, "owner_id": current_user["id"],
        "name": payload.name, "phone": payload.phone, "email": payload.email,
        "case_type": payload.case_type, "description": payload.description,
        "source": payload.source, "stage": "novos_leads", "score": score, "notes": "",
        "urgency": "media", "tags": [],
        "created_at": now_iso(), "updated_at": now_iso(),
    }
    await db.leads.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api_router.get("/leads")
async def list_leads(current_user=Depends(get_current_user)):
    leads = await db.leads.find({"owner_id": current_user["id"]}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    return leads

@api_router.patch("/leads/{lead_id}")
async def update_lead(lead_id: str, payload: LeadUpdate, current_user=Depends(get_current_user)):
    update = {k: v for k, v in payload.model_dump().items() if v is not None}
    update["updated_at"] = now_iso()
    result = await db.leads.update_one({"id": lead_id, "owner_id": current_user["id"]}, {"$set": update})
    if result.matched_count == 0:
        raise HTTPException(404, "Lead não encontrado")
    lead = await db.leads.find_one({"id": lead_id}, {"_id": 0})
    return lead

@api_router.delete("/leads/{lead_id}")
async def delete_lead(lead_id: str, current_user=Depends(get_current_user)):
    await db.leads.delete_one({"id": lead_id, "owner_id": current_user["id"]})
    return {"ok": True}

@api_router.post("/public/leads")
async def public_lead(payload: LeadCreate):
    first = await db.users.find_one({}, {"_id": 0, "id": 1})
    owner_id = first["id"] if first else "unassigned"
    lead_id = str(uuid.uuid4())
    doc = {
        "id": lead_id, "owner_id": owner_id,
        "name": payload.name, "phone": payload.phone, "email": payload.email,
        "case_type": payload.case_type, "description": payload.description,
        "source": payload.source or "landing", "stage": "novos_leads", "score": 50, "notes": "",
        "urgency": "media", "tags": [],
        "created_at": now_iso(), "updated_at": now_iso(),
    }
    await db.leads.insert_one(doc)
    return {"ok": True, "id": lead_id}

@api_router.get("/crm/stages")
async def get_stages():
    return [{"id": s, **STAGE_META[s]} for s in CRM_STAGES]

# ==================== CHATBOT IA (copiloto) ====================

# ============================================================
# BASE DE CONHECIMENTO — PREVIDÊNCIA RURAL (atualizada jan/2026)
# ============================================================
# Fonte: EC 103/2019 + Lei 8.213/91 + Lei 13.846/2019 + STJ/STF + IN INSS
# Salário mínimo 2026: R$ 1.621,00
# ============================================================
RURAL_PREVIDENCIA_KB_2026 = """
📚 CONHECIMENTO JURÍDICO ATUAL — PREVIDÊNCIA RURAL (jan/2026)

REGRAS VIGENTES (sem alteração pela EC 103/2019 para segurado especial):

1) APOSENTADORIA POR IDADE RURAL — SEGURADO ESPECIAL (art. 48, §§1º e 2º, Lei 8.213/91)
   • Idade mínima: 60 anos (homem) | 55 anos (mulher). NÃO mudou com a Reforma da Previdência.
   • Carência: 180 meses (15 anos) de atividade rural comprovada — NÃO precisa ser contínua, mas com "continuidade lógica" da vida rural.
   • Valor: 1 salário mínimo (R$ 1.621,00 em 2026), salvo contribuições facultativas.
   • Deve estar exercendo atividade rural OU em período de manutenção de qualidade na DER (data de entrada do requerimento).

2) SEGURADO ESPECIAL — QUEM É (art. 11, VII, Lei 8.213/91)
   • Produtor rural pessoa física em regime de economia familiar (sem empregados permanentes).
   • Pescador artesanal.
   • Indígena reconhecido pela FUNAI.
   • Cônjuge/companheiro e filhos maiores de 16 anos que trabalham em regime de economia familiar.
   • Inclui arrendatário, parceiro, meeiro outorgado e comodatário (todos na modalidade familiar).

3) EMPREGADO RURAL E CONTRIBUINTE INDIVIDUAL RURAL — EQUIPARADO AO URBANO
   Após a EC 103/2019, segue regra geral: 65 anos (H) / 62 anos (M) com 15-20 anos de contribuição,
   OU regras de transição (pontos, idade progressiva, pedágio 50%/100%).

4) CONTRIBUIÇÃO DO SEGURADO ESPECIAL (art. 25, II, Lei 8.212/91)
   • NÃO é mensal obrigatória.
   • 1,2% (INSS) + 0,1% (RAT) + 0,2% (SENAR) = 1,5% ao SENAR incluso, total ~2,1% sobre o valor bruto da comercialização da produção.
   • Pode contribuir facultativamente como Contribuinte Individual (20% do salário de contribuição) para aumentar o valor do benefício.

5) COMPROVAÇÃO DA ATIVIDADE RURAL
   • Pode ser feita por início de prova material + testemunhas (Súmula 149 STJ).
   • Documentos típicos: certidão de casamento/nascimento declarando profissão rural,
     ficha escolar dos filhos, contratos de arrendamento/parceria, nota fiscal de produtor,
     cadastro INCRA/CAR/DAP, declaração sindical rural homologada, bloco de produtor,
     título de propriedade rural, extrato ITR.
   • CNIS (Cadastro Nacional de Informações Sociais): a Lei 13.846/2019 previu que a partir
     de 2023 a comprovação seria EXCLUSIVA via CNIS, mas essa obrigatoriedade AINDA NÃO
     entrou em vigor — depende do CNIS cobrir 50%+ dos segurados especiais, meta não atingida.
   • Autodeclaração homologada pelo sindicato ou por órgão público é válida.

6) APOSENTADORIA HÍBRIDA (art. 48, §3º, Lei 8.213/91)
   • Permite somar tempo RURAL (mesmo sem contribuição) + tempo URBANO para cumprir a carência.
   • Idade: 65 anos (H) / 60 anos (M).
   • STJ/STF firmou que vale mesmo quando o trabalho rural é ANTERIOR à Lei 8.213/91
     (Tema 1.007 STJ, julgado 2020; STF RE 1.281.381).
   • Valor: calculado pela média salarial (geralmente > 1 SM).

7) PENSÃO POR MORTE (art. 74-77 Lei 8.213/91)
   • Prazos pra requerer e receber DESDE O ÓBITO:
     - Dependentes em geral: 90 dias após o óbito
     - Filhos menores de 16 anos: 180 dias após o óbito
   • Após esses prazos, DIB (Data de Início do Benefício) = data do requerimento, NÃO do óbito.
   • Duração: varia conforme idade do cônjuge no óbito e tempo de união (art. 77, §2º, V):
     < 22 anos = 3 anos; 22-27 = 6; 28-30 = 10; 31-41 = 15; 42-44 = 20; ≥ 45 = vitalícia.
   • STJ decidiu que a CONCESSÃO DE PENSÃO POR MORTE NÃO RENOVA o prazo decadencial de
     revisão do benefício originário (tema do falecido) — se já passou 10 anos, dependente
     não pode revisar o cálculo original.

8) PRAZOS CRÍTICOS
   • DECADÊNCIA PARA REVISAR BENEFÍCIO (art. 103 Lei 8.213/91): 10 anos da 1ª prestação.
   • PRESCRIÇÃO QUINQUENAL: parcelas vencidas antes dos 5 anos do requerimento são prescritas.
   • STJ (Tema 1.018 - 2023): para menor impúbere, o prazo não corre enquanto for incapaz.

9) APOSENTADORIA POR INVALIDEZ RURAL (hoje: aposentadoria por incapacidade permanente)
   • Carência: 12 meses de atividade rural (exceto acidente/doença grave — carência zero).
   • Valor: 60% da média + 2% por ano que exceder 20 (H) / 15 (M).
   • Reforma 2019 mudou o cálculo: antes era 100% da média.

10) BPC/LOAS RURAL (não é previdência, é assistencial)
    • Para quem tem 65+ OU deficiente, com renda familiar per capita < 1/4 SM.
    • Valor: 1 SM. Não gera 13º nem pensão por morte.
    • Útil como alternativa quando o rural não consegue comprovar os 180 meses de atividade.

DICAS PRÁTICAS DE CAUSA BOA:
• Cliente rural com 55 (M) ou 60 (H) + 15 anos de roça comprovados → aposentadoria por idade rural tranquila.
• Cliente que trabalhou rural "antigo" (anos 70-80) + depois foi pra cidade → aposentadoria HÍBRIDA, valor bem maior.
• Pensão por morte negada por "perda qualidade segurado" → verificar se o falecido estava em "período de graça" (art. 15 Lei 8.213/91 — até 36 meses mantém qualidade).
• INSS negou por "ausência de CNIS" → ainda NÃO é obrigatório; ajuizar com prova material.

SINAIS DE ALERTA (não prometer resultado):
• Trabalho "boia-fria" esporádico → pode dificultar comprovação.
• Urbano que só cultivou horta doméstica → não é atividade rural para fins previdenciários.
• Declaração sindical sem homologação de servidor ou sem período específico → INSS recusa.
"""

LEGAL_SYSTEM_PROMPT_BASE = """Você é a ANA, SECRETÁRIA do escritório Kênia Garcia Advocacia.
Você atende DIRETAMENTE o cliente final pelo chat. Sua missão é ESCUTAR o caso
do cliente, ACOLHER, fazer as perguntas certas para entender bem a situação, e
encaminhar o caso para a Dra. Kênia Garcia (a advogada titular).

═══════════════════════════════════════════════════════════
🔒 IDENTIDADE — REGRAS QUE NUNCA QUEBRAM
═══════════════════════════════════════════════════════════
- Você é a **Dra. Kênia Garcia**, advogada titular do escritório, atendendo PESSOALMENTE.
- NUNCA se apresente como "Ana", "secretária", "assistente", "bot", "robô" ou "IA". Você é a Dra. Kênia em pessoa.
- Quando se apresentar, diga: "Oi! Aqui é a Kênia Garcia, advogada." (apenas no PRIMEIRO contato — depois não repete).
- Use sempre o FEMININO ao falar de si mesma: "sou advogada", "fui consultada", "estou avaliando", "vou olhar". Nunca masculino.
- Você É a especialista — pode dar orientação preliminar firme baseada nos fatos, mas para conclusão definitiva diga: "pela sua descrição vejo X, mas pra fechar o parecer preciso analisar [documento]".

═══════════════════════════════════════════════════════════
🎯 PERSONA & TOM (HUMANIZAÇÃO PROFUNDA)
═══════════════════════════════════════════════════════════
- Fale como uma pessoa REAL, calorosa, simpática, segura. Nunca técnica fria.
- Use o NOME do cliente (quando souber) e linguagem de gente: "imagina", "olha", "entendo perfeitamente", "isso é muito comum acontecer". Nada de juridiquês.
- Reconheça a EMOÇÃO antes da técnica: "que situação difícil, sinto muito mesmo que esteja passando por isso". Demonstre que escutou.
- Frases curtas e diretas (3-6 linhas por mensagem). Nada de parágrafo gigante.
- Use 1 emoji sutil quando couber (☕ 📌 ⚖️ 🙏 🤍) — humaniza sem infantilizar.
- Contrações naturais: "tá", "pra", "tô", "né" — mostra que é gente real, não formal.

═══════════════════════════════════════════════════════════
🚫 REGRA DE OURO — PROIBIDO RESPONDER COM "OU" (UMA COISA OU OUTRA)
═══════════════════════════════════════════════════════════
Se você ficar em dúvida entre 2 ou mais respostas (ex: "pode ser trabalhista OU cível", "talvez seja estável OU intermitente", "pode dar X OU Y"), VOCÊ NÃO RESPONDE COM O "OU". Em vez disso, FAÇA UMA PERGUNTA OBJETIVA que elimine a dúvida e te leve à resposta única. É exatamente o que uma advogada faz: investigar com perguntas precisas.

ERRADO: "Pode ser caso trabalhista ou previdenciario, depende da situacao."
CERTO:  "Pra eu te orientar do jeito certo, me responde so uma coisa: voce foi registrado em carteira no ultimo emprego, ou trabalhava como autonomo?"

ERRADO: "Voce pode ter direito a indenizacao ou apenas as verbas rescisorias."
CERTO:  "Pra eu separar isso direitinho, me confirma: a empresa te demitiu sem justa causa, ou pediu pra voce assinar acordo de demissao? Me conta exatamente como foi."

ERRADO: "Talvez voce consiga aposentadoria por idade ou por tempo de contribuicao."
CERTO:  "Pra eu olhar teu caso direitinho, me confirma 2 coisas: 1) sua idade hoje  2) ha quantos anos voce contribui pro INSS (mesmo que com lacunas)."

REGRAS DERIVADAS:
1. NUNCA use "ou" como conjunção entre 2 respostas técnicas vagas.
2. SEMPRE que sentir ambiguidade técnica, transforme a dúvida em PERGUNTA numerada (1... 2... 3...) pra o cliente responder.
3. Se faltar dado, FALE explicitamente: "preciso só de MAIS UMA informação pra ver isso direito".

═══════════════════════════════════════════════════════════
📋 ROTEIRO DE ATENDIMENTO (UMA PERGUNTA POR VEZ)
═══════════════════════════════════════════════════════════
PASSO 1 — Cumprimentar; se for o primeiro contato, se apresentar como Kênia Garcia,
           advogada; pedir o NOME do cliente.
PASSO 2 — Perguntar O QUE ACONTECEU em uma frase. PERGUNTA ABERTA, sem listar
           categorias jurídicas.
PASSO 3 — Identificar o caso com 1 ou 2 perguntas-chave PRECISAS (vide regra
           do "ou" acima — sempre converta dúvida em pergunta).
PASSO 4 — Coletar TIMING (quando aconteceu? prazo correndo?), CIDADE/ESTADO,
           e se já há advogado(a)/processo anterior.
PASSO 5 — Dar uma orientação inicial firme (prazos, direitos comuns) e PROPOR
           reunião de análise por Google Meet ("vou marcar com você agora").
PASSO 6 — Encerrar tranquilizando: "perfeito, marcado. Te chamo no horário pelo Meet".

═══════════════════════════════════════════════════════════
⚖️ CONHECIMENTO TÉCNICO (BASE 2026 — pra você não passar vergonha)
═══════════════════════════════════════════════════════════
- Código Civil, CLT, Lei 8.213/91, EC 103/2019, Lei 13.846/2019, Lei 14.973/24,
  Súmulas STJ/STF, IN INSS, e jurisprudência atual.
- Salário mínimo 2026: R$ 1.621,00.
- Você pode citar a base legal de forma SIMPLES quando ela ajudar o cliente
  (ex: "a CLT garante esse direito no artigo 477"), mas SEM dar parecer.
- Se o tema for previdência rural / INSS / aposentadoria / pensão por morte /
  segurado especial, use a base abaixo com PRIORIDADE.

"""

LEGAL_SYSTEM_PROMPT = LEGAL_SYSTEM_PROMPT_BASE + RURAL_PREVIDENCIA_KB_2026


# ============================================================
# DAILY LEGISLATION UPDATE (cache atualizado 1x por dia)
# ============================================================
async def get_daily_legislation_brief() -> str:
    """Retorna um resumo curto do que e relevante na legislacao brasileira
    HOJE. Cache MongoDB com TTL de 24h. Se vencido, gera com IA usando data atual.
    """
    today_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cached = await db.legislation_cache.find_one({"date": today_key}, {"_id": 0})
    if cached and cached.get("brief"):
        return cached["brief"]
    # Gera brief novo
    sys = (
        "Voce e um especialista em direito brasileiro. Liste em ate 6 linhas as "
        "ATUALIZACOES legais e sumulas mais relevantes em vigor HOJE para escritorios "
        "de advocacia (foco trabalhista, previdenciario INSS, civel, familia, consumidor). "
        "Inclua: salario minimo atual, ultima reforma relevante, sumula recente do STJ/STF "
        "que mudou pratica forense, e 1 dica pratica do dia. Linguagem simples e objetiva."
    )
    brief = ""
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY, session_id=f"legbrief-{today_key}",
            system_message=sys,
        ).with_model("openai", "gpt-5.2")
        brief = await chat.send_message(UserMessage(
            text=f"Hoje e {today_key}. Gere o brief diario de legislacao para advogados."
        ))
    except Exception:
        log.exception("legislation brief gen failed")
        brief = (
            f"📅 Atualizacao {today_key}: Salario minimo R$ 1.621,00. "
            "Reforma da Previdencia EC 103/2019 vigente. CLT atualizada. "
            "Lei 14.973/2024 (refis previdenciario) ainda em vigor. "
            "STJ Tema 1.018: prazo decadencial nao corre p/ menor incapaz. "
            "Dica do dia: confira CNIS antes de qualquer requerimento INSS."
        )
    await db.legislation_cache.update_one(
        {"date": today_key},
        {"$set": {"date": today_key, "brief": brief, "created_at": now_iso()}},
        upsert=True,
    )
    return brief

@api_router.post("/chat/message")
async def chat_message(payload: ChatMessageIn):
    """Chat humanizado com analise do caso + audio (TTS) + indice de acertividade.

    - Modelo: gpt-5.2 (OpenAI) via Emergent LLM Key.
    - Persona: Dra. Ana (advogada virtual), tom acolhedor, faz perguntas em vez
      de responder com 'ou' (uma coisa ou outra).
    - Atualizacao diaria de legislacao injetada no prompt.
    - Retorna: response (texto), audio_base64 (TTS mp3), analysis (acertividade
      0-100, qualificacao, area, proxima_pergunta, motivo).
    """
    session_id = payload.session_id or str(uuid.uuid4())
    msg = (payload.message or "").strip()
    if not msg:
        raise HTTPException(400, "Mensagem vazia")

    await db.chat_messages.insert_one({
        "id": str(uuid.uuid4()), "session_id": session_id,
        "role": "user", "content": msg,
        "visitor_name": payload.visitor_name,
        "visitor_phone": payload.visitor_phone,
        "created_at": now_iso(),
    })

    # Carrega historico (ultimas 16 msgs) para dar contexto
    history = await db.chat_messages.find(
        {"session_id": session_id}, {"_id": 0}
    ).sort("created_at", 1).to_list(60)

    # Brief diario de legislacao (cache MongoDB, TTL 24h)
    leg_brief = await get_daily_legislation_brief()
    today_h = datetime.now(timezone.utc).strftime("%d/%m/%Y")

    # Contexto: nome do visitante quando informado
    name_block = ""
    if payload.visitor_name:
        name_block = f"\n\nNome do cliente que esta conversando com voce: {payload.visitor_name}. Use o nome dele(a) com naturalidade."

    system_prompt = (
        LEGAL_SYSTEM_PROMPT
        + name_block
        + f"\n\n═══════════════════════════════════════════════════════════\n"
        + f"📅 ATUALIZACAO DIARIA DE LEGISLACAO ({today_h})\n"
        + f"═══════════════════════════════════════════════════════════\n"
        + leg_brief
        + "\n═══════════════════════════════════════════════════════════\n"
        + "Use estes dados como referencia atual quando relevante.\n"
    )

    response = ""
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY, session_id=session_id,
            system_message=system_prompt,
        ).with_model("openai", "gpt-5.2")
        response = await chat.send_message(UserMessage(text=msg))
    except Exception:
        log.exception("AI chat error (gpt-5.2) — tentando fallback gpt-4o")
        try:
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY, session_id=session_id + "-fallback",
                system_message=system_prompt,
            ).with_model("openai", "gpt-4o")
            response = await chat.send_message(UserMessage(text=msg))
        except Exception:
            log.exception("AI chat fallback failed")
            response = (
                "Desculpe, estou com dificuldade tecnica nesse exato momento. "
                "Pode repetir sua pergunta em alguns segundos? 🙏"
            )

    response = (response or "").strip()

    await db.chat_messages.insert_one({
        "id": str(uuid.uuid4()), "session_id": session_id,
        "role": "assistant", "content": response, "created_at": now_iso(),
    })

    # Pos-processamento: gera AUDIO via OpenAI TTS (voz nova/fable)
    audio_b64 = None
    if payload.want_audio:
        try:
            tts = OpenAITextToSpeech(api_key=EMERGENT_LLM_KEY)
            audio_bytes = await tts.generate_speech(
                text=response[:4000], model="tts-1",
                voice=(payload.voice or "nova"), response_format="mp3",
            )
            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        except Exception:
            log.exception("TTS gen failed")

    # Analise estruturada do caso (acertividade + qualificacao)
    analysis = None
    if payload.return_analysis:
        try:
            analysis = await _analyze_case_session(session_id, history + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": response},
            ], visitor_name=payload.visitor_name, visitor_phone=payload.visitor_phone)
        except Exception:
            log.exception("case analysis failed")

    return {
        "session_id": session_id,
        "response": response,
        "audio_base64": audio_b64,
        "audio_mime": "audio/mpeg" if audio_b64 else None,
        "analysis": analysis,
        "legislation_date": today_h,
    }


async def _analyze_case_session(
    session_id: str,
    messages: List[Dict[str, Any]],
    visitor_name: Optional[str] = None,
    visitor_phone: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Analisa toda a conversa e retorna indice de acertividade + qualificacao.

    Saida JSON:
      {
        "area": "Trabalhista|Familia|Previdenciario/INSS|Civel|Criminal|...|Outro",
        "resumo": "1-2 linhas",
        "acertividade": 0-100,        # quao certeira e a leitura do caso
        "chance_exito": 0-100,        # chance de ganhar o caso baseado em dados reais
        "qualificacao": "qualificado|nao_qualificado|necessita_mais_info",
        "motivo": "1 frase curta",
        "proxima_pergunta": "pergunta especifica que precisa ser respondida pra fechar a analise (ou null se ja temos tudo)",
        "fundamentos": ["lei X", "sumula Y", "precedente Z"]
      }
    """
    import json as _json
    convo_lines = []
    for m in messages[-30:]:
        who = "Cliente" if m.get("role") == "user" else "Dra. Ana"
        c = (m.get("content") or "").strip()
        if c:
            convo_lines.append(f"{who}: {c[:400]}")
    convo = "\n".join(convo_lines)

    sys_prompt = (
        "Voce e um analista juridico senior. Analise a conversa abaixo entre uma "
        "advogada virtual e um cliente, e gere uma avaliacao tecnica do caso baseada "
        "em DIREITO BRASILEIRO ATUAL (2026) e jurisprudencia real.\n\n"
        "REGRAS:\n"
        "- 'acertividade' = quao certeira e sua leitura do caso (faltam dados? esta vago? = baixo).\n"
        "- 'chance_exito' = chance REAL de ganhar baseado em jurisprudencia e dados informados.\n"
        "- 'qualificacao' = 'qualificado' apenas se chance_exito >= 60 E acertividade >= 65.\n"
        "  'nao_qualificado' se chance_exito < 30 OU caso fora da advocacia.\n"
        "  'necessita_mais_info' se ainda faltam dados criticos pra decidir.\n"
        "- 'proxima_pergunta': se acertividade < 75, retorne UMA pergunta objetiva que "
        "fecharia a analise. Se >= 75, retorne null.\n"
        "- 'fundamentos': de 1 a 4 referencias legais reais (artigo, sumula, lei).\n\n"
        "Responda APENAS com JSON valido, sem markdown."
    )

    raw = "{}"
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"analyze-{session_id}-{uuid.uuid4().hex[:6]}",
            system_message=sys_prompt,
        ).with_model("openai", "gpt-5.2")
        raw = await chat.send_message(UserMessage(text=f"Conversa:\n{convo}\n\nGere o JSON de analise."))
    except Exception:
        log.exception("analyze llm call failed")
        return None

    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        data = _json.loads(raw)
    except Exception:
        log.warning(f"analyze parse failed: {raw[:200]}")
        return None

    # sanitiza
    def _i(v, lo=0, hi=100, default=50):
        try:
            n = int(v)
            return max(lo, min(hi, n))
        except Exception:
            return default

    out = {
        "area": str(data.get("area") or "Outro"),
        "resumo": str(data.get("resumo") or "")[:400],
        "acertividade": _i(data.get("acertividade"), default=50),
        "chance_exito": _i(data.get("chance_exito"), default=50),
        "qualificacao": str(data.get("qualificacao") or "necessita_mais_info"),
        "motivo": str(data.get("motivo") or "")[:300],
        "proxima_pergunta": (str(data.get("proxima_pergunta")).strip() or None) if data.get("proxima_pergunta") else None,
        "fundamentos": [str(x)[:120] for x in (data.get("fundamentos") or [])][:5],
    }
    if out["qualificacao"] not in ("qualificado", "nao_qualificado", "necessita_mais_info"):
        out["qualificacao"] = "necessita_mais_info"

    # Persiste / atualiza analise no MongoDB (1 doc por session)
    doc = {
        "session_id": session_id,
        "visitor_name": visitor_name,
        "visitor_phone": visitor_phone,
        "updated_at": now_iso(),
        **out,
    }
    existing = await db.case_analyses.find_one({"session_id": session_id}, {"_id": 0, "id": 1})
    if existing:
        await db.case_analyses.update_one(
            {"session_id": session_id}, {"$set": doc},
        )
        out["id"] = existing.get("id")
    else:
        cid = str(uuid.uuid4())
        doc["id"] = cid
        doc["created_at"] = now_iso()
        await db.case_analyses.insert_one(doc)
        out["id"] = cid

    return out

@api_router.get("/chat/history/{session_id}")
async def chat_history(session_id: str):
    msgs = await db.chat_messages.find({"session_id": session_id}, {"_id": 0}).sort("created_at", 1).to_list(500)
    return msgs

# ==================== WHATSAPP CONFIG ====================

KENIA_DEFAULT_PROMPT = """Você é a **Dra. Kênia Garcia** — advogada titular do escritório Kênia Garcia Advocacia. Você está atendendo PESSOALMENTE pelo WhatsApp. Você NÃO é secretária, NÃO é assistente, NÃO é bot. Você é a advogada, pessoa real, calorosa, experiente, ASSERTIVA e ESPECIALISTA nas áreas que atende.

REGRAS FUNDAMENTAIS (NÃO QUEBRAR):
- 🔴 **NA PRIMEIRA MENSAGEM com cada cliente novo, OBRIGATORIAMENTE se apresente como "Kênia Garcia".** Exemplos válidos: "Oi, [nome]! Aqui é a Kênia Garcia.", "Aqui é a Kênia, advogada — pode falar comigo direto.", "Sou a Dra. Kênia Garcia, prazer." Não importa a área do caso, SEMPRE diga seu nome na 1ª resposta.
- Quando perguntarem com quem está falando, responda: "Aqui é a Kênia Garcia, advogada — pode falar comigo direto." NUNCA diga "vou passar pra Dra.", "vou alinhar com ela", "ela vai te atender" — VOCÊ É ELA.
- JAMAIS diga que é robô, IA, bot, assistente, secretária, ou sistema. Você é a Kênia Garcia, advogada real falando pelo WhatsApp.
- Você é MULHER. Use sempre o feminino: "fui consultada", "estou avaliando", "sou advogada", "estou aqui". NUNCA o masculino.
- Sobre mérito: você É a especialista — pode dar orientação preliminar baseada nos fatos, mas SEMPRE diga que para conclusão definitiva precisa analisar documentos: "pela sua descrição vejo X, mas pra te dar um parecer firme preciso olhar [documento]".
- Honorários: NÃO dê valor antes de coletar o caso completo. Diga: "preciso entender melhor pra te passar o orçamento certo — assim eu vejo a complexidade e te falo o investimento real."
- Sigla **CNIS** sempre em maiúsculas (Cadastro Nacional de Informações Sociais). Nunca "qinis", "kinis", "cinis".

ÁREAS QUE VOCÊ ATENDE (suas especialidades — domine TODAS):
1. **Direito Trabalhista** — demissão sem justa causa, horas extras, assédio, FGTS, rescisão. Pergunte: tempo de empresa, data da demissão, salário, se assinou rescisão, valores pendentes.
2. **Direito de Família** — divórcio, pensão, guarda, partilha. Pergunte: situação atual, há filhos menores, há bens em comum, urgência.
3. **Direito Previdenciário (INSS/BPC)** — aposentadoria, BPC/LOAS, auxílio-doença, pensão por morte. Pergunte: idade, tempo de contribuição, CNIS, motivo do indeferimento.
4. **Direito Cível** — indenização, cobrança, contratos, acidente. Pergunte: o que aconteceu, quando, provas/documentos, valor envolvido.
5. **Direito do Consumidor** — produto/serviço defeituoso, cobrança indevida, negativação. Pergunte: data da compra, valor, tentativa amigável, notas fiscais.
6. **Direito Criminal** — denúncia, prisão, audiência, recurso. Pergunte com cuidado e sem julgar: situação atual (preso/liberdade?), prazo de audiência, tem defensoria. NUNCA pergunte se "fez" o crime — pergunte sobre o processo.
7. **Direito Empresarial** — contratos, registro, sociedade, tributário leve. Pergunte: porte (MEI/ME/EPP), área de atuação, principal dor.

⚠️⚠️ REGRA DE OURO PARA DESCOBRIR A ÁREA ⚠️⚠️
Faça pergunta ABERTA. NUNCA liste as 7 categorias acima — isso entrega o jogo e parece formulário.

❌ ERRADO: "Seu caso é trabalhista, de família, INSS, cível, criminal ou consumidor?"
✅ CERTO: "Me conta o que aconteceu — pode ser informal mesmo."
✅ CERTO: "Conta brevemente o que está acontecendo."
✅ CERTO: "Em poucas palavras, o que te trouxe aqui hoje?"

Quando IDENTIFICAR a área pelo relato espontâneo, ATIVE SILENCIOSAMENTE o roteiro daquela área. Não anuncie "agora vou perguntar sobre trabalhista" — apenas faça as perguntas certas, naturalmente.

OBJETIVO PRINCIPAL:
Você é uma CLOSER — conduzir o cliente até MARCAR uma reunião de análise no Google Meet com você. Cada mensagem deve dar UM PASSO concreto.

ROTEIRO BASE (siga em ordem, UMA pergunta por vez):
1. Cumprimentar pelo nome (se já souber) ou perguntar o NOME. Use o nome em TODAS as próximas respostas.
2. Pergunta ABERTA sobre o caso (NUNCA liste categorias).
3. Identificada a área, ATIVE o roteiro específico.
4. Identificar URGÊNCIA: "tem prazo correndo?", "já recebeu alguma carta?", "quando aconteceu?".
5. Coletar CIDADE/ESTADO (atendimento online ou presencial).
6. Perguntar se já tem ALGUM PROCESSO ou advogado anterior.
7. PROPOR uma análise de 20-30 min com você no Google Meet, oferecendo 2 horários (seg-sex, 09-12h e 14-18h). O sistema te passa os HORÁRIOS LIVRES no final do prompt — use SÓ esses. Nunca fim-de-semana.
8. Confirmar e tranquilizar: "perfeito, marcado. Vou te chamar exatamente nesse horário pelo Google Meet — te mando o link logo antes."

HUMANIZAÇÃO (essencial — VOCÊ É HUMANA):
- UMA pergunta por mensagem. NÃO enxurrada de perguntas juntas.
- Mensagens CURTAS: 2 a 4 linhas no máximo. WhatsApp não é e-mail.
- Use o NOME do cliente em todas as respostas.
- Marcadores naturais de fala humana (use 1 por mensagem, sem exagero): "Hm...", "Olha só,", "Entendi", "Nossa, sinto muito", "Ah!", "Tá", "Beleza", "Que bom que mandou", "Eita".
- Empatia ANTES da pergunta. SEMPRE reconheça a dor primeiro.
- Termine com pergunta direta OU proposta concreta.
- Se cliente tergiversar: "te entendo, [nome]. Pra eu te ajudar do jeito certo, me confirma só uma coisa: ..."
- VARIAR cumprimentos: NUNCA comece duas msgs seguidas com "Oi/Olá". Alterne: "Tudo bem, [nome]?", "Que bom que falou", "Olha só, [nome]", "Estou aqui".
- Use contrações naturais: "tá", "pra", "tô", "né" — sem exagero, mas mostra que é gente, não formal.
- Demonstre que ESTÁ NO CASO: "anotei aqui", "deixa eu te explicar", "vou olhar isso direitinho", "isso aqui é simples de resolver, viu".

QUANDO ENCERRAR / TRANSFERIR:
- Cliente quer falar com humano → "claro, [nome]! Sou eu mesma falando aqui. Quer que eu te ligue agora?"
- Cliente xinga/desrespeita → "prefiro continuar quando estiver mais tranquilo(a). Estou aqui quando precisar."
- Assunto fora da advocacia → "sobre isso não consigo te ajudar, mas pra qualquer questão jurídica conta comigo."

ASSINATURA:
- Use "— Kênia 🤍" SÓ no fechamento de agendamento ou em mensagem importante (não em toda resposta).

EXEMPLO BOM (caso trabalhista):
Cliente: "Fui demitida sem justa causa, tô perdida"
Você: "Nossa, que situação difícil — sinto muito, viu. Aqui é a Kênia Garcia, advogada. Antes de tudo, me ajuda com uma coisa: faz quanto tempo da demissão e já te entregaram a rescisão?"

EXEMPLO BOM (caso previdenciário):
Cliente: "Meu pedido de aposentadoria foi negado"
Você: "Ah, que chato — INSS dá um trabalhão mesmo. Aqui é a Kênia, advogada. Pra eu te avaliar direitinho, me conta: o indeferimento veio por tempo de contribuição, idade ou outro motivo? E você consegue puxar seu CNIS atualizado no meu.inss.gov.br?"

EXEMPLO RUIM (NÃO FAZER):
"Oi, sou a Nislainy, secretária da Dra. Kênia"
(ERRADO — você é a Dra. Kênia em pessoa)

"Vou passar pra Dra. Kênia te atender"
(ERRADO — VOCÊ É ELA)

"Seu caso é trabalhista, de família, INSS ou cível?"
(ERRADO — listou categorias. Faça pergunta aberta.)"""


async def get_wa_config(owner_id: str) -> Dict[str, Any]:
    cfg = await db.whatsapp_config.find_one({"owner_id": owner_id}, {"_id": 0})
    if not cfg:
        cfg = {
            "owner_id": owner_id,
            "provider": "zapi",
            "zapi_instance_id": os.environ.get("ZAPI_INSTANCE_ID", ""),
            "zapi_instance_token": os.environ.get("ZAPI_INSTANCE_TOKEN", ""),
            "zapi_client_token": os.environ.get("ZAPI_CLIENT_TOKEN", ""),
            "evo_base_url": "", "evo_api_key": "", "evo_instance": "",
            "meta_access_token": "", "meta_phone_number_id": "",
            "bot_enabled": False,
            "bot_prompt": KENIA_DEFAULT_PROMPT,
            "bot_voice_mode": "text_and_audio",
            "bot_voice": "nova",
        }
        await db.whatsapp_config.insert_one({**cfg})
        cfg.pop("_id", None)
    # backfill defaults para configs antigas que nao tinham os campos de voz
    if "bot_voice_mode" not in cfg:
        cfg["bot_voice_mode"] = "text_and_audio"
    if "bot_voice" not in cfg:
        cfg["bot_voice"] = "nova"
    if "voice_provider" not in cfg:
        cfg["voice_provider"] = "openai"
    if "elevenlabs_api_key" not in cfg:
        cfg["elevenlabs_api_key"] = None
    if "elevenlabs_voice_id" not in cfg:
        cfg["elevenlabs_voice_id"] = None
    if "elevenlabs_voice_name" not in cfg:
        cfg["elevenlabs_voice_name"] = None
    return cfg

@api_router.get("/whatsapp/config")
async def get_config(current_user=Depends(get_current_user)):
    cfg = await get_wa_config(current_user["id"])
    # mask some secrets for UI
    return cfg

@api_router.get("/whatsapp/default-prompt")
async def whatsapp_default_prompt(current_user=Depends(get_current_user)):
    """Retorna o prompt padrao (Kenia Garcia, assertivo) para a UI poder
    resetar/preencher o campo bot_prompt em 1 clique."""
    return {"prompt": KENIA_DEFAULT_PROMPT}

@api_router.put("/whatsapp/config")
async def set_config(payload: WhatsAppConfig, current_user=Depends(get_current_user)):
    data = payload.model_dump()
    data["owner_id"] = current_user["id"]
    data["updated_at"] = now_iso()
    await db.whatsapp_config.update_one(
        {"owner_id": current_user["id"]}, {"$set": data}, upsert=True,
    )
    return {"ok": True}

@api_router.post("/whatsapp/test-connection")
async def test_connection(current_user=Depends(get_current_user)):
    cfg = await get_wa_config(current_user["id"])
    prov = build_provider_from_config(cfg)
    if not prov:
        return {"ok": False, "error": "Provedor não configurado", "provider": cfg.get("provider")}
    res = await prov.test_connection()
    return {"provider": prov.name, **res}

@api_router.get("/whatsapp/diagnostics")
async def whatsapp_diagnostics(request: Request, public_url: Optional[str] = None,
                               current_user=Depends(get_current_user)):
    """Faz um diagnostico completo do setup WhatsApp para ajudar o usuario
    a entender por que nao esta recebendo mensagens novas."""
    cfg = await get_wa_config(current_user["id"])
    prov = build_provider_from_config(cfg)

    # Determina URL publica: prioriza query param do frontend (mais confiavel),
    # depois Origin header, depois X-Forwarded-*, por ultimo request.base_url.
    if public_url:
        public_base = public_url.rstrip("/")
    else:
        origin = request.headers.get("origin") or ""
        fwd_proto = request.headers.get("x-forwarded-proto") or ""
        fwd_host = request.headers.get("x-forwarded-host") or ""
        if origin:
            public_base = origin.rstrip("/")
        elif fwd_host:
            public_base = f"{fwd_proto or 'https'}://{fwd_host}"
        else:
            public_base = str(request.base_url).rstrip("/")
    expected_webhook = f"{public_base}/api/whatsapp/webhook/zapi"

    checks = []
    # 1) credenciais preenchidas
    creds_ok = bool(prov)
    checks.append({
        "id": "credentials",
        "label": "Credenciais do provedor",
        "ok": creds_ok,
        "msg": "Credenciais preenchidas." if creds_ok
               else "Preencha as credenciais do provedor selecionado.",
    })

    # 2) bot ligado
    bot_on = bool(cfg.get("bot_enabled"))
    checks.append({
        "id": "bot_enabled",
        "label": "Robô IA ativado",
        "ok": bot_on,
        "msg": "Robô ligado — vai responder mensagens recebidas." if bot_on
               else "Robô desligado. Ative o switch 'Robô Atendente IA' para responder automaticamente.",
    })

    # 3) conexao com o provedor (status)
    connected = False
    conn_msg = "Provedor nao configurado."
    conn_hint = None
    if prov:
        try:
            res = await prov.test_connection()
            connected = bool(res.get("connected"))
            conn_hint = res.get("hint")
            if connected:
                conn_msg = "WhatsApp conectado ao provedor."
            else:
                conn_msg = "Nao conectado. Verifique o QR Code (Z-API) ou estado da instancia."
        except Exception as e:
            conn_msg = f"Erro ao consultar status: {str(e)[:120]}"
    checks.append({
        "id": "provider_connected",
        "label": "Conexao com WhatsApp",
        "ok": connected,
        "msg": conn_msg,
        "hint": conn_hint,
    })

    # 4) webhook configurado (Z-API tem como verificar via API; Baileys e automatico)
    webhook_configured = False
    webhook_msg = "Verificacao automatica indisponivel para este provedor."
    if isinstance(prov, BaileysProvider):
        # Baileys: o webhook e interno (Node.js sidecar -> backend), sempre OK
        # se o sidecar esta rodando
        webhook_configured = bool(prov)
        webhook_msg = (
            "Baileys: webhook automatico (interno). Mensagens chegam direto ao backend."
            if webhook_configured else
            "Baileys: sidecar nao disponivel."
        )
    elif isinstance(prov, ZAPIProvider):
        # Z-API nao tem GET unificado para o webhook. Estrategia:
        # 1) Se ja recebemos mensagens nas ultimas 24h => webhook funcionando
        # 2) Se setup-webhook foi confirmado (cfg.webhook_url_configured),
        #    consideramos OK ate que o usuario reset
        # 3) Tentamos GET /webhook-received-by-me; se NOT_FOUND, marcamos
        #    como "verificacao indisponivel" mas com instrucoes claras.
        try:
            res = await prov._request("GET", f"{prov.base}/webhook-received-by-me", timeout=15.0)
            data = res.get("data") or {}
            current_url = data.get("value") or data.get("url") or ""
            err = str(data.get("error") or "").upper()
            saved_webhook = cfg.get("webhook_url_configured") or ""
            if current_url:
                webhook_configured = True
                if expected_webhook in current_url or current_url.endswith("/api/whatsapp/webhook/zapi"):
                    webhook_msg = f"Webhook configurado corretamente: {current_url}"
                else:
                    webhook_msg = (
                        f"Webhook configurado mas para outra URL: {current_url}. "
                        f"O esperado e: {expected_webhook}"
                    )
                    webhook_configured = False
            elif err == "NOT_FOUND":
                # Z-API nao expoe GET para ler webhook. Verifica se mensagens
                # estao chegando ou se o setup foi confirmado.
                since_24 = datetime.now(timezone.utc) - timedelta(hours=24)
                since24_iso = since_24.isoformat()
                got_msgs = await db.whatsapp_messages.count_documents({
                    "owner_id": current_user["id"],
                    "from_me": False,
                    "provider": "zapi",
                    "created_at": {"$gte": since24_iso},
                })
                if got_msgs > 0:
                    webhook_configured = True
                    webhook_msg = (
                        f"Webhook funcionando ({got_msgs} mensagem(ns) recebidas nas ultimas 24h)."
                    )
                elif saved_webhook and saved_webhook.endswith("/api/whatsapp/webhook/zapi"):
                    webhook_configured = True
                    webhook_msg = (
                        f"Webhook configurado via auto-setup: {saved_webhook}. "
                        "Envie uma mensagem do seu celular para validar o recebimento."
                    )
                else:
                    webhook_msg = (
                        "Clique em 'Configurar webhook automaticamente' para garantir que esta apontando "
                        f"para: {expected_webhook}. Depois envie uma mensagem do seu celular para testar."
                    )
            else:
                webhook_msg = (
                    "Webhook NAO configurado na Z-API. Sem isso voce NAO recebera mensagens novas. "
                    "Clique em 'Configurar Z-API automaticamente' ou cole esta URL no painel Z-API "
                    f"em Webhooks > Ao receber: {expected_webhook}"
                )
        except Exception as e:
            webhook_msg = f"Erro ao consultar webhook: {str(e)[:120]}"
    checks.append({
        "id": "webhook",
        "label": "Webhook de recebimento",
        "ok": webhook_configured,
        "msg": webhook_msg,
        "expected": expected_webhook,
    })

    # 5) mensagens recebidas recentemente (ultimas 24h)
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    since_iso = since.isoformat()
    recent_count = await db.whatsapp_messages.count_documents({
        "owner_id": current_user["id"],
        "from_me": False,
        "created_at": {"$gte": since_iso},
    })
    checks.append({
        "id": "recent_messages",
        "label": "Mensagens recebidas nas ultimas 24h",
        "ok": recent_count > 0,
        "msg": f"{recent_count} mensagem(ns) recebida(s)." if recent_count
               else "Nenhuma mensagem recebida nas ultimas 24h. Se acabou de configurar, envie uma mensagem do seu celular para o WhatsApp conectado para testar.",
    })

    overall_ok = all(c["ok"] for c in checks)
    return {
        "ok": overall_ok,
        "provider": cfg.get("provider"),
        "checks": checks,
        "expected_webhook_url": expected_webhook,
    }

@api_router.get("/whatsapp/qr")
async def get_qr(current_user=Depends(get_current_user)):
    cfg = await get_wa_config(current_user["id"])
    prov = build_provider_from_config(cfg)
    if isinstance(prov, BaileysProvider):
        return await prov.get_qr()
    if not isinstance(prov, ZAPIProvider):
        return {"ok": False, "error": "QR disponível apenas no provedor Z-API ou Baileys"}
    return await prov.get_qr()


@api_router.get("/whatsapp/baileys/status")
async def baileys_status(current_user=Depends(get_current_user)):
    """Status do servico Baileys (Node.js sidecar)."""
    prov = BaileysProvider()
    return await prov.test_connection()


@api_router.get("/whatsapp/baileys/qr")
async def baileys_qr(current_user=Depends(get_current_user)):
    prov = BaileysProvider()
    return await prov.get_qr()


@api_router.post("/whatsapp/baileys/logout")
async def baileys_logout(current_user=Depends(get_current_user)):
    """Desconecta do WhatsApp (limpa sessao Baileys e gera novo QR)."""
    prov = BaileysProvider()
    return await prov.logout()


@api_router.post("/whatsapp/baileys/restart")
async def baileys_restart(current_user=Depends(get_current_user)):
    """Reseta a sessão Baileys do zero: apaga auth_info, mata o socket e gera
    novo QR. Útil quando o QR expirou várias vezes ou ficou travado em
    "connecting".
    """
    token = os.environ.get("BAILEYS_INTERNAL_TOKEN") or "legalflow-baileys-2026"
    base_url = os.environ.get("BAILEYS_URL", "http://localhost:8002")
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post(
                f"{base_url}/restart",
                headers={"x-internal-token": token},
            )
            return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@api_router.post("/whatsapp/baileys/reconnect")
async def baileys_reconnect(current_user=Depends(get_current_user)):
    """Forca o sidecar a subir (caso esteja morto) e retorna status.
    Util quando o preview ficou ocioso e o sidecar caiu.
    """
    # 1) se sidecar nao esta respondendo, spawna
    if not await _baileys_health_ok():
        _spawn_baileys()
        import asyncio as _a
        # espera ate 8s pelo sidecar subir
        for _ in range(16):
            await _a.sleep(0.5)
            if await _baileys_health_ok(): break
    # 2) retorna status atualizado
    prov = BaileysProvider()
    return await prov.test_connection()


# ==================== VOZ (STT + TTS) ====================

from fastapi.responses import Response

@api_router.post("/voice/transcribe")
async def voice_transcribe(file: UploadFile = File(...), current_user=Depends(get_current_user)):
    """Transcreve um audio (mp3/ogg/m4a/wav/webm) usando Whisper."""
    raw = await file.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(400, "Arquivo maior que 25MB")
    bio = io.BytesIO(raw)
    bio.name = file.filename or "voice.ogg"
    try:
        stt = OpenAISpeechToText(api_key=EMERGENT_LLM_KEY)
        resp = await stt.transcribe(file=bio, model="whisper-1", language="pt", response_format="json")
        txt = (getattr(resp, "text", None) or "").strip()
        return {"ok": True, "text": txt}
    except Exception as e:
        log.exception("transcribe failed")
        raise HTTPException(500, f"Erro na transcricao: {str(e)[:200]}")


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "nova"


@api_router.post("/voice/tts")
async def voice_tts(payload: TTSRequest, current_user=Depends(get_current_user)):
    """Gera audio MP3 do texto informado usando OpenAI TTS."""
    try:
        tts = OpenAITextToSpeech(api_key=EMERGENT_LLM_KEY)
        audio_bytes = await tts.generate_speech(
            text=payload.text[:4000], model="tts-1",
            voice=payload.voice or "nova", response_format="mp3",
        )
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        log.exception("tts failed")
        raise HTTPException(500, f"Erro no TTS: {str(e)[:200]}")


class VoiceCommandRequest(BaseModel):
    audio_base64: str
    mime: Optional[str] = "audio/webm"
    voice: Optional[str] = "nova"


@api_router.post("/voice/command")
async def voice_command(payload: VoiceCommandRequest, current_user=Depends(get_current_user)):
    """Recebe audio (base64), transcreve, processa com IA e retorna texto + audio base64."""
    try:
        raw = base64.b64decode(payload.audio_base64)
        ext = "webm"
        ml = (payload.mime or "").lower()
        if "mp4" in ml or "m4a" in ml: ext = "m4a"
        elif "ogg" in ml: ext = "ogg"
        elif "mp3" in ml or "mpeg" in ml: ext = "mp3"
        elif "wav" in ml: ext = "wav"
        bio = io.BytesIO(raw)
        bio.name = f"voice.{ext}"
        stt = OpenAISpeechToText(api_key=EMERGENT_LLM_KEY)
        resp = await stt.transcribe(file=bio, model="whisper-1", language="pt", response_format="json")
        user_text = (getattr(resp, "text", None) or "").strip()
        if not user_text:
            return {"ok": False, "error": "nao-transcrito"}
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"voice-{current_user['id']}-{uuid.uuid4().hex[:6]}",
            system_message=LEGAL_SYSTEM_PROMPT,
        ).with_model("openai", "gpt-4o-mini")
        reply = await chat.send_message(UserMessage(text=user_text))
        tts = OpenAITextToSpeech(api_key=EMERGENT_LLM_KEY)
        audio_bytes = await tts.generate_speech(
            text=(reply or "")[:4000], model="tts-1",
            voice=payload.voice or "nova", response_format="mp3",
        )
        out_b64 = base64.b64encode(audio_bytes).decode("ascii")
        return {"ok": True, "transcription": user_text, "reply": reply, "audio_base64": out_b64, "audio_mime": "audio/mpeg"}
    except Exception as e:
        log.exception("voice command failed")
        raise HTTPException(500, f"Erro no comando de voz: {str(e)[:200]}")



class WebhookSetup(BaseModel):
    base_url: Optional[str] = None  # if not provided, use server-side backend

@api_router.post("/whatsapp/setup-webhook")
async def setup_webhook(payload: WebhookSetup, request: Request, current_user=Depends(get_current_user)):
    """Configura webhook na Z-API para apontar para o nosso backend.
    Estrategia: usa o endpoint /update-every-webhooks (1 chamada para todos
    os tipos de evento) e depois VERIFICA via GET /webhook-received-by-me.
    Retorna detalhes claros de exito ou falha."""
    cfg = await get_wa_config(current_user["id"])
    if cfg.get("provider") != "zapi":
        raise HTTPException(400, "Setup automático disponível apenas para Z-API")
    prov = build_provider_from_config(cfg)
    if not isinstance(prov, ZAPIProvider):
        raise HTTPException(400, "Credenciais Z-API incompletas")
    # Determine backend URL: prioriza payload, depois Origin do frontend,
    # depois X-Forwarded-Host, por ultimo request.base_url.
    base = payload.base_url
    if not base:
        origin = request.headers.get("origin") or ""
        fwd_proto = request.headers.get("x-forwarded-proto") or ""
        fwd_host = request.headers.get("x-forwarded-host") or ""
        if origin:
            base = origin.rstrip("/")
        elif fwd_host:
            base = f"{fwd_proto or 'https'}://{fwd_host}"
        else:
            base = str(request.base_url).rstrip("/")
    if base.startswith("http://"):
        # Z-API exige HTTPS para webhooks
        base = "https://" + base[len("http://"):]
    webhook_url = f"{base}/api/whatsapp/webhook/zapi"

    results: Dict[str, Any] = {}
    hint = None
    errors = []

    # 1) update-every-webhooks (atualiza todos de uma vez, recomendado)
    try:
        every_res = await prov._request(
            "PUT",
            f"{prov.base}/update-every-webhooks",
            json={"value": webhook_url, "notifySentByMe": False},
            timeout=20.0,
        )
        results["update-every-webhooks"] = {
            "status": every_res["status"],
            "body": every_res["data"],
        }
        if every_res.get("hint"):
            hint = every_res["hint"]
        if every_res["status"] >= 400:
            errors.append(
                f"update-every-webhooks retornou {every_res['status']}: "
                f"{str(every_res['data'])[:200]}"
            )
    except Exception as e:
        results["update-every-webhooks"] = {"error": str(e)}
        errors.append(str(e))

    # 2) Fallback individual (para Z-API antigas que nao suportam update-every-webhooks)
    if errors:
        for path in ("update-webhook-received", "update-webhook-delivery",
                     "update-webhook-message-status"):
            try:
                res = await prov._request(
                    "PUT", f"{prov.base}/{path}",
                    json={"value": webhook_url}, timeout=20.0,
                )
                results[path] = {"status": res["status"], "body": res["data"]}
                if res.get("hint") and not hint:
                    hint = res["hint"]
                if res["status"] >= 400:
                    errors.append(f"{path}: {res['status']}")
                else:
                    # se algum individual deu certo, limpa errors anteriores
                    errors = [e for e in errors if "update-every-webhooks" not in e]
            except Exception as e:
                results[path] = {"error": str(e)}

    # 3) VERIFICACAO: a Z-API nao expoe um GET unificado para ler o webhook configurado.
    # O proprio update-every-webhooks retorna {"value": true} quando aceita a operacao.
    # Tentamos GET /webhook-received-by-me como complemento, mas tratamos NOT_FOUND como
    # "endpoint indisponivel" e nao como falha.
    verified = False
    verified_url = None
    every_body = (results.get("update-every-webhooks") or {}).get("body") or {}
    every_status = (results.get("update-every-webhooks") or {}).get("status") or 0
    if every_status == 200 and (
        every_body.get("value") is True or
        (isinstance(every_body, dict) and not every_body.get("error"))
    ):
        verified = True
        verified_url = webhook_url
    else:
        # tenta o endpoint legado de leitura
        try:
            ver = await prov._request(
                "GET", f"{prov.base}/webhook-received-by-me", timeout=15.0,
            )
            results["verify"] = {"status": ver["status"], "body": ver["data"]}
            d = ver.get("data") or {}
            verified_url = d.get("value") or d.get("url") or ""
            if verified_url and (
                verified_url.rstrip("/") == webhook_url.rstrip("/") or
                verified_url.endswith("/api/whatsapp/webhook/zapi")
            ):
                verified = True
        except Exception as e:
            results["verify"] = {"error": str(e)}

    ok = verified or (not errors)
    out = {
        "ok": ok,
        "verified": verified,
        "verified_url": verified_url,
        "webhook_url": webhook_url,
        "results": results,
    }
    if hint:
        out["hint"] = hint
    if errors and not verified:
        out["errors"] = errors[:5]
        out["error_summary"] = (
            "A Z-API rejeitou a configuracao do webhook. "
            "Verifique se o Client-Token (Account Security Token) esta correto, "
            "ou cole manualmente a URL no painel Z-API > Webhooks > Ao receber."
        )
    # Persiste o sucesso no config para o diagnostico marcar como OK
    if verified:
        await db.whatsapp_config.update_one(
            {"owner_id": current_user["id"]},
            {"$set": {
                "webhook_url_configured": webhook_url,
                "webhook_set_at": now_iso(),
            }},
        )
    return out

# ==================== WHATSAPP CONTACTS / MESSAGES ====================

@api_router.get("/whatsapp/contacts")
async def whatsapp_contacts(current_user=Depends(get_current_user)):
    contacts = await db.whatsapp_contacts.find(
        {"owner_id": current_user["id"]}, {"_id": 0}
    ).sort("last_message_at", -1).to_list(200)
    return contacts

@api_router.get("/whatsapp/messages/{contact_id}")
async def whatsapp_messages(contact_id: str, current_user=Depends(get_current_user)):
    msgs = await db.whatsapp_messages.find(
        {"contact_id": contact_id, "owner_id": current_user["id"]}, {"_id": 0}
    ).sort("created_at", 1).to_list(500)
    return msgs

@api_router.post("/whatsapp/send")
async def whatsapp_send(payload: WhatsAppMessageIn, current_user=Depends(get_current_user)):
    """Envia mensagem para contato existente (busca telefone no contato)."""
    contact = await db.whatsapp_contacts.find_one(
        {"id": payload.contact_id, "owner_id": current_user["id"]}, {"_id": 0}
    )
    if not contact:
        raise HTTPException(404, "Contato não encontrado")

    cfg = await get_wa_config(current_user["id"])
    prov = build_provider_from_config(cfg)
    provider_result: Dict[str, Any] = {"ok": False, "error": "provider-not-configured"}
    delivered = False
    if prov:
        try:
            # Baileys: reply using stored @lid/phone JID when present so the
            # message goes back to the original chat (not a phantom rebuilt jid).
            if isinstance(prov, BaileysProvider):
                provider_result = await prov.send_text(
                    contact["phone"], payload.text,
                    jid=(contact.get("wa_jid") or None),
                )
            else:
                provider_result = await prov.send_text(contact["phone"], payload.text)
            delivered = bool(provider_result.get("ok"))
        except Exception as e:
            log.exception("send whatsapp error")
            provider_result = {"ok": False, "error": str(e)}

    msg_id = str(uuid.uuid4())
    doc = {
        "id": msg_id, "owner_id": current_user["id"],
        "contact_id": payload.contact_id, "text": payload.text,
        "from_me": True, "delivered": delivered,
        "provider": prov.name if prov else None,
        "created_at": now_iso(),
    }
    await db.whatsapp_messages.insert_one(doc)
    await db.whatsapp_contacts.update_one(
        {"id": payload.contact_id, "owner_id": current_user["id"]},
        {"$set": {"last_message": payload.text, "last_message_at": now_iso()}},
    )
    doc.pop("_id", None)
    return {"message": doc, "provider_result": provider_result}

@api_router.post("/whatsapp/send-direct")
async def whatsapp_send_direct(payload: WhatsAppSendDirect, current_user=Depends(get_current_user)):
    """Envia mensagem direta por telefone (cria contato se não existir)."""
    phone_n = normalize_phone(payload.phone)
    contact = await db.whatsapp_contacts.find_one(
        {"owner_id": current_user["id"], "phone_normalized": phone_n}, {"_id": 0}
    )
    if not contact:
        contact = {
            "id": str(uuid.uuid4()), "owner_id": current_user["id"],
            "name": payload.phone, "phone": payload.phone, "phone_normalized": phone_n,
            "last_message": payload.text, "unread": 0,
            "last_message_at": now_iso(), "avatar_color": "bg-emerald-500",
        }
        await db.whatsapp_contacts.insert_one({**contact})

    cfg = await get_wa_config(current_user["id"])
    prov = build_provider_from_config(cfg)
    provider_result: Dict[str, Any] = {"ok": False, "error": "provider-not-configured"}
    delivered = False
    if prov:
        try:
            provider_result = await prov.send_text(payload.phone, payload.text)
            delivered = bool(provider_result.get("ok"))
        except Exception as e:
            log.exception("send-direct error")
            provider_result = {"ok": False, "error": str(e)}

    msg_id = str(uuid.uuid4())
    doc = {
        "id": msg_id, "owner_id": current_user["id"],
        "contact_id": contact["id"], "text": payload.text,
        "from_me": True, "delivered": delivered,
        "provider": prov.name if prov else None,
        "created_at": now_iso(),
    }
    await db.whatsapp_messages.insert_one(doc)
    await db.whatsapp_contacts.update_one(
        {"id": contact["id"]},
        {"$set": {"last_message": payload.text, "last_message_at": now_iso()}},
    )
    return {"ok": True, "delivered": delivered, "provider_result": provider_result,
            "contact_id": contact["id"], "message_id": msg_id}

# ==================== WEBHOOK (mensagens recebidas) ====================

async def _resolve_owner_for_provider(provider_name: str, instance_id: Optional[str] = None) -> Optional[str]:
    """Resolve qual usuario e o dono das mensagens recebidas para um provider.
    Estrategia (do mais especifico ao mais geral):
      1) whatsapp_config com provider+instance_id, ordenado por updated_at DESC
      2) whatsapp_config com provider apenas (ultimo configurado)
      3) Primeiro usuario do sistema (retro-compat single-user)
    """
    query: Dict[str, Any] = {"provider": provider_name}
    if provider_name == "zapi" and instance_id:
        query["zapi_instance_id"] = instance_id
    cfg = await db.whatsapp_config.find_one(
        query, {"_id": 0, "owner_id": 1}, sort=[("updated_at", -1)],
    )
    if cfg and cfg.get("owner_id"):
        return cfg["owner_id"]
    # Fallback: qualquer config com esse provider, mais recente primeiro
    cfg = await db.whatsapp_config.find_one(
        {"provider": provider_name}, {"_id": 0, "owner_id": 1},
        sort=[("updated_at", -1)],
    )
    if cfg and cfg.get("owner_id"):
        return cfg["owner_id"]
    # Ultimo fallback: primeiro user
    first = await db.users.find_one({}, {"_id": 0, "id": 1})
    return first["id"] if first else None


async def _save_incoming_message(
    owner_id: str, phone: str, name: str, text: str, provider_name: str,
    jid: Optional[str] = None, is_lid: bool = False, phone_jid: Optional[str] = None,
):
    """Persist incoming WhatsApp message + upsert contact.

    The `jid` parameter is the ORIGINAL remoteJid we received the message on
    (e.g. `1234567890@s.whatsapp.net` OR an anonymous `xxxxx@lid`).  We store
    it on the contact so that replies are routed back to that exact chat
    instead of being sent to a rebuilt ${digits}@s.whatsapp.net (which would
    create a phantom new conversation — the "+8955..." bug).
    """
    phone_n = normalize_phone(phone)
    contact = await db.whatsapp_contacts.find_one(
        {"owner_id": owner_id, "phone_normalized": phone_n}, {"_id": 0}
    )
    now = now_iso()
    update_set: Dict[str, Any] = {"last_message": text, "last_message_at": now}
    if jid:
        update_set["wa_jid"] = jid
        update_set["is_lid"] = bool(is_lid)
    if phone_jid:
        update_set["wa_phone_jid"] = phone_jid
    if not contact:
        contact = {
            "id": str(uuid.uuid4()), "owner_id": owner_id,
            "name": name or phone, "phone": phone, "phone_normalized": phone_n,
            "last_message": text, "unread": 1,
            "last_message_at": now, "avatar_color": "bg-blue-500",
            "wa_jid": jid or "",
            "is_lid": bool(is_lid),
            "wa_phone_jid": phone_jid or "",
        }
        await db.whatsapp_contacts.insert_one({**contact})
    else:
        await db.whatsapp_contacts.update_one(
            {"id": contact["id"]},
            {"$set": update_set, "$inc": {"unread": 1}},
        )
        contact.update(update_set)

    msg = {
        "id": str(uuid.uuid4()), "owner_id": owner_id, "contact_id": contact["id"],
        "text": text, "from_me": False, "provider": provider_name,
        "created_at": now,
    }
    await db.whatsapp_messages.insert_one(msg)
    return contact, msg

async def _maybe_create_appointment_from_message(
    owner_id: str,
    contact: Dict[str, Any],
    user_text: str,
    recent_msgs: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Detecta se o cliente confirmou um horario e cria appointment + meeting link.
    Regras de seguranca:
    - Precisa haver pelo menos 1 msg do bot ANTES (contexto de proposta de horario)
    - Nao cria duplicado se ja houve appointment pro contato nas ultimas 2h
    - Se detectado, cria appointment no banco e retorna {id, starts_at, human_when, meeting_link}
    """
    if not user_text or len(user_text.strip()) < 2:
        return None
    # Evita duplicar: ja existe appointment recente pra esse contato?
    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    existing = await db.appointments.find_one({
        "owner_id": owner_id,
        "contact_id": contact["id"],
        "created_at": {"$gte": two_hours_ago},
    }, {"_id": 0, "id": 1})
    if existing:
        return None
    # Contexto: valida intencao de agendamento por DOIS caminhos:
    # (a) bot ofereceu horarios recentemente, OU
    # (b) o proprio cliente esta propondo um horario com palavras de confirmacao
    bot_offered = False
    for m in (recent_msgs or [])[-6:]:
        if m.get("from_me") and m.get("bot"):
            t = (m.get("text") or "").lower()
            if any(k in t for k in ["hoje", "amanhã", "amanha", "horário", "horario",
                                     "h ", "hs", ":00", ":30", "às", "as ", "marcar",
                                     "agendar", "consulta", "horarios", "disponibilidade"]):
                bot_offered = True
                break
    client_proposes = False
    ut = (user_text or "").lower()
    action_kw = ["marcar", "agendar", "combinar", "reservar", "pode ser",
                 "fica otimo", "fica ótimo", "fechado", "confirmo",
                 "topo", "aceito", "quero hoje", "quero amanhã", "quero amanha"]
    time_kw = ["h ", "hs", ":00", ":30", "manha", "manhã", "tarde", "noite",
               "meio-dia", "hoje", "amanhã", "amanha", "às", "as "]
    has_action = any(k in ut for k in action_kw)
    has_time = any(k in ut for k in time_kw)
    if has_action and has_time:
        client_proposes = True
    # regex extra: numero+h (ex: 14h, 16h30)
    import re as _re
    if _re.search(r"\b\d{1,2}[h:]\d{0,2}\b", ut) and (has_action or "hoje" in ut or "amanh" in ut):
        client_proposes = True
    if not (bot_offered or client_proposes):
        return None

    # Usa LLM para extrair datetime estruturado
    import json as _json
    # contexto temporal (fuso BR)
    from datetime import timezone as _tz
    br_tz = _tz(timedelta(hours=-3))
    now_br = datetime.now(br_tz)
    today_str = now_br.strftime("%Y-%m-%d")
    amanha_str = (now_br + timedelta(days=1)).strftime("%Y-%m-%d")
    depois_str = (now_br + timedelta(days=2)).strftime("%Y-%m-%d")
    weekday_pt = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"][now_br.weekday()]

    # Historico curto
    hist_lines = []
    for m in (recent_msgs or [])[-6:]:
        who = "Cliente" if not m.get("from_me") else "Secretaria"
        t = (m.get("text") or "").strip()
        if t:
            hist_lines.append(f"{who}: {t}")
    hist_lines.append(f"Cliente: {user_text}")
    hist = "\n".join(hist_lines)

    sys_prompt = (
        "Voce extrai horario de agendamento confirmado pelo CLIENTE em conversa de WhatsApp com a SECRETARIA da Dra. Kenia Garcia. "
        f"Hoje e {weekday_pt} {today_str} (fuso America/Sao_Paulo, UTC-3). "
        f"'amanha' = {amanha_str}, 'depois de amanha' = {depois_str}. "
        "O cliente so esta CONFIRMANDO um horario se:\n"
        "1) A secretaria ofereceu horarios concretos OU o cliente sugeriu um horario especifico, E\n"
        "2) O cliente respondeu com um horario definitivo (ex: 'pode ser 16h', 'amanha 10h fica otimo', 'hoje 14h', 'as 15', 'fechado 16h'), E\n"
        "3) Nao e uma pergunta ('pode ser amanha?' e pergunta, NAO confirma).\n\n"
        "Responda APENAS com JSON valido:\n"
        "{\"confirmed\": true/false, \"datetime\": \"YYYY-MM-DDTHH:MM:00-03:00\" ou null, \"human\": \"string legivel em PT-BR\" ou null}\n"
        "Exemplos:\n"
        "- 'pode ser amanha 16h' -> {\"confirmed\":true,\"datetime\":\"" + amanha_str + "T16:00:00-03:00\",\"human\":\"amanha as 16h\"}\n"
        "- 'hoje 14h fica otimo' -> {\"confirmed\":true,\"datetime\":\"" + today_str + "T14:00:00-03:00\",\"human\":\"hoje as 14h\"}\n"
        "- 'obrigado' -> {\"confirmed\":false,\"datetime\":null,\"human\":null}\n"
        "- 'pode ser amanha?' -> {\"confirmed\":false,\"datetime\":null,\"human\":null}"
    )
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"schedule-{contact['id']}-{uuid.uuid4().hex[:6]}",
            system_message=sys_prompt,
        ).with_model("openai", "gpt-4o-mini")
        raw = await chat.send_message(UserMessage(
            text=f"Conversa:\n{hist}\n\nO cliente confirmou algum horario? Retorne apenas o JSON."
        ))
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        data = _json.loads(raw)
    except Exception:
        log.exception("schedule-extract failed")
        return None

    if not data.get("confirmed") or not data.get("datetime"):
        return None
    try:
        dt = datetime.fromisoformat(str(data["datetime"]).replace("Z", "+00:00"))
    except Exception:
        return None
    # Sanidade: nao pode ser no passado (mais de 30 min atras) nem alem de 30 dias
    now_utc = datetime.now(timezone.utc)
    if dt < now_utc - timedelta(minutes=30):
        return None
    if dt > now_utc + timedelta(days=30):
        return None

    # REGRA: a Dra. Kenia NAO atende sabado (5) nem domingo (6). Agendamento em
    # fim-de-semana nao e criado automaticamente — a secretaria pede pra confirmar
    # manualmente com a doutora.
    dt_br = dt.astimezone(br_tz)
    if dt_br.weekday() >= 5:
        try:
            when_human_br = dt_br.strftime("%d/%m às %Hh%M")
        except Exception:
            when_human_br = data.get("human") or "esse horário"
        log.info(f"[appointment] weekend blocked: {dt_br.isoformat()} weekday={dt_br.weekday()}")
        return {
            "id": None,
            "weekend": True,
            "starts_at": dt.isoformat(),
            "human_when": when_human_br,
        }

    # Checa CONFLITO com outros appointments da Dra.
    # Considera-se conflito se existir appointment dentro de [dt - 45min, dt + 45min]
    # (consulta padrao dura 30min; margem de 15 min entre atendimentos).
    window_before = (dt - timedelta(minutes=45)).isoformat()
    window_after = (dt + timedelta(minutes=45)).isoformat()
    conflict = await db.appointments.find_one({
        "owner_id": owner_id,
        "status": {"$ne": "cancelado"},
        "contact_id": {"$ne": contact["id"]},  # permite mesmo contato reagendar
        "starts_at": {"$gte": window_before, "$lte": window_after},
    }, {"_id": 0, "starts_at": 1, "client_name": 1})
    if conflict:
        # Nao cria o agendamento e retorna um marcador pro bot avisar
        try:
            conflict_dt = datetime.fromisoformat(conflict["starts_at"])
            conflict_human = conflict_dt.astimezone(br_tz).strftime("%d/%m às %Hh%M")
        except Exception:
            conflict_human = "um horário próximo"
        log.info(f"[appointment] conflict detected at {dt.isoformat()} vs {conflict.get('starts_at')}")
        return {
            "id": None,
            "conflict": True,
            "starts_at": dt.isoformat(),
            "human_when": data.get("human") or dt.astimezone(br_tz).strftime("%d/%m às %Hh%M"),
            "conflict_with": conflict_human,
        }

    # Cria appointment com Meet link
    aid = str(uuid.uuid4())
    code = f"{uuid.uuid4().hex[:3]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:3]}"
    meeting_link = f"https://meet.google.com/{code}"
    human_when = data.get("human") or dt.astimezone(br_tz).strftime("%d/%m às %Hh%M")
    appt = {
        "id": aid,
        "owner_id": owner_id,
        "title": f"Consulta — {contact.get('name') or contact.get('phone')}",
        "client_name": contact.get("name") or contact.get("phone"),
        "contact_id": contact["id"],
        "starts_at": dt.isoformat(),
        "duration_min": 30,
        "location": "Google Meet",
        "meeting_link": meeting_link,
        "notes": f"Agendamento automatico via WhatsApp (Kenia IA). Cliente: '{user_text[:200]}'",
        "status": "confirmado",
        "source": "whatsapp-bot",
        "created_at": now_iso(),
    }
    await db.appointments.insert_one({**appt})

    # Promove o lead para "qualificado" (agendou consulta e lead eh quente)
    try:
        phone_digits = "".join(ch for ch in (contact.get("phone") or "") if ch.isdigit())
        suffix = phone_digits[-8:] if len(phone_digits) >= 8 else phone_digits
        if suffix:
            all_leads = await db.leads.find({"owner_id": owner_id}, {"_id": 0}).to_list(5000)
            for l in all_leads:
                lp = "".join(ch for ch in (l.get("phone") or "") if ch.isdigit())
                if lp.endswith(suffix):
                    await db.leads.update_one(
                        {"id": l["id"]},
                        {"$set": {
                            "stage": "qualificado",
                            "score": max(l.get("score", 50), 85),
                            "updated_at": now_iso(),
                        }},
                    )
                    break
    except Exception:
        log.exception("lead promote failed")

    appt["human_when"] = human_when
    return appt


# ============================================================================
# TTS unificado: usa OpenAI (Emergent LLM Key) por padrao, ou ElevenLabs com
# voz clonada da Dra. Kenia se cfg.voice_provider == "elevenlabs" e tiver chave.
# ============================================================================
async def _tts_generate(text: str, cfg: Dict[str, Any]) -> bytes:
    provider = (cfg.get("voice_provider") or "openai").lower()
    if provider == "elevenlabs":
        api_key = cfg.get("elevenlabs_api_key")
        voice_id = cfg.get("elevenlabs_voice_id")
        if not api_key or not voice_id:
            raise RuntimeError("ElevenLabs configurado mas falta api_key/voice_id — caindo para OpenAI.")
        from elevenlabs.client import ElevenLabs as _EL
        client = _EL(api_key=api_key)
        # SDK retorna generator; concatena os chunks.
        gen = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        buf = b""
        for chunk in gen:
            if isinstance(chunk, (bytes, bytearray)):
                buf += chunk
        return buf
    # default: OpenAI TTS via Emergent LLM Key
    voice_name = (cfg.get("bot_voice") or "nova").lower()
    tts = OpenAITextToSpeech(api_key=EMERGENT_LLM_KEY)
    return await tts.generate_speech(
        text=text, model="tts-1",
        voice=voice_name, response_format="mp3",
    )


# ============================================================================
# ENDPOINTS de voice cloning ElevenLabs
# ============================================================================
@api_router.post("/whatsapp/elevenlabs/clone")
async def elevenlabs_clone_voice(
    voice_name: str = Form(...),
    description: Optional[str] = Form(None),
    audio_file: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    """Clona voz da Dra. Kenia via ElevenLabs IVC (Instant Voice Cloning).
    Recebe audio_file (>= 30s recomendado), cria voz no plano ElevenLabs do
    cliente e salva voice_id em whatsapp_config.
    """
    cfg = await get_wa_config(current_user["id"])
    api_key = cfg.get("elevenlabs_api_key")
    if not api_key:
        raise HTTPException(400, "Configure sua ElevenLabs API key antes de clonar a voz.")
    audio_bytes = await audio_file.read()
    if len(audio_bytes) < 50_000:
        raise HTTPException(400, "Áudio muito curto. Envie um arquivo com pelo menos 30 segundos de fala clara.")
    try:
        from elevenlabs.client import ElevenLabs as _EL
        from io import BytesIO as _BIO
        client = _EL(api_key=api_key)
        bio = _BIO(audio_bytes)
        bio.name = audio_file.filename or "kenia_voice.mp3"
        try:
            voice = client.voices.ivc.create(
                name=voice_name,
                files=[bio],
                description=description or "Voz clonada da Dra. Kênia Garcia",
            )
        except AttributeError:
            voice = client.voices.add(
                name=voice_name,
                files=[bio],
                description=description or "Voz clonada da Dra. Kênia Garcia",
            )
        voice_id = getattr(voice, "voice_id", None) or getattr(voice, "id", None)
        if not voice_id:
            raise HTTPException(500, "ElevenLabs não retornou voice_id — verifique sua API key e plano.")
        await db.whatsapp_config.update_one(
            {"owner_id": current_user["id"]},
            {"$set": {
                "elevenlabs_voice_id": voice_id,
                "elevenlabs_voice_name": voice_name,
                "voice_provider": "elevenlabs",
            }},
            upsert=True,
        )
        return {"ok": True, "voice_id": voice_id, "voice_name": voice_name}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("ElevenLabs clone failed")
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg or "missing_permissions" in msg:
            raise HTTPException(401, "API key inválida ou sem permissão de clonagem. Verifique: 1) a key começa com 'sk_'; 2) você marcou 'Voices > Read' E 'Voices > Write' na hora de criar a key em elevenlabs.io/app/settings/api-keys.")
        elif "402" in msg or "quota" in msg.lower() or "subscription" in msg.lower():
            raise HTTPException(402, "Plano sem créditos para clonagem de voz. Faça upgrade em elevenlabs.io/app/subscription (Starter $5/mês inclui Instant Voice Cloning).")
        elif "voice_limit" in msg.lower() or "limit" in msg.lower():
            raise HTTPException(400, "Limite de vozes do seu plano atingido. Apague vozes antigas em elevenlabs.io/app/voice-library ou faça upgrade.")
        raise HTTPException(500, f"Falha no clone de voz: {msg[:200]}")


@api_router.get("/whatsapp/elevenlabs/voices")
async def elevenlabs_list_voices(current_user=Depends(get_current_user)):
    """Lista as vozes disponíveis na conta ElevenLabs do usuário."""
    cfg = await get_wa_config(current_user["id"])
    api_key = cfg.get("elevenlabs_api_key")
    if not api_key:
        return {"ok": False, "voices": [], "error": "API key não configurada"}
    try:
        from elevenlabs.client import ElevenLabs as _EL
        client = _EL(api_key=api_key)
        resp = client.voices.get_all()
        voices = getattr(resp, "voices", None) or resp
        out = []
        for v in voices:
            out.append({
                "voice_id": getattr(v, "voice_id", None) or getattr(v, "id", None),
                "name": getattr(v, "name", None),
                "category": getattr(v, "category", None),
            })
        return {"ok": True, "voices": out}
    except Exception as e:
        # Extrai mensagem amigavel — SDK do ElevenLabs as vezes retorna texto bruto
        # com status HTTP, outras vezes um JSON estruturado. Tenta achar a causa real.
        msg = str(e)
        # Tenta extrair body JSON da resposta do ElevenLabs (mais informativo)
        raw_detail = None
        try:
            import re as _re
            m = _re.search(r'"detail"\s*:\s*\{[^}]*"message"\s*:\s*"([^"]+)"', msg)
            if m:
                raw_detail = m.group(1)
            else:
                m2 = _re.search(r'"detail"\s*:\s*"([^"]+)"', msg)
                if m2:
                    raw_detail = m2.group(1)
        except Exception:
            pass

        clean = msg[:180]
        if "401" in msg or "Unauthorized" in msg or "missing_permissions" in msg or "invalid_api_key" in msg:
            clean = (
                "API key inválida. Causas mais comuns:\n"
                "1) Você copiou a key incompleta (cole novamente, sem espaços no fim).\n"
                "2) A key foi REVOGADA — vá em elevenlabs.io/app/settings/api-keys e crie outra.\n"
                "3) Permissões insuficientes — ao criar a key, marque 'Has access to all scopes'.\n"
                "4) Sua conta ElevenLabs não está verificada (confirme o e-mail)."
            )
        elif "402" in msg or "quota" in msg.lower() or "credits" in msg.lower():
            clean = "Sem créditos na conta ElevenLabs. Faça upgrade em elevenlabs.io/app/subscription."
        elif "429" in msg:
            clean = "Muitas requisições — aguarde alguns segundos e tente de novo."
        elif "free_users_not_allowed" in msg or "subscription" in msg.lower():
            clean = "Recurso bloqueado no plano FREE. Faça upgrade pro plano Starter ($5/mês) em elevenlabs.io/app/subscription."
        log.warning(f"ElevenLabs list voices error: {msg[:500]}")
        if raw_detail:
            clean = f"{clean}\n\n📨 Erro do ElevenLabs: {raw_detail}"
        return {"ok": False, "voices": [], "error": clean, "raw_error": msg[:300]}


@api_router.post("/whatsapp/elevenlabs/test")
async def elevenlabs_test_voice(
    text: str = Form("Olá! Esta é uma amostra da minha voz clonada. Tudo certo?"),
    current_user=Depends(get_current_user),
):
    """Gera áudio de teste com a voz clonada e retorna base64."""
    cfg = await get_wa_config(current_user["id"])
    if not cfg.get("elevenlabs_api_key") or not cfg.get("elevenlabs_voice_id"):
        raise HTTPException(400, "Clone uma voz antes de testar.")
    cfg2 = {**cfg, "voice_provider": "elevenlabs"}
    audio = await _tts_generate(text[:500], cfg2)
    return {"ok": True, "audio_base64": base64.b64encode(audio).decode("ascii"), "audio_mime": "audio/mpeg"}


async def _maybe_autorespond(
    owner_id: str,
    contact: Dict[str, Any],
    incoming_text: str,
    incoming_was_audio: bool = False,
):
    cfg = await get_wa_config(owner_id)
    if not cfg.get("bot_enabled"):
        return None
    prov = build_provider_from_config(cfg)
    if not prov:
        return None
    # detect sinestesic style — usa o atual + acumulado historico
    sty = detect_sinestesic(incoming_text) or contact.get("sinestesic_style")
    if sty and sty != contact.get("sinestesic_style"):
        await db.whatsapp_contacts.update_one(
            {"id": contact["id"]},
            {"$set": {"sinestesic_style": sty}},
        )
    style_hint = ""
    if sty and sty in SINESTESIC_STRATEGIES:
        style_hint = "\n\n=== ADAPTAÇÃO AO PERFIL DO CLIENTE ===\n" + SINESTESIC_STRATEGIES[sty]

    # Detecta nome real do cliente (se ja foi mencionado) — DEFINIDO CEDO pois e
    # usado no rastreador de estado abaixo.
    contact_name = (contact.get("name") or "").strip()
    contact_phone = (contact.get("phone") or "").strip()
    # Considera "phone-only" ou placeholder generico (ainda nao temos nome real)
    is_phone_only = (
        contact_name.isdigit()
        or "+" in contact_name
        or contact_name.startswith("55")
        or contact_name == contact_phone
        or contact_name.lower() in {"cliente", "contato", "whatsapp", ""}
    )
    name_hint = ""
    if contact_name and not is_phone_only:
        name_hint = f"\n\nNome do cliente (já confirmado): {contact_name}. Use este nome nas respostas."

    # Carrega historico (12 ultimas msgs) e monta narrativa cronologica
    last_msgs = await db.whatsapp_messages.find(
        {"contact_id": contact["id"]}, {"_id": 0}
    ).sort("created_at", -1).to_list(12)
    last_msgs.reverse()
    history_lines = []
    for m in last_msgs[:-1]:
        who = "Cliente" if not m.get("from_me") else "Secretaria"
        txt = (m.get("text") or "").strip()
        if txt:
            history_lines.append(f"{who}: {txt}")
    history_block = ""
    if history_lines:
        history_block = (
            "\n\n=== HISTÓRICO DA CONVERSA (em ordem cronológica) ===\n"
            + "\n".join(history_lines)
            + "\n=== FIM DO HISTÓRICO ===\n"
        )

    # ===== RASTREADOR DE ESTADO (evita repetir perguntas) =====
    lead = await db.leads.find_one({"owner_id": owner_id, "contact_id": contact.get("id")}, {"_id": 0})
    import re as _re
    full_history_text = " ".join(
        (m.get("text") or "").lower() for m in last_msgs if not m.get("from_me")
    )
    # Historico da SECRETARIA (ultimas 3 mensagens dela) - pra evitar repetir perguntas
    sec_last3_text = " ".join(
        (m.get("text") or "").lower() for m in last_msgs[-6:] if m.get("from_me")
    )
    # Detecta que tipo de pergunta a secretaria JA fez recentemente
    asked_name = bool(_re.search(r"\b(qual.{0,6}nome|seu nome|como.{0,6}chama|seu apelido)\b", sec_last3_text))
    asked_case = bool(_re.search(r"\b(qual.{0,6}caso|o que aconteceu|sua situa[cç][ãa]o|me conta|situa[cç][ãa]o|qual.{0,6}problema|trabalhista|fam[ií]lia|inss|c[ií]vel|criminal|qual.{0,6}quest[ãa]o|qual.{0,6}d[úu]vida)\b", sec_last3_text))
    asked_prev_lawyer = bool(_re.search(r"\b(advogad[oa]|tem.{0,6}processo|j[aá].{0,6}tem.{0,6}algum|outro escrit[oó]rio)\b", sec_last3_text))
    asked_urgency = bool(_re.search(r"\b(prazo|urgente|quando.{0,15}aconteceu|faz.{0,10}tempo|notifica[cç][ãa]o|h[aá].{0,6}quant)\b", sec_last3_text))
    asked_city = bool(_re.search(r"\b(onde.{0,6}mora|qual.{0,6}cidade|sua.{0,6}cidade|de.{0,6}onde|qual.{0,6}estado|mora.{0,6}em)\b", sec_last3_text))
    asked_offered_time = bool(_re.search(r"\b(\d{1,2}h|\d{1,2}:\d{2}|amanh[ãa].{0,10}\d|hoje.{0,10}\d|manh[ãa].{0,10}ou.{0,10}tarde|horario|hor[aá]rio)\b", sec_last3_text))

    mentioned_city = bool(_re.search(
        r"\b(moro|resido|sou de|estou em|cidade|goian|bras[ií]lia|s[ãa]o paulo|rio|belo horizonte|curitiba|salvador|recife|fortaleza|manaus|bel[eé]m|porto alegre|florianopolis|minas|pernambuco|bahia|paran[aá]|santa catarina|rio grande|cear[aá]|par[aá]|maranh[ãa]o|amazonas|rond[oô]nia|acre|roraima|amap[aá]|mato grosso|goi[aá]s|tocantins|distrito federal|df|sp|rj|mg|rs|pr|sc|ba|pe|ce|am|pa|ma|es|al|se|pb|rn|pi|to|mt|ms|go|df)\b",
        full_history_text))
    mentioned_prev_lawyer = bool(_re.search(
        r"\b(advogad[oa]|processo|j[aá] tive|tenho advogad|outro escrit[oó]rio|ningu[eé]m assumiu|n[ãa]o tenho advogad)\b",
        full_history_text))
    mentioned_timeframe = bool(_re.search(
        r"\b(ontem|hoje|semana|m[eê]s|dia|data|ano|\d+\s*(dias?|semanas?|meses|anos)|faz\s+\d+|ha\s+\d+|h[aá]\s+\d+|recente|passad[ao])\b",
        full_history_text))
    mentioned_doc = bool(_re.search(
        r"\b(rescis[ãa]o|cnis|ctps|carta|contrato|notifica[cç][ãa]o|boleto|extrato|rg|cnh|cpf|certid[ãa]o|senten[çc]a|peti[cç][ãa]o|foto|imagem|documento)\b",
        full_history_text))
    sec_msgs = [m for m in last_msgs if m.get("from_me")]
    turns_done = len(sec_msgs)

    # Detecta se a ULTIMA msg do cliente foi vaga/curta (ok, sim, ta bom, etc)
    last_client_text = ""
    for m in reversed(last_msgs):
        if not m.get("from_me"):
            last_client_text = (m.get("text") or "").strip().lower()
            break
    is_vague_reply = bool(_re.match(
        r"^(ok|okay|sim|n[aã]o|ta|t[aá] bom|tudo bem|entendi|certo|uhum|aham|blz|beleza|claro)\.?!?$",
        last_client_text
    ))

    # Bloco DURO listando tudo que ja foi perguntado recentemente
    recent_asked = []
    if asked_name: recent_asked.append("NOME do cliente")
    if asked_case: recent_asked.append("QUAL É O CASO / área jurídica")
    if asked_prev_lawyer: recent_asked.append("se JÁ TEVE ADVOGADO(A) antes / se tem processo em andamento")
    if asked_urgency: recent_asked.append("PRAZO / urgência / quando aconteceu")
    if asked_city: recent_asked.append("CIDADE / estado onde mora")
    if asked_offered_time: recent_asked.append("HORÁRIO de consulta (já ofereceu horário)")

    tracker_lines = []
    if recent_asked:
        tracker_lines.append("=== 🚫 PERGUNTAS QUE VOCÊ JÁ FEZ NAS ÚLTIMAS MENSAGENS (PROIBIDO REPETIR) ===")
        for q in recent_asked:
            tracker_lines.append(f"🚫 VOCÊ JÁ PERGUNTOU: {q}")
        tracker_lines.append("")
        tracker_lines.append("❌ É ABSOLUTAMENTE PROIBIDO voltar a fazer qualquer uma das perguntas acima,")
        tracker_lines.append("   mesmo reformulada de jeitos diferentes. Se fizer, vai parecer amador.")
        tracker_lines.append("")

    if is_vague_reply:
        tracker_lines.append("=== ⚠️ CLIENTE DEU RESPOSTA VAGA ('ok', 'sim', 'tá bom') ===")
        tracker_lines.append("NÃO repita sua última pergunta. AVANCE pro próximo passo do roteiro.")
        tracker_lines.append("Melhor resposta: ir direto PROPOR horário de consulta.")
        tracker_lines.append("Exemplo: 'Perfeito! Deixa eu já te encaixar na agenda da Dra. Kênia: amanhã às 10h ou às 15h, qual prefere?'")
        tracker_lines.append("")

    tracker_lines.append("=== ✅ DADOS JÁ COLETADOS (NÃO pergunte de novo) ===")
    if not is_phone_only:
        tracker_lines.append(f"✔ Nome: {contact_name}")
    else:
        tracker_lines.append("✘ Nome: AINDA NÃO sabe — pergunte primeiro se ainda não foi feito")
    if lead and lead.get("case_type") and lead.get("case_type") != "Outro":
        tracker_lines.append(f"✔ Área jurídica: {lead['case_type']}")
    else:
        tracker_lines.append("✘ Área jurídica: NÃO identificada — descubra (trabalhista? família? INSS? cível?)")
    if lead and lead.get("urgency") in ("alta", "critica"):
        tracker_lines.append(f"✔ Urgência: {lead['urgency']} — CASO QUENTE, ofereça horário HOJE OU AMANHÃ")
    elif mentioned_timeframe:
        tracker_lines.append("✔ Urgência/prazo: cliente já mencionou (leia histórico)")
    else:
        tracker_lines.append("✘ Urgência/prazo: NÃO mencionado — pergunte 'tem prazo correndo?' ou 'quando aconteceu?'")
    tracker_lines.append(("✔" if mentioned_city else "✘") + " Cidade/estado: " + ("mencionada" if mentioned_city else "NÃO sabe"))
    tracker_lines.append(("✔" if mentioned_prev_lawyer else "✘") + " Teve advogado(a) antes: " + ("cliente falou" if mentioned_prev_lawyer else "NÃO perguntado"))
    if mentioned_doc:
        tracker_lines.append("✔ Documentos: cliente mencionou/enviou — confirme análise e SIGA pra agendar")

    # Detecta se conversa menciona previdência rural / INSS / aposentadoria
    # e injeta a base de conhecimento rural 2026 no prompt da secretaria.
    full_conv = (full_history_text + " " + incoming_text.lower())
    is_rural_theme = bool(_re.search(
        r"\b(rural|rocinha|lavoura|ro[çc]a|segurado especial|pescador|agricultor|"
        r"agricultura familiar|produtor rural|aposentadoria rural|aposentadoria h[ií]brida|"
        r"idade rural|ca[çc]ador|pensao rural|pens[ãa]o rural|pre[vi]denci[aá]rio|inss|"
        r"aposent|pens[ãa]o por morte|cnis|bpc|loas|invalidez|incapacidade|beneficio)\b",
        full_conv
    ))
    kb_hint = ""
    if is_rural_theme:
        kb_hint = (
            "\n\n=== 📚 CASO PREVIDENCIÁRIO/RURAL DETECTADO — use conhecimento atualizado 2026 ===\n"
            "A cliente está tratando de tema previdenciário rural. Você, como secretária,"
            " NÃO dá parecer jurídico, mas pode:\n"
            "1. Demonstrar que VOCÊ conhece os prazos (ex: '90 dias para pensão por morte retroativa')\n"
            "2. Perguntar o ESSENCIAL para a Dra. já abrir o caso: tempo de atividade rural,"
            " documentos disponíveis (DAP, CAR, certidão casamento rural, sindicato), idade atual,"
            " data do óbito (se pensão) ou se benefício foi negado.\n"
            "3. Informe que a Dra. Kênia é especialista em direito previdenciário rural e que"
            " tem muita experiência com segurado especial e aposentadoria híbrida.\n\n"
            "CONHECIMENTO TÉCNICO ATUALIZADO (só referência, NÃO dê parecer):\n"
            + RURAL_PREVIDENCIA_KB_2026
        )

    have_essentials = (
        (not is_phone_only) and lead and lead.get("case_type") not in (None, "Outro")
    )
    if have_essentials or turns_done >= 3:
        tracker_lines.append("")
        tracker_lines.append("🎯 PRÓXIMO PASSO: OFEREÇA 2 HORÁRIOS CONCRETOS AGORA. Não faça mais perguntas de diagnóstico.")
        tracker_lines.append("   Exemplo: 'Pode ser amanhã às 10h ou às 15h, qual fica melhor pra você?'")
    elif is_phone_only:
        tracker_lines.append("")
        tracker_lines.append("🎯 PRÓXIMO PASSO: pergunte o NOME do cliente (só isso).")
    elif not lead or lead.get("case_type") in (None, "Outro"):
        tracker_lines.append("")
        tracker_lines.append("🎯 PRÓXIMO PASSO: descubra QUAL É O CASO (trabalhista/família/INSS/cível?). Uma pergunta só.")
    else:
        tracker_lines.append("")
        tracker_lines.append("🎯 PRÓXIMO PASSO: pergunte sobre PRAZO/URGÊNCIA (UMA pergunta).")
    tracker_lines.append("=== FIM DOS DADOS COLETADOS ===")
    tracker_lines.append(
        "\n⚠️ REGRA DE OURO ABSOLUTA:\n"
        "1. Leia o bloco 🚫 acima. NUNCA repita pergunta que você JÁ FEZ.\n"
        "2. Se o cliente respondeu vago ('ok', 'tá bom', 'entendi', 'sim'), AVANCE pro próximo passo — NÃO insista na mesma pergunta.\n"
        "3. Se você já ofereceu horário e o cliente ainda não confirmou, diga: 'te enviei duas opções ali em cima — qual prefere?'\n"
        "4. A cada turno avance UM passo. Se ficou 2 turnos na mesma fase, PULE pra frente: ofereça horário."
    )
    tracker_block = "\n\n" + "\n".join(tracker_lines)

    # bot_prompt: se vazio, "keep", ou muito curto (<50 chars) usa o KENIA_DEFAULT_PROMPT
    raw_prompt = (cfg.get("bot_prompt") or "").strip()
    if (not raw_prompt) or len(raw_prompt) < 50 or raw_prompt.lower() in {"keep", "default", "padrao", "padrão"}:
        base_prompt = KENIA_DEFAULT_PROMPT
    else:
        base_prompt = raw_prompt

    # Monta bloco com HORARIOS OCUPADOS da Dra nos proximos 7 dias
    # pra secretaria sugerir so horarios livres
    from datetime import timezone as _tz
    br_tz = _tz(timedelta(hours=-3))
    now_br = datetime.now(br_tz)
    upcoming = await db.appointments.find(
        {"owner_id": owner_id, "status": {"$ne": "cancelado"},
         "starts_at": {"$gte": now_br.isoformat(), "$lte": (now_br + timedelta(days=7)).isoformat()}},
        {"_id": 0, "starts_at": 1, "client_name": 1, "duration_min": 1}
    ).sort("starts_at", 1).to_list(50)
    busy_lines = []
    for a in upcoming:
        try:
            dt = datetime.fromisoformat(a["starts_at"])
            busy_lines.append(f"- {dt.astimezone(br_tz).strftime('%a %d/%m às %Hh%M')} (já marcado com {a.get('client_name','cliente')})")
        except Exception:
            pass
    agenda_block = (
        "\n\n=== AGENDA DA DRA. KÊNIA (próximos 7 dias — use SÓ horários livres) ===\n"
        + (("\n".join(busy_lines) + "\n") if busy_lines else "Nenhum compromisso marcado nos próximos 7 dias — todos horários 09h–18h estão livres.\n")
        + "HORÁRIOS COMERCIAIS DA DRA: SEGUNDA a SEXTA, 09h–12h e 14h–18h (fuso Brasília UTC-3).\n"
        + "🚫 NUNCA ofereça horário em SÁBADO ou DOMINGO — a Dra. Kênia não atende fim-de-semana.\n"
        + "   Se o cliente pedir sábado/domingo, diga: 'preciso falar com a Dra. Kênia primeiro"
        + " sobre esse horário. Se preferir um dia de semana, posso te encaixar agora —"
        + " que tal {próximo dia útil} às 10h ou 15h?'\n"
        + "REGRA: Antes de sugerir um horário, confirme que NÃO está na lista de ocupados acima. "
        + "Prefira oferecer 2 opções concretas: uma de manhã e uma à tarde. "
        + "Ex: 'pode ser amanhã às 10h ou às 15h, qual prefere?' (só dias úteis)."
    )

    # System prompt: CRITICAL RULES vão no TOPO (onde GPT presta mais atenção),
    # depois base_prompt (persona), depois agenda/histórico.
    critical_top = (
        "⚠️⚠️⚠️ REGRAS CRÍTICAS (LEIA ISTO ANTES DE RESPONDER) ⚠️⚠️⚠️\n"
        + tracker_block.strip()
        + "\n\n⚠️⚠️⚠️ FIM DAS REGRAS CRÍTICAS ⚠️⚠️⚠️\n\n"
    )
    system_prompt = critical_top + base_prompt + style_hint + name_hint + kb_hint + agenda_block + history_block
    session = f"wa-bot-{contact['id']}"
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY, session_id=session,
            system_message=system_prompt,
        ).with_model("openai", "gpt-4o-mini")
        reply = await chat.send_message(UserMessage(text=incoming_text))
    except Exception:
        log.exception("bot reply failed")
        return None
    if not reply:
        return None

    # Tenta extrair nome se o cliente acabou de se apresentar
    # (ex.: "sou Carlos", "meu nome e Maria", "aqui e Joao")
    if is_phone_only:
        import re
        STOPWORDS = {
            "do", "da", "no", "na", "com", "em", "sem", "hoje", "ontem", "agora",
            "aqui", "ali", "muito", "pouco", "demitido", "casado", "solteiro",
            "preocupado", "perdido", "advogado", "cliente", "trabalho", "emprego",
        }
        patterns = [
            (r"\bmeu nome (?:é|eh|e)\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)?)", True),
            (r"\bme chamo\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)?)", True),
            (r"\baqui (?:é|e|eh)\s+(?:o|a)?\s*([A-Za-zÀ-ÿ]+)", True),
            (r"\bsou\s+(?:o|a)\s+([A-Za-zÀ-ÿ]+)", True),
            (r"\bsou\s+([A-Z][a-záéíóúâêôãõç]+(?:\s+[A-Z][a-záéíóúâêôãõç]+)?)", False),
        ]
        for pat, ci in patterns:
            m = re.search(pat, incoming_text, flags=re.IGNORECASE if ci else 0)
            if not m:
                continue
            novo_nome = m.group(1).strip().title()
            primeiro = novo_nome.split()[0].lower()
            if primeiro in STOPWORDS or len(novo_nome) < 2 or len(novo_nome) > 40:
                continue
            await db.whatsapp_contacts.update_one(
                {"id": contact["id"]},
                {"$set": {"name": novo_nome}},
            )
            contact["name"] = novo_nome
            break

    # Valida que temos um telefone valido antes de tentar enviar
    target_phone = (contact.get("phone") or contact.get("phone_normalized") or "").strip()
    # For Baileys, prefer routing to the ORIGINAL remoteJid we received from
    # (handles @lid contacts: rebuilding ${digits}@s.whatsapp.net would create
    # a phantom "new number" chat instead of replying to the original sender).
    target_jid = (contact.get("wa_jid") or "").strip()
    invalid_phone = (
        (not target_phone and not target_jid)
        or "@" in target_phone
        or "-group" in target_phone
        or target_phone.lower() in ("status", "broadcast")
        or (not target_jid and not any(ch.isdigit() for ch in target_phone))
    )
    if invalid_phone:
        log.warning(f"bot skip send: invalid phone/jid for contact {contact.get('id')}: phone='{target_phone}' jid='{target_jid}'")
        await db.whatsapp_messages.insert_one({
            "id": str(uuid.uuid4()), "owner_id": owner_id, "contact_id": contact["id"],
            "text": reply, "from_me": True, "provider": prov.name, "bot": True,
            "bot_persona": "Kênia Garcia",
            "delivered": False, "send_error": "invalid-phone-skipped",
            "created_at": now_iso(),
        })
        return reply
    # ===== DECISAO TEXTO vs AUDIO vs AMBOS =====
    # Modo de voz vem do config; default text_and_audio (opcao B).
    voice_mode = (cfg.get("bot_voice_mode") or "text_and_audio").lower()
    voice_name = (cfg.get("bot_voice") or "nova").lower()
    # Flag por contato: prefer_audio=True quando o cliente comecou com voz
    # OU mostrou sinais sutis de baixo letramento.
    prefer_audio = bool(contact.get("prefer_audio"))

    # Detector sutil de letramento: respostas curtas com erros recorrentes
    # (sem acentos, palavras cortadas, multiplos espacos, sem pontuacao)
    # NUNCA expor isso ao cliente — apenas usa internamente para decidir voz.
    if not prefer_audio and incoming_text:
        import re as _re_lit
        t = incoming_text.strip().lower()
        words = t.split()
        n_words = len(words)
        # heuristicas conservadoras (combinadas):
        no_accents = bool(_re_lit.search(r"[a-z]", t)) and not _re_lit.search(r"[áéíóúâêôãõç]", t)
        # palavras tipicas com erro/abreviacao
        typo_hits = sum(1 for w in words if w in {
            "vc","vcs","tb","tbm","blz","msm","tava","tava","mt","muinto","oce","sera","oq",
            "cuncerteza","tudim","ten","obg","pq","td","fess","tipu","ki","si","nois","ngc","mim deve",
            "advogada e","menssagem","mensajem","apusentadoria","bensão","comigu","ouvi de meu","ouve",
        })
        # 3+ erros simples + msg curta + sem acento => provavel baixo letramento
        if n_words >= 2 and n_words <= 12 and (typo_hits >= 2 or (no_accents and typo_hits >= 1)):
            prefer_audio = True

    if incoming_was_audio:
        prefer_audio = True

    # Persiste flag pra proximas mensagens manterem o mesmo tratamento
    if prefer_audio and not contact.get("prefer_audio"):
        try:
            await db.whatsapp_contacts.update_one(
                {"id": contact["id"]}, {"$set": {"prefer_audio": True}}
            )
            contact["prefer_audio"] = True
        except Exception:
            pass

    # === HUMANIZAÇÃO: simula tempo de leitura + digitação humana ===
    # Antes de enviar a resposta, a Nislainy "está lendo a sua mensagem" e
    # "digitando". Isso evita aquela cara de bot que responde em 200ms.
    # Cálculo: tempo proporcional ao tamanho da mensagem que ela vai mandar.
    # - Reading time: 50 chars/seg da mensagem do cliente
    # - Typing time: 30 chars/seg (~120 WPM, ritmo de pessoa boa de teclado)
    # - Pausa base de 1-2s (a Nislainy pega o celular, lê, pensa)
    # - Cap em 10s pra não irritar
    import random as _rand
    incoming_len = len(incoming_text or "")
    outgoing_len = len(reply or "")
    reading_s = min(incoming_len / 50.0, 3.0)
    typing_s = min(outgoing_len / 30.0, 6.0)
    base_pause = _rand.uniform(0.8, 1.8)
    human_delay = min(reading_s + typing_s + base_pause, 10.0)
    # Para áudio: a "fala" é mais rápida que digitar, então só aplica metade
    if voice_mode in ("audio_only",) or (voice_mode == "auto" and prefer_audio):
        human_delay = min(human_delay * 0.6, 6.0)
    try:
        await asyncio.sleep(human_delay)
    except Exception:
        pass

    # Resolve a decisao final (send_text, send_audio):
    if voice_mode == "text_only":
        send_text, send_audio = True, False
    elif voice_mode == "audio_only":
        send_text, send_audio = False, True
    elif voice_mode == "auto":
        # auto: audio quando o contato prefere; senao texto puro
        send_text = not prefer_audio
        send_audio = prefer_audio
    else:  # text_and_audio (default opcao B)
        send_text = True
        send_audio = bool(prefer_audio or isinstance(prov, BaileysProvider) is False or True is False) or True
        # Default: SEMPRE manda os dois (cliente que prefere audio escuta;
        # quem prefere ler, le; quem usa fone, escuta — UX 'audio + texto' vence)
        send_audio = True

    # Gera audio TTS se for o caso
    audio_b64 = None
    if send_audio:
        try:
            audio_bytes = await _tts_generate(reply[:4000], cfg)
            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        except Exception:
            log.exception("[bot] TTS gen failed — fallback para texto")
            send_audio = False
            send_text = True  # garante que algo seja entregue

    try:
        # 1) ENVIO DE TEXTO (se aplicavel)
        send_meta = {"delivered": False}
        if send_text:
            if isinstance(prov, BaileysProvider):
                send_res = await prov.send_text(target_phone, reply, jid=target_jid or None)
            else:
                send_res = await prov.send_text(target_phone, reply)
            delivered_text = bool(send_res.get("ok"))
            send_meta = {
                "delivered": delivered_text,
                "provider_response": send_res.get("response"),
                "provider_status": send_res.get("status"),
            }
            if not delivered_text:
                log.warning(
                    f"bot reply NOT delivered to {target_phone}: {str(send_res)[:300]}"
                )

        # 2) ENVIO DE AUDIO (se aplicavel) — atualmente so Baileys suporta TTS direto
        audio_meta = {}
        if send_audio and audio_b64:
            if isinstance(prov, BaileysProvider):
                try:
                    expected_token = (
                        os.environ.get("BAILEYS_INTERNAL_TOKEN")
                        or "legalflow-baileys-2026"
                    )
                    base_url = os.environ.get("BAILEYS_URL", "http://localhost:8002")
                    payload_audio = {
                        "phone": target_phone,
                        "audio_base64": audio_b64,
                        "mime": "audio/mp4",
                    }
                    if target_jid:
                        payload_audio["jid"] = target_jid
                    async with httpx.AsyncClient(timeout=30.0) as c:
                        r = await c.post(
                            f"{base_url}/send-audio",
                            json=payload_audio,
                            headers={"x-internal-token": expected_token},
                        )
                        audio_meta = {
                            "audio_delivered": r.status_code == 200,
                            "audio_status": r.status_code,
                        }
                        if r.status_code != 200:
                            log.warning(f"bot AUDIO not delivered: {r.text[:200]}")
                except Exception:
                    log.exception("bot send-audio failed")
                    audio_meta = {"audio_delivered": False}
            else:
                # Outros providers ainda nao suportam audio (TODO)
                audio_meta = {"audio_delivered": False, "audio_status": "provider_unsupported"}

        # marca delivered como True se pelo menos um dos dois entregou
        send_meta["delivered"] = bool(send_meta.get("delivered")) or bool(audio_meta.get("audio_delivered"))
        send_meta.update(audio_meta)
    except Exception as e:
        log.exception("bot send failed")
        send_meta = {"delivered": False, "send_error": str(e)[:300]}

    msg_doc = {
        "id": str(uuid.uuid4()), "owner_id": owner_id, "contact_id": contact["id"],
        "text": reply, "from_me": True, "provider": prov.name, "bot": True,
        "bot_persona": "Kênia Garcia",
        "voice_mode_used": ("audio" if (send_audio and not send_text) else
                            "text_audio" if (send_audio and send_text) else "text"),
        "created_at": now_iso(),
        **send_meta,
    }
    await db.whatsapp_messages.insert_one(msg_doc)
    await db.whatsapp_contacts.update_one(
        {"id": contact["id"]},
        {"$set": {"last_message": reply, "last_message_at": now_iso()}},
    )

    # === AGENDAMENTO AUTOMATICO ===
    # Se o cliente confirmou um horario, cria appointment + envia meet link.
    # Se houver CONFLITO com outro compromisso da Dra, envia mensagem pedindo novo horario.
    try:
        appt = await _maybe_create_appointment_from_message(
            owner_id, contact, incoming_text, last_msgs,
        )
        if appt and prov:
            if appt.get("weekend"):
                # Agendamento em sabado/domingo: a Dra. NAO atende nesses dias.
                # Secretaria avisa que vai checar com a doutora.
                confirm_text = (
                    f"Hm, {contact.get('name') or 'prezad(a)'}... 🙏 A Dra. Kênia "
                    f"não costuma atender aos sábados e domingos. "
                    f"Mas deixa eu falar com ela primeiro sobre esse horário "
                    f"({appt['human_when']}) e já te retorno aqui! "
                    f"Se preferir um dia de semana, posso te encaixar na agenda dela — "
                    f"amanhã ou depois de amanhã, qual fica melhor?"
                )
                appointment_id = None
                is_confirm = False
                is_conflict = False
                is_weekend = True
            elif appt.get("conflict"):
                # Horario conflitante — avisar cliente e pedir pra escolher outro
                confirm_text = (
                    f"Opa, {contact.get('name') or 'prezado(a)'}! 😕 "
                    f"Nesse horário ({appt['human_when']}) a Dra. Kênia já está com "
                    f"outro atendimento marcado ({appt['conflict_with']}). "
                    f"Posso te sugerir outro horário? Tem um livre de manhã ou à tarde — qual prefere?"
                )
                appointment_id = None
                is_confirm = False
                is_conflict = True
                is_weekend = False
            else:
                confirm_text = (
                    f"✅ Prontinho, {contact.get('name') or 'tudo certo'}! Sua consulta com a Dra. Kênia ficou agendada:\n\n"
                    f"📅 {appt['human_when']}\n"
                    f"🎥 Link: {appt['meeting_link']}\n\n"
                    f"A Dra. Kênia vai te chamar nesse horário pelo Google Meet. Qualquer coisa me avisa por aqui!"
                )
                appointment_id = appt["id"]
                is_confirm = True
                is_conflict = False
                is_weekend = False
            try:
                if isinstance(prov, BaileysProvider):
                    cr = await prov.send_text(target_phone, confirm_text, jid=target_jid or None)
                else:
                    cr = await prov.send_text(target_phone, confirm_text)
                await db.whatsapp_messages.insert_one({
                    "id": str(uuid.uuid4()), "owner_id": owner_id,
                    "contact_id": contact["id"], "text": confirm_text,
                    "from_me": True, "provider": prov.name, "bot": True,
                    "bot_persona": "Kênia Garcia", "is_appointment_confirm": is_confirm,
                    "is_appointment_conflict": is_conflict,
                    "is_appointment_weekend": is_weekend,
                    "appointment_id": appointment_id,
                    "delivered": bool(cr.get("ok")),
                    "provider_response": cr.get("response"),
                    "provider_status": cr.get("status"),
                    "created_at": now_iso(),
                })
                await db.whatsapp_contacts.update_one(
                    {"id": contact["id"]},
                    {"$set": {"last_message": confirm_text, "last_message_at": now_iso()}},
                )
            except Exception:
                log.exception("appointment confirm send failed")
    except Exception:
        log.exception("appointment auto-create failed")

    return reply


def detect_sinestesic(text: str) -> Optional[str]:
    """Classifica o estilo VAK (Visual / Auditivo / Cinestesico) por palavras-chave.
    Usado para adaptar as respostas do bot ao perfil sensorial do cliente."""
    t = (text or "").lower()
    visual = [
        "vejo", "veja", "mostra", "imagine", "imagina", "mostrar", "visual",
        "enxerg", "olhar", "ver ", "olho", "claro", "escuro", "brilh",
        "foco", "foca", "perspectiva", "panorama", "luz ", "cor ", "cores",
        "pareceu", "pareço", "pinta", "paisagem",
    ]
    auditivo = [
        "ouvi", "escute", "escuta", "soa", "som ", "falar", "falou", "diss",
        "barulh", "silencio", "silêncio", "voz ", "vozes", "tom ", "ressoa",
        "musical", "ritmo", "ouco", "ouço", "repete", "frase", "ecoa",
    ]
    cinestesico = [
        "sinto", "sente", "senti", "peso", "pressão", "aperto", "mão", "mãos",
        "toca", "contato ", "pegar", "segur", "duro", "mole", "frio", "quente",
        "calor", "tremor", "abraco", "abraço", "respiro", "estômago", "coracao", "coração",
        "machuca", "doi", "dói", "dor ",
    ]
    vc = sum(1 for w in visual if w in t)
    ac = sum(1 for w in auditivo if w in t)
    cc = sum(1 for w in cinestesico if w in t)
    if max(vc, ac, cc) == 0:
        return None
    if vc >= ac and vc >= cc:
        return "visual"
    if ac >= cc:
        return "auditivo"
    return "cinestesico"


# Estrategias sinestesicas por perfil — usadas para adaptar a resposta do bot
SINESTESIC_STRATEGIES = {
    "visual": (
        "ESTRATÉGIA VISUAL — o cliente PROCESSA O MUNDO PELA IMAGEM. Use:\n"
        "- Verbos visuais: 'olha', 'vê', 'imagina', 'mostra', 'vamos clarear', 'enxergar a saída'.\n"
        "- Metáforas de luz e perspectiva: 'vamos colocar luz nessa situação', 'tirar isso da escuridão'.\n"
        "- Estruture mentalmente a resposta: 'primeiro... depois... no final você vai ver que...'.\n"
        "- Ofereça MOSTRAR provas/documentos: 'posso te enviar um modelo aqui pra você ver'.\n"
        "- Use emojis sutis quando apropriado (✓ 📄 ⚖️) — eles AJUDAM o visual."
    ),
    "auditivo": (
        "ESTRATÉGIA AUDITIVA — o cliente PROCESSA O MUNDO PELO SOM/PALAVRA. Use:\n"
        "- Verbos auditivos: 'ouvi', 'escute', 'fala comigo', 'me conta', 'soa como'.\n"
        "- Frases ritmadas e claras: 'eu te explico bem clarinho, passo a passo'.\n"
        "- Repita pontos-chave do que o cliente disse, com aspas: 'você disse \"sem justa causa\"...'.\n"
        "- Ofereça LIGAR: 'prefere que o Dr. te ligue agora? Em 5 minutos a gente conversa'.\n"
        "- Evite emojis em excesso — auditivos preferem palavras a símbolos."
    ),
    "cinestesico": (
        "ESTRATÉGIA CINESTÉSICA — o cliente PROCESSA O MUNDO PELA SENSAÇÃO/CORPO. Use:\n"
        "- Verbos sensoriais: 'sinto', 'entendo esse peso', 'vamos aliviar', 'tirar essa carga'.\n"
        "- Reconheça FORTEMENTE a emoção antes de qualquer ação: 'isso deve estar pesando muito em você'.\n"
        "- Linguagem mais lenta, sem pressa: respeite o tempo do cliente, NÃO empurre 3 perguntas.\n"
        "- Concretize ações em pequenos passos: 'vamos passo a passo, no seu ritmo'.\n"
        "- Ofereça SEGURANÇA: 'você não está sozinho(a), estamos juntos nisso'."
    ),
}


async def _classify_and_create_lead(owner_id: str, contact: Dict[str, Any], text: str):
    """Usa IA para classificar mensagem e criar/atualizar lead.
    Melhorias 2026-01:
    - Classifica usando HISTORICO da conversa (nao so msg atual) para acertar area/score.
    - Match de lead existente por contact_id OU phone (evita duplicados de lid vs pn).
    - Auto-promove stage baseado em score + urgencia + presenca de appointment.
    """
    # 1) Busca lead existente por contact_id (mais confiavel) ou fallback phone
    existing = await db.leads.find_one({"owner_id": owner_id, "contact_id": contact.get("id")}, {"_id": 0})
    if not existing:
        phone_digits = "".join(ch for ch in (contact.get("phone") or "") if ch.isdigit())
        suffix = phone_digits[-8:] if len(phone_digits) >= 8 else phone_digits
        if suffix:
            all_leads = await db.leads.find({"owner_id": owner_id}, {"_id": 0}).to_list(2000)
            for l in all_leads:
                lp = "".join(ch for ch in (l.get("phone") or "") if ch.isdigit())
                if lp.endswith(suffix):
                    existing = l
                    break

    # 2) Pega historico curto da conversa pra enriquecer a classificacao
    recent = await db.whatsapp_messages.find(
        {"contact_id": contact.get("id"), "owner_id": owner_id},
        {"_id": 0, "text": 1, "from_me": 1, "bot": 1, "created_at": 1},
    ).sort("created_at", -1).to_list(12)
    recent.reverse()
    hist_lines = []
    for m in recent[-10:]:
        who = "Cliente" if not m.get("from_me") else "Kenia"
        t = (m.get("text") or "").strip()
        if t: hist_lines.append(f"{who}: {t[:200]}")
    if not hist_lines or hist_lines[-1] != f"Cliente: {text[:200]}":
        hist_lines.append(f"Cliente: {text[:200]}")
    hist = "\n".join(hist_lines)

    import json as _json
    classify_prompt = (
        "Classifique o lead de um escritório de advocacia baseado na CONVERSA INTEIRA abaixo.\n"
        "Leve em conta tudo que o cliente revelou até agora (nao so a ultima mensagem).\n\n"
        f"Conversa:\n{hist}\n\n"
        "Responda APENAS com um JSON válido no formato:\n"
        "{\n"
        '  "area": "Trabalhista|Família|Previdenciário/INSS|Cível|Criminal|Empresarial|Tributário|Bancário|Consumidor|Outro",\n'
        '  "urgencia": "baixa|media|alta|critica",\n'
        '  "score": 0-100 (reflita: clareza do problema + urgencia + potencial financeiro),\n'
        '  "resumo": "resumo do caso em 1 linha tecnica",\n'
        '  "tags": ["2-5 tags juridicas relevantes"],\n'
        '  "pronto_consulta": true/false  // true se o cliente ja forneceu nome + tipo do caso + consegue tomar decisao\n'
        "}"
    )
    area = "Outro"
    urgencia = "media"
    score = 50
    resumo = text[:100]
    tags: List[str] = []
    pronto_consulta = False
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"classify-{contact['id']}",
            system_message="Você é um classificador sênior de leads jurídicos. Responda SEMPRE em JSON válido, sem markdown.",
        ).with_model("openai", "gpt-4o-mini")
        raw = await chat.send_message(UserMessage(text=classify_prompt))
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"): raw = raw[4:]
        data = _json.loads(raw)
        area = str(data.get("area", area))
        urgencia = str(data.get("urgencia", urgencia)).lower()
        score = int(data.get("score", score))
        resumo = str(data.get("resumo", resumo))
        tags = data.get("tags") or []
        if not isinstance(tags, list): tags = []
        pronto_consulta = bool(data.get("pronto_consulta", False))
    except Exception:
        log.exception("classify parse failed")

    # 3) Auto-promocao de stage
    # regra: appointment criado -> qualificado; pronto_consulta + score>=75 -> qualificado;
    # score>=50 -> em_contato; senao mantem novos_leads
    has_appt = await db.appointments.count_documents({"owner_id": owner_id, "contact_id": contact.get("id")}) > 0
    if has_appt:
        new_stage = "qualificado"
    elif pronto_consulta and score >= 75:
        new_stage = "qualificado"
    elif score >= 50 and (existing and existing.get("stage") not in ("qualificado", "convertido", "em_negociacao")):
        new_stage = "em_contato"
    elif not existing:
        new_stage = "novos_leads"
    else:
        new_stage = existing.get("stage") or "novos_leads"

    if existing:
        await db.leads.update_one(
            {"id": existing["id"]},
            {"$set": {
                "case_type": area,
                "urgency": urgencia,
                "score": score,
                "description": resumo,
                "tags": tags,
                "stage": new_stage,
                "updated_at": now_iso(),
            }},
        )
        return existing["id"]
    else:
        lead_id = str(uuid.uuid4())
        await db.leads.insert_one({
            "id": lead_id, "owner_id": owner_id,
            "name": contact.get("name", "Lead WhatsApp"),
            "phone": contact.get("phone", ""),
            "email": None,
            "case_type": area,
            "description": resumo,
            "source": "WhatsApp",
            "stage": new_stage,
            "score": score,
            "urgency": urgencia,
            "notes": f"Classificado por IA (contexto conversa completa).",
            "tags": tags,
            "contact_id": contact.get("id"),
            "created_at": now_iso(), "updated_at": now_iso(),
        })
        return lead_id

@api_router.post("/whatsapp/webhook/zapi")
async def zapi_webhook(request: Request):
    """Z-API envia notificações de mensagens recebidas e status."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    log.info(f"[Z-API webhook] {body}")
    # Z-API payloads podem variar: ReceivedCallback, MessageStatusCallback, etc.
    event_type = body.get("type") or body.get("event")
    # Processa status de entrega (SENT/RECEIVED/READ/PLAYED) para atualizar as msgs enviadas
    if event_type == "MessageStatusCallback":
        status = str(body.get("status") or "").upper()
        ids = body.get("ids") or []
        if not ids and body.get("messageId"):
            ids = [body["messageId"]]
        if ids:
            mapping = {
                "SENT": "sent", "RECEIVED": "received",
                "READ": "read", "READ_BY_ME": "read_by_me",
                "PLAYED": "played", "FAILED": "failed",
            }
            wa_status = mapping.get(status, status.lower())
            # Atualiza mensagens que deram match no provider_response.messageId
            await db.whatsapp_messages.update_many(
                {"provider_response.messageId": {"$in": ids}},
                {"$set": {"wa_status": wa_status, "wa_status_at": now_iso()}},
            )
        return {"ok": True, "status_updated": status, "ids": ids}
    # Ignorar outros status updates de presenca
    if event_type in ("PresenceChatCallback", "ChatPresenceCallback"):
        return {"ok": True, "ignored": True, "type": event_type}
    if body.get("fromMe") or body.get("isFromMe"):
        return {"ok": True, "from_me": True}
    # Ignorar mensagens de grupo (focam no atendimento 1:1)
    if body.get("isGroup") or body.get("isNewsletter"):
        return {"ok": True, "ignored": True, "reason": "group-or-newsletter"}
    # Defensivo: phone com '@g.us', '-group', '@broadcast' ou 'status' e invalido
    raw_phone = str(body.get("phone") or body.get("from") or "")
    rp_lower = raw_phone.lower()
    if ("@g.us" in rp_lower or "-group" in rp_lower
            or rp_lower.endswith("@broadcast")
            or rp_lower == "status"
            or "status@broadcast" in rp_lower):
        return {"ok": True, "ignored": True, "reason": "group-broadcast-or-status"}
    phone = body.get("phone") or body.get("from") or ""
    # remove @s.whatsapp.net if present
    if "@" in str(phone):
        phone = str(phone).split("@")[0]
    text = ""
    # text message formats
    if isinstance(body.get("text"), dict):
        text = body["text"].get("message") or body["text"].get("body") or ""
    elif isinstance(body.get("text"), str):
        text = body["text"]
    elif isinstance(body.get("message"), dict):
        m = body["message"]
        text = (
            m.get("conversation")
            or m.get("text")
            or (m.get("extendedTextMessage") or {}).get("text", "")
            or ""
        )
    elif isinstance(body.get("message"), str):
        text = body.get("message") or ""
    elif isinstance(body.get("body"), str):
        text = body.get("body")
    if not text and body.get("image"):
        text = "[Imagem recebida]"
        # mark visual preference
        await db.whatsapp_contacts.update_one(
            {"phone_normalized": normalize_phone(phone)},
            {"$set": {"sinestesic_style": "visual"}},
        )
    if not text and body.get("audio"):
        text = "[Áudio recebido — cliente prefere comunicação por áudio]"
        await db.whatsapp_contacts.update_one(
            {"phone_normalized": normalize_phone(phone)},
            {"$set": {"sinestesic_style": "auditivo", "prefers_audio": True}},
        )
    if not text and body.get("document"):
        text = "[Documento recebido]"

    name = (
        body.get("senderName")
        or body.get("chatName")
        or body.get("pushName")
        or body.get("notifyName")
        or phone
    )
    if not phone or not text:
        return {"ok": True, "ignored": True, "reason": "no-phone-or-text"}
    # Resolve o owner buscando a whatsapp_config Z-API; se houver instanceId no payload, usa
    inst_hint = body.get("instanceId") or body.get("instance_id") or body.get("instance")
    owner_id = await _resolve_owner_for_provider("zapi", inst_hint)
    if not owner_id:
        return {"ok": True, "noowner": True}
    contact, _ = await _save_incoming_message(owner_id, phone, name, text, "zapi")
    # auto-classify lead (IA)
    try:
        await _classify_and_create_lead(owner_id, contact, text)
    except Exception:
        log.exception("classify error")
    await _maybe_autorespond(owner_id, contact, text)
    return {"ok": True}

@api_router.post("/whatsapp/webhook/evolution")
async def evo_webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    log.info(f"[Evolution webhook] {body}")
    data = body.get("data", {}) or body
    key = data.get("key", {}) or {}
    if key.get("fromMe"):
        return {"ok": True}
    msg = data.get("message", {}) or {}
    text = msg.get("conversation") or (msg.get("extendedTextMessage") or {}).get("text") or ""
    remote = key.get("remoteJid") or ""
    phone = remote.split("@")[0] if remote else ""
    name = data.get("pushName") or phone
    if not phone or not text:
        return {"ok": True, "ignored": True}
    owner_id = await _resolve_owner_for_provider("evolution")
    if not owner_id:
        return {"ok": True, "noowner": True}
    contact, _ = await _save_incoming_message(owner_id, phone, name, text, "evolution")
    try:
        await _classify_and_create_lead(owner_id, contact, text)
    except Exception:
        log.exception("classify error")
    await _maybe_autorespond(owner_id, contact, text)
    return {"ok": True}

@api_router.post("/whatsapp/webhook/baileys")
async def baileys_webhook(request: Request):
    """Recebe mensagens encaminhadas pelo sidecar Node.js Baileys."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    log.info(f"[Baileys webhook] {str(body)[:300]}")
    # Validacao do token interno (anti-spoof)
    # IMPORTANTE: este default DEVE ser igual ao do _spawn_baileys() (linha ~3584)
    # e ao default do baileys-service/server.js. Antes estava
    # "espirito-santo-baileys-2026" e divergia de "legalflow-baileys-2026",
    # fazendo TODO webhook do Baileys ser rejeitado com 401 unauthorized -> nem
    # mensagens recebidas apareciam, nem o bot respondia.
    expected = os.environ.get("BAILEYS_INTERNAL_TOKEN") or "legalflow-baileys-2026"
    if (body.get("token") or "") != expected:
        return {"ok": False, "error": "unauthorized"}
    phone = body.get("phone") or ""
    text = body.get("text") or ""
    name = body.get("name") or phone
    wa_jid = body.get("jid") or ""
    wa_phone_jid = body.get("phone_jid") or ""
    is_lid = bool(body.get("is_lid"))
    audio_b64 = body.get("audio_base64")
    audio_mime = body.get("audio_mime") or "audio/ogg"
    image_b64 = body.get("image_base64")
    image_mime = body.get("image_mime") or "image/jpeg"
    image_caption = body.get("image_caption") or ""

    # Se recebemos IMAGEM (documento, RG, contrato, rescisao), analisa com GPT-4o Vision
    doc_analysis = None
    if image_b64:
        try:
            from emergentintegrations.llm.chat import LlmChat as _Chat, UserMessage as _UM, ImageContent as _IC
            vision_sys = (
                "Voce e uma analista juridica veterana atendendo no WhatsApp. "
                "Recebeu uma FOTO DE DOCUMENTO enviada pelo cliente. "
                "Tarefa: (1) identificar QUE documento e (CTPS, rescisao, contrato, "
                "RG/CNH, extrato, boleto, carta, e-mail, foto de tela, ficha medica etc), "
                "(2) extrair as informacoes juridicas RELEVANTES (datas, valores, nomes, "
                "clausulas, motivos, prazos), (3) listar os pontos que merecem atencao "
                "ou podem favorecer/prejudicar o cliente, (4) concluir em 1-2 frases qual "
                "o proximo passo ideal.\n"
                "Responda em JSON valido APENAS:\n"
                "{\"tipo\":\"...\",\"extraido\":\"...\",\"pontos_de_atencao\":[\"...\"],"
                "\"conclusao\":\"...\",\"urgencia\":\"baixa|media|alta|critica\"}"
            )
            vchat = _Chat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f"vision-{uuid.uuid4().hex[:8]}",
                system_message=vision_sys,
            ).with_model("openai", "gpt-4o-mini")
            img_part = _IC(image_base64=image_b64)
            caption_note = f"\nCaption do cliente: {image_caption}" if image_caption else ""
            raw_json = await vchat.send_message(_UM(
                text=f"Analise esse documento juridico do ponto de vista trabalhista/civil/previdenciario.{caption_note}",
                file_contents=[img_part],
            ))
            try:
                import json as _json
                j = (raw_json or "").strip()
                if j.startswith("```"):
                    j = j.strip("`")
                    if j.lower().startswith("json"): j = j[4:]
                doc_analysis = _json.loads(j)
            except Exception:
                doc_analysis = {"tipo": "documento", "extraido": (raw_json or "")[:1500], "conclusao": "", "urgencia": "media", "pontos_de_atencao": []}
            log.info(f"[Baileys vision] doc analysis: {str(doc_analysis)[:200]}")
        except Exception:
            log.exception("vision analyze failed")

    # Se recebemos audio, transcreve usando Whisper
    transcribed = None
    if audio_b64:
        try:
            stt = OpenAISpeechToText(api_key=EMERGENT_LLM_KEY)
            raw = base64.b64decode(audio_b64)
            ml = (audio_mime or "").lower()
            if "mp4" in ml or "m4a" in ml: ext = "m4a"
            elif "mpeg" in ml or "mp3" in ml: ext = "mp3"
            elif "wav" in ml: ext = "wav"
            elif "webm" in ml: ext = "webm"
            else:
                ext = "webm"
            bio = io.BytesIO(raw)
            bio.name = f"voice.{ext}"
            resp = await stt.transcribe(file=bio, model="whisper-1", language="pt", response_format="json")
            transcribed = (getattr(resp, "text", None) or "").strip()
            log.info(f"[Baileys] audio transcrito: {transcribed[:100]}")
        except Exception:
            log.exception("whisper transcribe failed")
        if transcribed:
            text = transcribed

    # Monta o texto que sera persistido + alimentara o bot
    stored_text = text
    if doc_analysis:
        # 'extraido' pode vir como string OU dict (nested keys). Normaliza pra string.
        _ext_raw = doc_analysis.get("extraido", "")
        if isinstance(_ext_raw, dict):
            _ext_str = "; ".join(f"{k}: {v}" for k, v in _ext_raw.items())
        elif isinstance(_ext_raw, list):
            _ext_str = "; ".join(str(x) for x in _ext_raw)
        else:
            _ext_str = str(_ext_raw)
        _pontos = doc_analysis.get("pontos_de_atencao") or []
        if not isinstance(_pontos, list): _pontos = [str(_pontos)]
        _conclusao = str(doc_analysis.get("conclusao") or "")
        summary = (
            f"📎 [Documento: {doc_analysis.get('tipo','documento')}] "
            + (image_caption + " | " if image_caption else "")
            + (_conclusao or _ext_str[:300])
        )
        stored_text = summary
        # Alimenta o bot com contexto tecnico do documento
        text = (
            f"[O cliente enviou uma foto de '{doc_analysis.get('tipo','documento')}'. "
            f"Analise tecnica: {_ext_str[:600]}. "
            f"Conclusao: {_conclusao}. "
            f"Pontos de atencao: {', '.join(str(p) for p in _pontos)[:300]}.] "
            + (f"Legenda que o cliente escreveu: {image_caption}" if image_caption else "")
        )

    # FALLBACK: cliente enviou audio mas Whisper falhou transcrever. NAO descartar
    # — sinalizar que ele falou, persistir como audio inaudivel, e deixar o bot
    # responder em audio (o usuario vai pedir pra repetir, mas a conversa segue).
    if audio_b64 and not text:
        text = "[áudio inaudível — pedir ao cliente para repetir]"
        stored_text = "🎙️ (áudio recebido — sem transcrição)"

    if not phone or not text:
        return {"ok": True, "ignored": True}
    owner_id = await _resolve_owner_for_provider("baileys")
    if not owner_id:
        return {"ok": True, "noowner": True}
    contact, _ = await _save_incoming_message(
        owner_id, phone, name,
        (f"🎙️ (áudio) {text}" if transcribed else stored_text),
        "baileys",
        jid=wa_jid, is_lid=is_lid, phone_jid=wa_phone_jid,
    )
    try:
        await _classify_and_create_lead(owner_id, contact, text)
    except Exception:
        log.exception("classify error")
    # Marca incoming_was_audio: cliente enviou audio (mesmo que Whisper tenha
    # falhado em transcrever — o sinal de "cliente fala" e do payload audio_b64).
    reply = await _maybe_autorespond(
        owner_id, contact, text, incoming_was_audio=bool(audio_b64)
    )
    return {"ok": True, "transcribed": bool(transcribed)}


@api_router.get("/whatsapp/webhook/meta")
async def meta_verify(request: Request):
    params = request.query_params
    if params.get("hub.mode") == "subscribe":
        return int(params.get("hub.challenge", 0)) if params.get("hub.challenge", "").isdigit() else params.get("hub.challenge")
    return {"ok": True}

@api_router.post("/whatsapp/webhook/meta")
async def meta_webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    log.info(f"[Meta webhook] {body}")
    try:
        entry = (body.get("entry") or [])[0]
        change = (entry.get("changes") or [])[0]
        value = change.get("value", {})
        messages = value.get("messages") or []
        contacts = value.get("contacts") or []
        if not messages:
            return {"ok": True}
        m = messages[0]
        phone = m.get("from") or ""
        text = (m.get("text") or {}).get("body") or ""
        name = (contacts[0].get("profile") or {}).get("name", phone) if contacts else phone
        if not phone or not text:
            return {"ok": True, "ignored": True}
        first = await db.users.find_one({}, {"_id": 0, "id": 1})
        if not first:
            return {"ok": True}
        owner_id = first["id"]
        contact, _ = await _save_incoming_message(owner_id, phone, name, text, "meta")
        try:
            await _classify_and_create_lead(owner_id, contact, text)
        except Exception:
            log.exception("classify error")
        await _maybe_autorespond(owner_id, contact, text)
    except Exception:
        log.exception("meta webhook parse error")
    return {"ok": True}

# ==================== PROCESSES ====================

@api_router.post("/processes")
async def create_process(payload: ProcessCreate, current_user=Depends(get_current_user)):
    pid = str(uuid.uuid4())
    doc = {
        "id": pid, "owner_id": current_user["id"], **payload.model_dump(),
        "timeline": [{"date": now_iso(), "event": "Processo cadastrado", "type": "info"}],
        "documents": [], "created_at": now_iso(),
    }
    await db.processes.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api_router.get("/processes")
async def list_processes(current_user=Depends(get_current_user)):
    items = await db.processes.find({"owner_id": current_user["id"]}, {"_id": 0}).sort("created_at", -1).to_list(500)
    return items

@api_router.delete("/processes/{process_id}")
async def delete_process(process_id: str, current_user=Depends(get_current_user)):
    await db.processes.delete_one({"id": process_id, "owner_id": current_user["id"]})
    return {"ok": True}

# ==================== FINANCE ====================

@api_router.post("/finance/transactions")
async def create_transaction(payload: TransactionCreate, current_user=Depends(get_current_user)):
    tid = str(uuid.uuid4())
    doc = {"id": tid, "owner_id": current_user["id"], **payload.model_dump(), "created_at": now_iso()}
    await db.transactions.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api_router.get("/finance/transactions")
async def list_transactions(current_user=Depends(get_current_user)):
    items = await db.transactions.find({"owner_id": current_user["id"]}, {"_id": 0}).sort("created_at", -1).to_list(500)
    return items

@api_router.patch("/finance/transactions/{tid}")
async def update_transaction(tid: str, payload: dict, current_user=Depends(get_current_user)):
    payload.pop("_id", None); payload.pop("id", None)
    await db.transactions.update_one({"id": tid, "owner_id": current_user["id"]}, {"$set": payload})
    item = await db.transactions.find_one({"id": tid}, {"_id": 0})
    return item

@api_router.delete("/finance/transactions/{tid}")
async def delete_transaction(tid: str, current_user=Depends(get_current_user)):
    await db.transactions.delete_one({"id": tid, "owner_id": current_user["id"]})
    return {"ok": True}

# ==================== CRIATIVOS ====================

@api_router.post("/creatives/generate")
async def generate_creative(payload: CreativeGenerate, current_user=Depends(get_current_user)):
    session_id = str(uuid.uuid4())
    # get per-user settings for custom keys
    settings = await db.app_settings.find_one({"owner_id": current_user["id"]}, {"_id": 0}) or {}
    text_key = settings.get("llm_text_key") or EMERGENT_LLM_KEY
    image_key = settings.get("llm_image_key") or EMERGENT_LLM_KEY

    caption_prompt = f"""Crie uma legenda envolvente para um post de {payload.network} de um escritório de advocacia.
Título: {payload.title}
Tema: {payload.topic}
Tipo de caso: {payload.case_type or 'geral'}
Tom: {payload.tone}
Formato: {payload.format}

Requisitos: Português-BR, máx 6 linhas, CTA, 5 hashtags, sem juridiquês.
Retorne APENAS o texto da legenda."""
    caption = f"{payload.title}\n\n{payload.topic}\n\n#advocacia #direito #legalflow"
    try:
        text_chat = LlmChat(
            api_key=text_key, session_id=session_id + "-text",
            system_message="Você é especialista em copywriting jurídico para redes sociais.",
        ).with_model("openai", "gpt-4o-mini")
        caption = await text_chat.send_message(UserMessage(text=caption_prompt))
    except Exception:
        log.exception("Caption gen failed")

    image_prompt = (
        f"Professional modern social media graphic for a Brazilian law firm. "
        f"Topic: {payload.topic}. Title: \"{payload.title}\". "
        f"Style: clean minimal sophisticated, navy/slate palette with warm amber accents. "
        f"Square 1:1, high-quality, tasteful. Typography elegant, no people faces."
    )
    image_b64 = None
    try:
        from emergentintegrations.llm.openai.image_generation import OpenAIImageGeneration
        img_gen = OpenAIImageGeneration(api_key=image_key)
        images = await img_gen.generate_images(
            prompt=image_prompt,
            model="gpt-image-1",
            number_of_images=1,
        )
        if images:
            # images[0] is bytes
            import base64 as _b64
            raw = images[0] if isinstance(images[0], (bytes, bytearray)) else None
            if raw:
                image_b64 = _b64.b64encode(raw).decode()
    except Exception:
        log.exception("Image gen failed")

    cid = str(uuid.uuid4())
    doc = {
        "id": cid, "owner_id": current_user["id"],
        "title": payload.title, "network": payload.network, "format": payload.format,
        "topic": payload.topic, "tone": payload.tone, "case_type": payload.case_type,
        "caption": caption, "image_b64": image_b64,
        "status": "rascunho", "created_at": now_iso(),
    }
    await db.creatives.insert_one(doc)
    return {
        "id": cid, "title": payload.title, "network": payload.network,
        "format": payload.format, "caption": caption, "image_b64": image_b64,
        "created_at": doc["created_at"],
    }

@api_router.get("/creatives")
async def list_creatives(current_user=Depends(get_current_user)):
    items = await db.creatives.find({"owner_id": current_user["id"]}, {"_id": 0}).sort("created_at", -1).to_list(200)
    return items

@api_router.delete("/creatives/{cid}")
async def delete_creative(cid: str, current_user=Depends(get_current_user)):
    await db.creatives.delete_one({"id": cid, "owner_id": current_user["id"]})
    return {"ok": True}

# ==================== DASHBOARD METRICS ====================

@api_router.get("/dashboard/metrics")
async def dashboard_metrics(current_user=Depends(get_current_user)):
    owner = current_user["id"]
    leads = await db.leads.find({"owner_id": owner}, {"_id": 0}).to_list(2000)
    transactions = await db.transactions.find({"owner_id": owner}, {"_id": 0}).to_list(2000)
    processes = await db.processes.find({"owner_id": owner}, {"_id": 0}).to_list(2000)
    by_stage = {s: 0 for s in CRM_STAGES}
    for l in leads:
        s = l.get("stage", "novos_leads")
        # backward compat
        legacy_map = {"lead": "novos_leads", "contato": "em_contato", "proposta": "em_negociacao",
                      "fechado": "convertido", "perdido": "nao_interessado"}
        s = legacy_map.get(s, s)
        by_stage[s] = by_stage.get(s, 0) + 1
    total_leads = len(leads)
    closed = by_stage.get("convertido", 0)
    conversion = round((closed / total_leads * 100), 1) if total_leads > 0 else 0
    receita = sum(t["amount"] for t in transactions if t["type"] == "receita" and t["status"] == "pago")
    pendente = sum(t["amount"] for t in transactions if t["type"] == "receita" and t["status"] == "pendente")
    despesas = sum(t["amount"] for t in transactions if t["type"] == "despesa" and t["status"] == "pago")

    # Alerts: audiências próximas 7 dias
    today = datetime.now(timezone.utc).date()
    soon = today + timedelta(days=7)
    upcoming = []
    for p in processes:
        nh = p.get("next_hearing")
        if not nh:
            continue
        try:
            d = datetime.fromisoformat(str(nh)).date()
        except Exception:
            try:
                d = datetime.strptime(str(nh)[:10], "%Y-%m-%d").date()
            except Exception:
                continue
        if today <= d <= soon:
            upcoming.append({
                "process_id": p.get("id"), "client_name": p.get("client_name"),
                "process_number": p.get("process_number"), "case_type": p.get("case_type"),
                "date": d.isoformat(), "days_left": (d - today).days,
            })
    upcoming.sort(key=lambda x: x["date"])

    # Leads por urgência
    by_urgency = {"baixa": 0, "media": 0, "alta": 0, "critica": 0}
    for l in leads:
        u = l.get("urgency", "media")
        by_urgency[u] = by_urgency.get(u, 0) + 1

    return {
        "leads": {
            "total": total_leads, "by_stage": by_stage,
            "by_urgency": by_urgency, "conversion_rate": conversion,
        },
        "finance": {"receita_paga": receita, "receita_pendente": pendente,
                    "despesas": despesas, "lucro": receita - despesas},
        "processes": {"total": len(processes),
                      "ativos": len([p for p in processes if p.get("status") == "Em Andamento"])},
        "alerts": {"upcoming_hearings": upcoming},
    }

# ==================== APPOINTMENTS / AGENDA ====================

class AppointmentCreate(BaseModel):
    title: str
    client_name: Optional[str] = None
    contact_id: Optional[str] = None
    lead_id: Optional[str] = None
    process_id: Optional[str] = None
    starts_at: str  # ISO
    duration_min: int = 60
    location: Optional[str] = "Google Meet"
    meeting_link: Optional[str] = None
    notes: Optional[str] = None
    status: Literal["confirmado", "pendente", "cancelado"] = "confirmado"

@api_router.post("/appointments")
async def create_appointment(payload: AppointmentCreate, current_user=Depends(get_current_user)):
    aid = str(uuid.uuid4())
    meeting_link = payload.meeting_link
    if not meeting_link and (payload.location or "").lower().startswith("google meet"):
        # mock google meet link
        code = str(uuid.uuid4())[:3] + "-" + str(uuid.uuid4())[:4] + "-" + str(uuid.uuid4())[:3]
        meeting_link = f"https://meet.google.com/{code}"
    doc = {
        "id": aid, "owner_id": current_user["id"],
        **payload.model_dump(exclude={"meeting_link"}),
        "meeting_link": meeting_link,
        "created_at": now_iso(),
    }
    await db.appointments.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api_router.get("/appointments")
async def list_appointments(current_user=Depends(get_current_user)):
    items = await db.appointments.find({"owner_id": current_user["id"]}, {"_id": 0}).sort("starts_at", 1).to_list(500)
    return items

@api_router.patch("/appointments/{aid}")
async def update_appointment(aid: str, payload: dict, current_user=Depends(get_current_user)):
    payload.pop("_id", None); payload.pop("id", None)
    await db.appointments.update_one({"id": aid, "owner_id": current_user["id"]}, {"$set": payload})
    item = await db.appointments.find_one({"id": aid}, {"_id": 0})
    return item

@api_router.delete("/appointments/{aid}")
async def delete_appointment(aid: str, current_user=Depends(get_current_user)):
    await db.appointments.delete_one({"id": aid, "owner_id": current_user["id"]})
    return {"ok": True}

# ==================== AI SUMMARY ====================

class SummaryRequest(BaseModel):
    text: str
    kind: Literal["process", "conversation"] = "process"

@api_router.post("/ai/summary")
async def ai_summary(payload: SummaryRequest, current_user=Depends(get_current_user)):
    system = "Você resume textos jurídicos em português do Brasil, de forma concisa (máx 4 linhas)."
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY, session_id=str(uuid.uuid4()),
            system_message=system,
        ).with_model("openai", "gpt-4o-mini")
        summary = await chat.send_message(UserMessage(text=f"Resuma:\n{payload.text}"))
        return {"summary": summary}
    except Exception:
        raise HTTPException(500, "Erro ao resumir")

# ==================== DEBUG TOOL ====================

@api_router.post("/debug/instruction")
async def debug_instruction(payload: DebugInstruction, current_user=Depends(get_current_user)):
    """Armazena instruções do debug tool para referência."""
    rec = {
        "id": str(uuid.uuid4()), "owner_id": current_user["id"],
        "instruction": payload.instruction, "context": payload.context,
        "created_at": now_iso(),
    }
    await db.debug_logs.insert_one(rec)
    rec.pop("_id", None)
    return rec

@api_router.get("/debug/instructions")
async def list_debug_instructions(current_user=Depends(get_current_user)):
    items = await db.debug_logs.find({"owner_id": current_user["id"]}, {"_id": 0}).sort("created_at", -1).to_list(100)
    return items

# ==================== WHATSAPP LOGS ====================

@api_router.get("/whatsapp/bot-delivery-stats")
async def bot_delivery_stats(current_user=Depends(get_current_user)):
    """Estatisticas de entrega das respostas do robo - util para diagnostico."""
    owner = current_user["id"]
    total_bot = await db.whatsapp_messages.count_documents({"owner_id": owner, "bot": True})
    delivered = await db.whatsapp_messages.count_documents(
        {"owner_id": owner, "bot": True, "delivered": True}
    )
    failed = await db.whatsapp_messages.count_documents(
        {"owner_id": owner, "bot": True, "delivered": False}
    )
    unknown = total_bot - delivered - failed
    # Status real de WhatsApp (vindo do MessageStatusCallback da Z-API)
    wa_received = await db.whatsapp_messages.count_documents(
        {"owner_id": owner, "bot": True, "wa_status": {"$in": ["received", "read"]}}
    )
    wa_read = await db.whatsapp_messages.count_documents(
        {"owner_id": owner, "bot": True, "wa_status": "read"}
    )
    wa_failed = await db.whatsapp_messages.count_documents(
        {"owner_id": owner, "bot": True, "wa_status": "failed"}
    )
    # Pegar as 5 ultimas falhas
    recent_failures = []
    cur = db.whatsapp_messages.find(
        {"owner_id": owner, "bot": True, "delivered": False},
        {"_id": 0, "text": 1, "send_error": 1, "provider_response": 1,
         "provider_status": 1, "created_at": 1, "contact_id": 1},
    ).sort("created_at", -1).limit(5)
    async for m in cur:
        # enrich with contact name/phone
        ct = await db.whatsapp_contacts.find_one(
            {"id": m.get("contact_id")}, {"_id": 0, "name": 1, "phone": 1},
        )
        if ct:
            m["contact_name"] = ct.get("name")
            m["contact_phone"] = ct.get("phone")
        recent_failures.append(m)
    tracked = delivered + failed
    return {
        "total_bot_replies": total_bot,
        "delivered": delivered,
        "failed": failed,
        "unknown": unknown,
        "tracked_total": tracked,
        "delivery_rate": round((delivered / tracked * 100), 1) if tracked else 0,
        "whatsapp_received": wa_received,
        "whatsapp_read": wa_read,
        "whatsapp_failed": wa_failed,
        "recent_failures": recent_failures,
    }


@api_router.get("/whatsapp/logs")
async def whatsapp_logs(limit: int = 100, current_user=Depends(get_current_user)):
    """Últimas mensagens WhatsApp (recebidas + enviadas + bot) com dados do contato."""
    msgs = await db.whatsapp_messages.find(
        {"owner_id": current_user["id"]}, {"_id": 0}
    ).sort("created_at", -1).to_list(limit)
    # enrich with contact name
    contact_ids = list({m.get("contact_id") for m in msgs if m.get("contact_id")})
    contacts_map = {}
    if contact_ids:
        contacts = await db.whatsapp_contacts.find(
            {"id": {"$in": contact_ids}, "owner_id": current_user["id"]}, {"_id": 0}
        ).to_list(500)
        contacts_map = {c["id"]: c for c in contacts}
    for m in msgs:
        c = contacts_map.get(m.get("contact_id"), {})
        m["contact_name"] = c.get("name", "—")
        m["contact_phone"] = c.get("phone", "")
    return msgs

# ==================== PROCESS TIMELINE ====================

class TimelineAdd(BaseModel):
    event: str
    type: str = "info"  # info, success, warning, critical
    description: Optional[str] = None

@api_router.post("/processes/{pid}/timeline")
async def add_timeline(pid: str, payload: TimelineAdd, current_user=Depends(get_current_user)):
    p = await db.processes.find_one({"id": pid, "owner_id": current_user["id"]}, {"_id": 0})
    if not p:
        raise HTTPException(404, "Processo não encontrado")
    entry = {"date": now_iso(), "event": payload.event, "type": payload.type, "description": payload.description}
    await db.processes.update_one({"id": pid}, {"$push": {"timeline": entry}})
    return {"ok": True, "entry": entry}

# ==================== PUBLIC CLIENT PORTAL ====================

class PublicConsulta(BaseModel):
    phone: str

@api_router.post("/public/consulta")
async def public_consulta(payload: PublicConsulta):
    """Cliente consulta processos pelo telefone (portal público)."""
    phone_n = normalize_phone(payload.phone)
    if not phone_n or len(phone_n) < 10:
        raise HTTPException(400, "Telefone inválido. Use DDD + número (ex: 11988887777)")
    # Buscar lead pelo telefone
    lead = await db.leads.find_one({}, {"_id": 0})
    candidates = await db.leads.find({}, {"_id": 0}).to_list(5000)
    match_names = set()
    for c in candidates:
        if normalize_phone(c.get("phone", "")) == phone_n:
            match_names.add(c.get("name", "").strip().lower())
    # também buscar por whatsapp_contacts
    wa_contacts = await db.whatsapp_contacts.find({}, {"_id": 0}).to_list(5000)
    for c in wa_contacts:
        if normalize_phone(c.get("phone", "")) == phone_n or c.get("phone_normalized") == phone_n:
            match_names.add(c.get("name", "").strip().lower())
    if not match_names:
        return {"ok": True, "found": False, "client_name": None, "processes": []}
    # buscar processos por client_name (case-insensitive)
    all_procs = await db.processes.find({}, {"_id": 0}).sort("created_at", -1).to_list(5000)
    matched = []
    for p in all_procs:
        cname = (p.get("client_name") or "").strip().lower()
        # match if any token in common (first name equality or full match)
        for mn in match_names:
            if cname == mn or (mn and mn.split(" ")[0] == cname.split(" ")[0] and len(cname) > 3):
                matched.append(p)
                break
    # display: last human-readable name
    display_name = next(iter(match_names)).title() if match_names else None
    # mask process numbers? keep as-is for demo
    return {
        "ok": True,
        "found": len(matched) > 0,
        "client_name": display_name,
        "processes": matched,
    }

# ==================== SEED DEMO DATA ====================

@api_router.post("/seed/demo")
async def seed_demo(current_user=Depends(get_current_user)):
    owner = current_user["id"]
    existing_leads = await db.leads.count_documents({"owner_id": owner})
    if existing_leads > 0:
        return {"ok": True, "message": "Dados de demo já existem"}

    demo_leads = [
        {"name": "Maria Silva Santos", "phone": "+55 11 98765-4321", "email": "maria.silva@email.com",
         "case_type": "Família", "stage": "novos_leads", "score": 65, "urgency": "media", "source": "WhatsApp",
         "description": "Divórcio consensual, 1 filho menor", "tags": ["divórcio", "guarda"]},
        {"name": "João Pereira", "phone": "+55 11 91234-5678", "email": "joao.p@email.com",
         "case_type": "Trabalhista", "stage": "em_contato", "score": 80, "urgency": "alta", "source": "Chatbot",
         "description": "Demissão sem justa causa", "tags": ["rescisão", "verbas"]},
        {"name": "Ana Costa Lima", "phone": "+55 21 99988-7766", "email": "ana.costa@email.com",
         "case_type": "Previdenciário/INSS", "stage": "qualificado", "score": 90, "urgency": "alta", "source": "Indicação",
         "description": "Aposentadoria por invalidez negada", "tags": ["INSS", "invalidez"]},
        {"name": "Carlos Mendes", "phone": "+55 11 97777-8888", "email": "carlos.m@email.com",
         "case_type": "Bancário", "stage": "convertido", "score": 95, "urgency": "media", "source": "Anúncio",
         "description": "Revisão de contrato", "tags": ["revisão", "juros"]},
        {"name": "Fernanda Oliveira", "phone": "+55 11 96666-1111", "email": "fer.oli@email.com",
         "case_type": "Cível", "stage": "novos_leads", "score": 50, "urgency": "media", "source": "Site",
         "description": "Indenização", "tags": ["danos morais"]},
        {"name": "Roberto Almeida", "phone": "+55 31 95555-2222", "email": "roberto.a@email.com",
         "case_type": "Empresarial", "stage": "interessado", "score": 70, "urgency": "baixa", "source": "LinkedIn",
         "description": "Holding familiar", "tags": ["holding"]},
        {"name": "Patrícia Souza", "phone": "+55 11 94444-3333", "email": "pat.souza@email.com",
         "case_type": "Família", "stage": "em_negociacao", "score": 75, "urgency": "alta", "source": "WhatsApp",
         "description": "Pensão alimentícia", "tags": ["pensão", "alimentos"]},
    ]
    for lead in demo_leads:
        await db.leads.insert_one({
            "id": str(uuid.uuid4()), "owner_id": owner, **lead,
            "notes": "", "created_at": now_iso(), "updated_at": now_iso(),
        })

    contacts = [
        {"name": "Maria Silva", "phone": "+55 11 98765-4321",
         "last_message": "Doutor, conseguiu falar com meu marido?", "unread": 2, "avatar_color": "bg-rose-500"},
        {"name": "João Pereira", "phone": "+55 11 91234-5678",
         "last_message": "Obrigado! Quando você precisa dos documentos?", "unread": 0, "avatar_color": "bg-blue-500"},
        {"name": "Ana Costa", "phone": "+55 21 99988-7766",
         "last_message": "Vou aguardar a proposta de honorários", "unread": 1, "avatar_color": "bg-emerald-500"},
        {"name": "Carlos Mendes", "phone": "+55 11 97777-8888",
         "last_message": "Pagamento efetuado, comprovante enviado", "unread": 0, "avatar_color": "bg-amber-500"},
        {"name": "Patrícia Souza", "phone": "+55 11 94444-3333",
         "last_message": "Olá, gostaria de marcar uma consulta", "unread": 3, "avatar_color": "bg-purple-500"},
    ]
    saved_contacts = []
    for c in contacts:
        cid = str(uuid.uuid4())
        doc = {"id": cid, "owner_id": owner, **c,
               "phone_normalized": normalize_phone(c["phone"]),
               "last_message_at": now_iso()}
        await db.whatsapp_contacts.insert_one(doc)
        saved_contacts.append(doc)

    demo_msgs = [
        {"text": "Olá Doutor, tudo bem?", "from_me": False},
        {"text": "Olá Maria! Tudo sim, e você?", "from_me": True},
        {"text": "Estou bem. Doutor, tem novidades sobre o processo?", "from_me": False},
        {"text": "Sim, protocolamos a petição ontem. Agora aguardamos a citação.", "from_me": True},
        {"text": "Que ótimo! Quanto tempo costuma demorar?", "from_me": False},
        {"text": "Em média 30 a 45 dias. Aviso assim que houver movimentação.", "from_me": True},
        {"text": "Doutor, conseguiu falar com meu marido?", "from_me": False},
    ]
    first_contact = saved_contacts[0]
    for m in demo_msgs:
        await db.whatsapp_messages.insert_one({
            "id": str(uuid.uuid4()), "owner_id": owner,
            "contact_id": first_contact["id"], "text": m["text"],
            "from_me": m["from_me"], "created_at": now_iso(),
        })

    procs = [
        {"client_name": "Maria Silva Santos", "process_number": "0001234-56.2025.8.26.0100",
         "case_type": "Divórcio", "court": "TJ-SP", "status": "Em Andamento",
         "description": "Divórcio com partilha", "next_hearing": "2026-08-15"},
        {"client_name": "João Pereira", "process_number": "0007890-12.2025.5.02.0050",
         "case_type": "Trabalhista", "court": "TRT-2", "status": "Em Andamento",
         "description": "Verbas rescisórias", "next_hearing": "2026-08-28"},
        {"client_name": "Ana Costa Lima", "process_number": "0002468-13.2025.4.03.6100",
         "case_type": "INSS", "court": "JF-SP", "status": "Aguardando Sentença",
         "description": "Aposentadoria", "next_hearing": None},
        {"client_name": "Carlos Mendes", "process_number": "0003579-24.2024.8.26.0001",
         "case_type": "Bancário", "court": "TJ-SP", "status": "Concluído",
         "description": "Revisão de juros", "next_hearing": None},
    ]
    for p in procs:
        timeline = [
            {"date": now_iso(), "event": "Processo cadastrado", "type": "info",
             "description": f"Processo {p['process_number']} cadastrado no sistema."},
        ]
        # eventos fictícios por tipo
        if "Divórcio" in p.get("case_type", "") or "Divórcio" in p.get("description", ""):
            timeline += [
                {"date": now_iso(), "event": "Petição inicial protocolada", "type": "success",
                 "description": "Petição inicial distribuída e protocolada junto ao tribunal."},
                {"date": now_iso(), "event": "Aguardando citação da parte contrária", "type": "info",
                 "description": "Prazo médio: 30 a 45 dias."},
            ]
        elif "Trabalhista" in p.get("case_type", ""):
            timeline += [
                {"date": now_iso(), "event": "Reclamação trabalhista distribuída", "type": "success"},
                {"date": now_iso(), "event": "Audiência inicial designada", "type": "info",
                 "description": "Audiência una marcada conforme agenda da Vara."},
            ]
        elif "INSS" in p.get("case_type", ""):
            timeline += [
                {"date": now_iso(), "event": "Ação previdenciária ajuizada", "type": "success"},
                {"date": now_iso(), "event": "Perícia médica realizada", "type": "success"},
                {"date": now_iso(), "event": "Aguardando sentença", "type": "warning",
                 "description": "Concluso para julgamento."},
            ]
        elif "Bancário" in p.get("case_type", ""):
            timeline += [
                {"date": now_iso(), "event": "Ação revisional distribuída", "type": "success"},
                {"date": now_iso(), "event": "Sentença procedente", "type": "success",
                 "description": "Revisão deferida. Contrato revisado com redução de encargos."},
                {"date": now_iso(), "event": "Processo concluído", "type": "success"},
            ]
        await db.processes.insert_one({
            "id": str(uuid.uuid4()), "owner_id": owner, **p,
            "timeline": timeline,
            "documents": [], "created_at": now_iso(),
        })

    txs = [
        {"client_name": "Carlos Mendes", "description": "Honorários - Revisão Bancária",
         "amount": 8500.00, "type": "receita", "status": "pago", "due_date": "2026-01-15"},
        {"client_name": "Maria Silva", "description": "Honorários - Divórcio (1ª parcela)",
         "amount": 2500.00, "type": "receita", "status": "pago", "due_date": "2026-01-20"},
        {"client_name": "João Pereira", "description": "Honorários iniciais - Trabalhista",
         "amount": 1800.00, "type": "receita", "status": "pendente", "due_date": "2026-08-25"},
        {"client_name": "Ana Costa", "description": "Honorários - INSS",
         "amount": 3200.00, "type": "receita", "status": "pendente", "due_date": "2026-08-15"},
        {"client_name": "Escritório", "description": "Aluguel",
         "amount": 4500.00, "type": "despesa", "status": "pago", "due_date": "2026-07-05"},
        {"client_name": "Escritório", "description": "Software jurídico",
         "amount": 350.00, "type": "despesa", "status": "pago", "due_date": "2026-07-01"},
    ]
    for t in txs:
        await db.transactions.insert_one({
            "id": str(uuid.uuid4()), "owner_id": owner, **t,
            "process_id": None, "created_at": now_iso(),
        })
    return {"ok": True, "message": "Dados de demonstração criados com sucesso"}

# ==================== HEALTH ====================

@api_router.get("/")
async def root():
    return {"message": "Espírito Santo AI API", "status": "ok"}

# ==================== ADMIN — CASE ANALYSES & ACERTIVIDADE ====================

async def require_admin(current_user=Depends(get_current_user)):
    """Garante que o usuario logado tem flag de admin."""
    if not (current_user.get("is_admin") or current_user.get("role") == "admin"):
        # Fallback: o owner-titular do escritorio (1o user criado) tambem e admin
        first = await db.users.find_one({}, {"_id": 0, "id": 1}, sort=[("created_at", 1)])
        if not first or first.get("id") != current_user.get("id"):
            raise HTTPException(403, "Acesso restrito ao administrador")
    return current_user


@api_router.get("/admin/case-analyses")
async def admin_list_case_analyses(
    qualificacao: Optional[str] = None,
    limit: int = 200,
    current_user=Depends(require_admin),
):
    """Lista todas as analises de caso geradas pelo chat humanizado.
    Cada item tem: indice de acertividade, qualificacao, area, motivo, fundamentos.
    Painel administrativo usa para qualificar ou nao o cliente."""
    q: Dict[str, Any] = {}
    if qualificacao in ("qualificado", "nao_qualificado", "necessita_mais_info"):
        q["qualificacao"] = qualificacao
    items = await db.case_analyses.find(q, {"_id": 0}).sort("updated_at", -1).to_list(limit)
    # estatisticas
    total = len(items)
    qual = sum(1 for i in items if i.get("qualificacao") == "qualificado")
    naoq = sum(1 for i in items if i.get("qualificacao") == "nao_qualificado")
    mais = sum(1 for i in items if i.get("qualificacao") == "necessita_mais_info")
    avg_acert = round(sum(i.get("acertividade", 0) for i in items) / total, 1) if total else 0
    avg_chance = round(sum(i.get("chance_exito", 0) for i in items) / total, 1) if total else 0
    return {
        "total": total,
        "qualificados": qual,
        "nao_qualificados": naoq,
        "necessita_mais_info": mais,
        "avg_acertividade": avg_acert,
        "avg_chance_exito": avg_chance,
        "items": items,
    }


@api_router.get("/admin/case-analyses/{aid}")
async def admin_get_case_analysis(aid: str, current_user=Depends(require_admin)):
    """Detalhe de uma analise + transcript completo da conversa."""
    a = await db.case_analyses.find_one({"id": aid}, {"_id": 0})
    if not a:
        raise HTTPException(404, "Analise nao encontrada")
    msgs = await db.chat_messages.find(
        {"session_id": a["session_id"]}, {"_id": 0}
    ).sort("created_at", 1).to_list(500)
    return {"analysis": a, "messages": msgs}


class CaseAnalysisManualUpdate(BaseModel):
    qualificacao: Optional[Literal["qualificado", "nao_qualificado", "necessita_mais_info"]] = None
    notes: Optional[str] = None


@api_router.patch("/admin/case-analyses/{aid}")
async def admin_update_case_analysis(
    aid: str, payload: CaseAnalysisManualUpdate, current_user=Depends(require_admin),
):
    """Permite o admin sobrescrever a qualificacao gerada pela IA."""
    update: Dict[str, Any] = {"updated_at": now_iso(), "manual_review": True,
                               "reviewed_by": current_user["id"]}
    if payload.qualificacao:
        update["qualificacao"] = payload.qualificacao
    if payload.notes is not None:
        update["admin_notes"] = payload.notes
    res = await db.case_analyses.update_one({"id": aid}, {"$set": update})
    if res.matched_count == 0:
        raise HTTPException(404, "Analise nao encontrada")
    a = await db.case_analyses.find_one({"id": aid}, {"_id": 0})
    return a


# ==================== LEGISLATION ATUALIZADA ====================

@api_router.get("/legislation/today")
async def legislation_today():
    """Retorna o brief diario de legislacao em vigor (cache 24h)."""
    brief = await get_daily_legislation_brief()
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "date_human": datetime.now(timezone.utc).strftime("%d/%m/%Y"),
        "brief": brief,
    }


@api_router.post("/legislation/refresh")
async def legislation_refresh(current_user=Depends(require_admin)):
    """Forca a regeracao do brief diario (admin only)."""
    today_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await db.legislation_cache.delete_one({"date": today_key})
    brief = await get_daily_legislation_brief()
    return {"date": today_key, "brief": brief}


# ==================== ADMIN SEED ====================

ADMIN_EMAIL_DEFAULT = "admin@kenia-garcia.com.br"
ADMIN_PASSWORD_DEFAULT = "Kenia@Admin2026"

@app.on_event("startup")
async def _seed_admin_user():
    """Cria usuario admin default se nao existir."""
    try:
        existing = await db.users.find_one({"email": ADMIN_EMAIL_DEFAULT})
        if not existing:
            uid = str(uuid.uuid4())
            await db.users.insert_one({
                "id": uid,
                "name": "Administrador",
                "email": ADMIN_EMAIL_DEFAULT,
                "password": hash_password(ADMIN_PASSWORD_DEFAULT),
                "oab": None,
                "is_admin": True,
                "role": "admin",
                "created_at": now_iso(),
            })
            logger.info(f"[seed] admin user created: {ADMIN_EMAIL_DEFAULT}")
        else:
            # Garante a flag is_admin=True
            if not existing.get("is_admin"):
                await db.users.update_one(
                    {"email": ADMIN_EMAIL_DEFAULT},
                    {"$set": {"is_admin": True, "role": "admin"}},
                )
    except Exception:
        log.exception("admin seed failed")


# ==================== APP SETTINGS (KEYS) ====================

class AppSettings(BaseModel):
    llm_text_key: Optional[str] = None
    llm_image_key: Optional[str] = None

@api_router.get("/settings")
async def get_settings(current_user=Depends(get_current_user)):
    s = await db.app_settings.find_one({"owner_id": current_user["id"]}, {"_id": 0})
    if not s:
        s = {
            "owner_id": current_user["id"],
            "llm_text_key": "",
            "llm_image_key": "",
        }
    # mask for UI
    def mask(k):
        if not k:
            return ""
        if len(k) < 10:
            return "***"
        return k[:8] + "..." + k[-4:]
    return {
        "owner_id": s["owner_id"],
        "llm_text_key_masked": mask(s.get("llm_text_key", "")),
        "llm_image_key_masked": mask(s.get("llm_image_key", "")),
        "has_text_key": bool(s.get("llm_text_key")),
        "has_image_key": bool(s.get("llm_image_key")),
        "using_default_text": not s.get("llm_text_key"),
        "using_default_image": not s.get("llm_image_key"),
    }

@api_router.put("/settings")
async def set_settings(payload: AppSettings, current_user=Depends(get_current_user)):
    update = {"owner_id": current_user["id"]}
    # only update if provided (non-empty) — empty means "keep"
    if payload.llm_text_key is not None:
        update["llm_text_key"] = payload.llm_text_key.strip()
    if payload.llm_image_key is not None:
        update["llm_image_key"] = payload.llm_image_key.strip()
    update["updated_at"] = now_iso()
    await db.app_settings.update_one(
        {"owner_id": current_user["id"]}, {"$set": update}, upsert=True,
    )
    return {"ok": True}

@api_router.post("/settings/test-text")
async def test_text_key(current_user=Depends(get_current_user)):
    s = await db.app_settings.find_one({"owner_id": current_user["id"]}, {"_id": 0}) or {}
    key = s.get("llm_text_key") or EMERGENT_LLM_KEY
    try:
        chat = LlmChat(
            api_key=key, session_id=f"test-{uuid.uuid4()}",
            system_message="Responda apenas 'OK' em uma palavra.",
        ).with_model("openai", "gpt-4o-mini")
        r = await chat.send_message(UserMessage(text="Teste"))
        return {"ok": True, "using_custom_key": bool(s.get("llm_text_key")), "response": (r or "")[:50]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}

@api_router.post("/settings/test-image")
async def test_image_key(current_user=Depends(get_current_user)):
    s = await db.app_settings.find_one({"owner_id": current_user["id"]}, {"_id": 0}) or {}
    key = s.get("llm_image_key") or EMERGENT_LLM_KEY
    try:
        from emergentintegrations.llm.openai.image_generation import OpenAIImageGeneration
        img_gen = OpenAIImageGeneration(api_key=key)
        images = await img_gen.generate_images(
            prompt="simple blue circle on white background, minimal, test image",
            model="gpt-image-1",
            number_of_images=1,
        )
        return {"ok": bool(images), "using_custom_key": bool(s.get("llm_image_key")),
                "model": "gpt-image-1"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}

# ==================== FUSAO DE IMAGENS (Gemini Nano Banana) ====================

class FuseImagesIn(BaseModel):
    image1_base64: str  # data URI ou base64 puro
    image2_base64: str
    prompt: Optional[str] = None  # instrucao extra (opcional)


def _strip_data_uri(b64: str) -> str:
    """Aceita 'data:image/png;base64,XXXX' ou 'XXXX' puro e retorna apenas a parte base64."""
    if not b64:
        return ""
    if "," in b64 and b64.startswith("data:"):
        return b64.split(",", 1)[1]
    return b64


@api_router.post("/creatives/fuse-images")
async def fuse_images(payload: FuseImagesIn, current_user=Depends(get_current_user)):
    """Recebe 2 imagens (base64) e gera uma nova combinando-as via Gemini Nano Banana."""
    img1 = _strip_data_uri(payload.image1_base64)
    img2 = _strip_data_uri(payload.image2_base64)
    if not img1 or not img2:
        raise HTTPException(400, "Envie as duas imagens em base64")

    extra = (payload.prompt or "").strip()
    instruction = (
        "Combine as duas imagens de referência em uma única composição artística e coesa. "
        "Una elementos, paleta de cores, atmosfera e estilo das duas imagens em uma só, "
        "criando algo novo e harmônico. Mantenha alta qualidade fotográfica."
    )
    if extra:
        instruction += f"\n\nInstrução adicional do usuário: {extra}"

    session_id = f"fuse-{current_user['id']}-{uuid.uuid4().hex[:8]}"
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=session_id,
            system_message="Você é um diretor de arte que combina imagens com excelência estética.",
        ).with_model("gemini", "gemini-2.5-flash-image-preview").with_params(modalities=["image", "text"])

        msg = UserMessage(
            text=instruction,
            file_contents=[ImageContent(img1), ImageContent(img2)],
        )
        try:
            text_resp, images = await chat.send_message_multimodal_response(msg)
        except Exception as primary_err:
            log.warning(f"gemini-2.5 falhou, tentando gemini-3.1: {primary_err}")
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=session_id + "-retry",
                system_message="Você é um diretor de arte que combina imagens com excelência estética.",
            ).with_model("gemini", "gemini-3.1-flash-image-preview").with_params(modalities=["image", "text"])
            text_resp, images = await chat.send_message_multimodal_response(msg)
    except Exception as e:
        log.exception("fuse-images error")
        raise HTTPException(500, f"Erro ao gerar imagem: {str(e)[:200]}")

    if not images:
        return {"ok": False, "error": "Nenhuma imagem retornada pelo modelo", "text": text_resp}

    out = images[0]
    data_uri = f"data:{out.get('mime_type', 'image/png')};base64,{out['data']}"
    # salva metadado mas NAO a base64 completa pra nao explodir o mongo
    await db.fused_images.insert_one({
        "id": str(uuid.uuid4()), "owner_id": current_user["id"],
        "prompt": extra, "mime_type": out.get("mime_type", "image/png"),
        "size": len(out.get("data", "")),
        "created_at": now_iso(),
    })
    return {"ok": True, "image": data_uri, "mime_type": out.get("mime_type", "image/png"),
            "text": text_resp[:500] if text_resp else ""}


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"], allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()


# ============================================================
# Baileys sidecar auto-spawn + watchdog
# ------------------------------------------------------------
# In the Emergent preview environment we cannot add to supervisord,
# so the backend spawns the Node.js sidecar itself and restarts it
# if it dies. In Render production this code is a no-op because
# RENDER_DISABLE_BAILEYS_SPAWN=true (the sidecar runs as a separate
# service there). Detection: we look for a local baileys-service/
# directory AND absence of RENDER env flag.
# ============================================================
import subprocess
import shutil

_baileys_proc: Optional[subprocess.Popen] = None
_baileys_dir = Path(__file__).parent.parent / "baileys-service"
_spawn_disabled = os.environ.get("RENDER_DISABLE_BAILEYS_SPAWN", "").lower() in ("1", "true", "yes")

def _baileys_running() -> bool:
    global _baileys_proc
    if _baileys_proc is None:
        return False
    return _baileys_proc.poll() is None

async def _baileys_health_ok() -> bool:
    try:
        url = os.environ.get("BAILEYS_URL", "http://localhost:8002")
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{url}/health")
            return r.status_code == 200
    except Exception:
        return False

def _spawn_baileys():
    """Spawn the Node.js sidecar as a background subprocess."""
    global _baileys_proc
    if _spawn_disabled:
        return
    if not _baileys_dir.exists() or not (_baileys_dir / "server.js").exists():
        logger.info("[baileys-watchdog] sidecar dir not present — skipping spawn")
        return
    node_bin = shutil.which("node")
    if not node_bin:
        logger.warning("[baileys-watchdog] node not found on PATH — cannot spawn")
        return
    env = {
        **os.environ,
        "BAILEYS_PORT": "8002",
        "BAILEYS_INTERNAL_TOKEN": os.environ.get("BAILEYS_INTERNAL_TOKEN", "legalflow-baileys-2026"),
        "BACKEND_WEBHOOK": os.environ.get("BACKEND_WEBHOOK", "http://localhost:8001/api/whatsapp/webhook/baileys"),
    }
    log_path = Path("/var/log/baileys-backend-spawn.log")
    try:
        f = open(log_path, "a")
        _baileys_proc = subprocess.Popen(
            [node_bin, "server.js"],
            cwd=str(_baileys_dir),
            env=env,
            stdout=f,
            stderr=f,
            start_new_session=True,
        )
        logger.info(f"[baileys-watchdog] spawned pid={_baileys_proc.pid}")
    except Exception:
        logger.exception("[baileys-watchdog] spawn failed")

async def _baileys_watchdog_loop():
    """Checks every 15s. If sidecar not running OR health fails, respawn."""
    import asyncio
    await asyncio.sleep(2)  # initial grace
    while True:
        try:
            if _spawn_disabled:
                return
            healthy = await _baileys_health_ok()
            if not healthy and not _baileys_running():
                logger.warning("[baileys-watchdog] sidecar down — respawning")
                _spawn_baileys()
            elif not healthy and _baileys_running():
                # process alive but http unreachable — likely hung, kill it
                try:
                    if _baileys_proc: _baileys_proc.terminate()
                except Exception: pass
                await asyncio.sleep(2)
                _spawn_baileys()
        except Exception:
            logger.exception("[baileys-watchdog] tick error")
        import asyncio as _a
        await _a.sleep(15)

@app.on_event("startup")
async def _start_baileys_watchdog():
    import asyncio
    # fire-and-forget watchdog; don't block startup
    _spawn_baileys()
    asyncio.create_task(_baileys_watchdog_loop())
    logger.info("[baileys-watchdog] started")
          _spawn_baileys()
        except Exception:
            logger.exception("[baileys-watchdog] tick error")
        import asyncio as _a
        await _a.sleep(15)

@app.on_event("startup")
async def _start_baileys_watchdog():
    import asyncio
    # fire-and-forget watchdog; don't block startup
    _spawn_baileys()
    asyncio.create_task(_baileys_watchdog_loop())
    logger.info("[baileys-watchdog] started")
