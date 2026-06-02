
import re, pickle, torch, faiss, streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer

MODEL_PATH  = "/content/clinical-phi2-final"
BASE_MODEL  = "/content/clinical-phi2-final"
INDEX_PATH  = "/content/clinical_faiss.index"
CORPUS_PATH = "/content/clinical_corpus.pkl"

SYSTEM_PROMPT = (
    "You are a clinical assistant helping patients understand their health. "
    "Provide accurate, clear, and compassionate answers based on verified medical knowledge. "
    "Always recommend consulting a doctor for personal medical decisions."
)

DANGEROUS_PATTERNS = [
    r"what dose should i take",
    r"how many (mg|pills?|tablets?)\s*(should|can)\s*i",
    r"should i stop (taking|my)",
    r"prescribe me", r"diagnose me",
    r"am i (dying|pregnant|infected)", r"is it cancer",
]

HALLUCINATION_SIGNALS = [
    r"\b\d+\s?mg\b", r"take \d+ (pill|tablet)",
    r"100% (safe|effective|cure)", r"guaranteed to",
]

@st.cache_resource(show_spinner=False)
def load_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                              bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=False
    )
    from peft import PeftConfig
    model = PeftModel.from_pretrained(base, MODEL_PATH, local_files_only=True)
    model.eval()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(INDEX_PATH)
    with open(CORPUS_PATH, "rb") as f2:
        corpus = pickle.load(f2)
    return model, tokenizer, embedder, index, corpus, device

def retrieve(query, embedder, index, corpus, k=2):
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [{"question": corpus["questions"][i], "answer": corpus["answers"][i], "score": float(s)}
            for i, s in zip(I[0], D[0]) if s > 0.3]

def input_guard(q):
    return not any(re.search(p, q.lower()) for p in DANGEROUS_PATTERNS)

def output_guard(a):
    return any(re.search(p, a.lower()) for p in HALLUCINATION_SIGNALS)

def generate(question, model, tokenizer, ctx, device):
    rag = ("\n### Relevant Medical Context:\n" + "".join(f"- {d['answer'][:200]}\n" for d in ctx)) if ctx else ""
    prompt = f"### System:\n{SYSTEM_PROMPT}\n{rag}\n### Patient Question:\n{question}\n\n### Clinical Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                              repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = decoded.split("### Clinical Answer:")[-1].strip() if "### Clinical Answer:" in decoded else decoded
    parts = answer.split(".")
    return ".".join(parts[:8]) + "." if len(parts) > 8 else answer

st.set_page_config(page_title="Clinical QA", page_icon="🏥", layout="wide")

with st.sidebar:
    st.title("🏥 Clinical QA")
    st.markdown("**Model:** Phi-2 + QLoRA\n\n**RAG:** FAISS\n\n**Dataset:** MedQuAD/NIH")
    st.divider()
    st.subheader("Sample Questions")
    for s in ["What is type 2 diabetes?", "What causes high blood pressure?",
               "How is asthma treated?", "What are symptoms of depression?"]:
        if st.button(s, use_container_width=True):
            st.session_state["prefill"] = s
    st.divider()
    st.warning("Educational use only. Always consult a healthcare professional.")

st.title("🏥 Clinical Question Answering Assistant")
st.caption("Fine-tuned Phi-2 (QLoRA) · FAISS RAG · Input & Output Guardrails")

with st.spinner("Loading model (~2 min first time)..."):
    try:
        model, tokenizer, embedder, index, corpus, device = load_all()
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

st.success(f"Model ready on {device.upper()}")
prefill  = st.session_state.pop("prefill", "")
question = st.text_area("Ask a medical question:", value=prefill, height=100,
                         placeholder="e.g. What is type 2 diabetes?")
c1, c2 = st.columns([1, 5])
ask = c1.button("Ask", type="primary")
clear = c2.button("Clear")
if clear:
    st.session_state["prefill"] = ""
    st.rerun()
if ask and question.strip():
    if not input_guard(question):
        st.error("Blocked: this question asks for personal prescribing/diagnosis.")
    else:
        with st.spinner("Retrieving context..."):
            ctx = retrieve(question, embedder, index, corpus)
        with st.spinner("Generating answer..."):
            answer = generate(question, model, tokenizer, ctx, device)
        st.subheader("Clinical Answer")
        st.write(answer)
        if output_guard(answer):
            st.warning("May contain specific claims. Verify with a healthcare professional.")
        if ctx:
            with st.expander(f"Retrieved Context ({len(ctx)} source(s))"):
                for i, d in enumerate(ctx):
                    st.markdown(f"**Source {i+1}** — similarity: {d['score']:.3f}")
                    st.write(d["answer"][:300] + "...")
                    st.divider()
        st.caption("Always consult a qualified healthcare professional.")
elif ask:
    st.warning("Please enter a question.")
