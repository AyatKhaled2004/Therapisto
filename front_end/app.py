import os
import time
import uuid
from pathlib import Path

import requests
import streamlit as st
from dotenv import dotenv_values


# ============================================================
# 1. Configuration
# ============================================================

APP_DIR = Path(__file__).resolve().parent

# Look for .env beside this file first, then in the project root
ENV_CANDIDATES = [APP_DIR / ".env", APP_DIR.parent / ".env"]
ENV_PATH = next((p for p in ENV_CANDIDATES if p.is_file()), None)
env_values = dotenv_values(ENV_PATH) if ENV_PATH else {}


def get_setting(name: str, default: str = "") -> str:
    """.env file first, then real environment variables."""
    return (env_values.get(name) or os.getenv(name) or default).strip()


# "localhost" can resolve to IPv6 on Windows while the server listens on IPv4
API_URL = get_setting("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = API_URL.replace("localhost", "127.0.0.1")
API_TOKEN = get_setting("API_TOKEN")

TIMEOUT = 600   # a local model can take a while on CPU

ROUTE_LABELS = {
    "answer": "📚 من قاعدة المعرفة",
    "web": "🌐 مع بحث على الويب",
    "crisis": "🆘 رسالة أمان",
}

EXAMPLE_QUESTIONS = [
    "إيه هي أعراض القلق؟",
    "إزاي أتعامل مع نوبات الهلع؟",
    "إيه هو العلاج المعرفي السلوكي؟",
    "إيه الفرق بين الحزن والاكتئاب؟",
    "إزاي أحسّن نومي لو عندي أرق؟",
    "What is cognitive behavioral therapy?",
]


# ============================================================
# 2. Page setup and styling
# ============================================================

st.set_page_config(page_title="Therapisto", page_icon="🧠", layout="centered")

st.markdown(
    """
    <style>
    /* hide Streamlit's default menu/footer (keep the header: it holds the sidebar toggle) */
    #MainMenu, footer { visibility: hidden; }

    /* each paragraph/list item follows its own language direction (Arabic RTL, English LTR) */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        unicode-bidi: plaintext;
        text-align: start;
        line-height: 1.75;
    }

    [data-testid="stChatMessage"] { padding: 0.9rem 1rem; border-radius: 14px; }

    /* full-width buttons */
    [data-testid="stSidebar"] button { width: 100%; }

    .hero { text-align: center; padding: 1.2rem 0 0.4rem 0; }
    .hero h1 { margin-bottom: 0.2rem; }
    .hero p { opacity: 0.75; margin-top: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. Session state
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending" not in st.session_state:
    st.session_state.pending = None


# ============================================================
# 4. Backend helpers
# ============================================================

@st.cache_data(ttl=5, show_spinner=False)
def backend_ok() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def ask_backend(question: str) -> dict:
    if not API_TOKEN:
        searched = " ، ".join(str(p) for p in ENV_CANDIDATES)
        raise RuntimeError(
            "API_TOKEN مش موجود. اتأكدي إنه في ملف .env في واحد من الأماكن دي: "
            f"{searched}"
        )

    response = requests.post(
        f"{API_URL}/chat",
        json={"question": question, "session_id": st.session_state.session_id},
        headers={"Authorization": f"Bearer {API_TOKEN}"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def friendly_error(error: Exception) -> str:
    if isinstance(error, RuntimeError):
        return str(error)

    if isinstance(error, requests.exceptions.HTTPError):
        code = error.response.status_code
        if code == 401:
            return "API_TOKEN في الواجهة مش مطابق للي في السيرفر."
        if code == 422:
            return "السؤال غير صالح (فاضي أو أطول من 2000 حرف)."
        if code >= 500:
            return "حصل خطأ داخلي في السيرفر. راجعي الـ Terminal بتاع FastAPI."
        return f"حصل خطأ في السيرفر ({code})."

    if isinstance(error, requests.exceptions.Timeout):
        return "الرد أخد وقت طويل. جرّبي تاني."

    if isinstance(error, requests.exceptions.ConnectionError):
        return "مقدرتش أوصل للسيرفر. اتأكدي إن FastAPI شغال."

    if isinstance(error, (KeyError, ValueError)):
        return "السيرفر رجّع استجابة غير متوقعة."

    return "حصلت مشكلة غير متوقعة أثناء معالجة السؤال."


# ============================================================
# 5. Rendering helpers
# ============================================================

def show_sources(sources: list[dict]):
    if not sources:
        return

    # remove duplicates (same link / same title)
    unique, seen = [], set()
    for s in sources:
        key = s.get("url") or (s.get("title"), s.get("page"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)

    with st.expander(f"📎 المصادر ({len(unique)})"):
        for s in unique:
            title = (s.get("title") or s.get("source") or "Source").strip()
            if len(title) > 90:
                title = title[:90] + "…"

            label = f"[{title}]({s['url']})" if s.get("url") else f"**{title}**"

            extras = []
            if s.get("page") is not None:
                extras.append(f"صفحة {s['page']}")
            if s.get("pmcid"):
                extras.append(s["pmcid"])
            if s.get("type") == "kb" and s.get("score"):
                extras.append(f"تطابق {s['score']:.0%}")

            st.markdown(f"- {label}" + (f" — {' · '.join(extras)}" if extras else ""))


def render_message(message: dict):
    if message.get("error"):
        st.error(message["content"])
        return

    if message.get("route") == "crisis":
        with st.container(border=True):
            st.markdown(message["content"])
    else:
        st.markdown(message["content"])

    if message["role"] == "assistant":
        parts = []
        if message.get("route") in ROUTE_LABELS:
            parts.append(ROUTE_LABELS[message["route"]])
        if message.get("elapsed"):
            parts.append(f"⏱ {message['elapsed']:.0f} ثانية")
        if parts:
            st.caption(" · ".join(parts))

        show_sources(message.get("sources", []))


def conversation_as_text() -> str:
    lines = []
    for m in st.session_state.messages:
        who = "أنا" if m["role"] == "user" else "Therapisto"
        lines.append(f"{who}:\n{m['content']}\n")
    return "\n".join(lines)


# ============================================================
# 6. Sidebar
# ============================================================

with st.sidebar:
    st.title("🧠 Therapisto")
    st.caption("مساعد للتوعية والمعلومات في الصحة النفسية")

    online = backend_ok()
    if online:
        st.success("السيرفر شغال")
    else:
        st.error("السيرفر مش شغال")
        st.caption("شغّلي: uv run uvicorn APP.main:app --port 8000")

    if st.button("🔄 تحديث الحالة"):
        backend_ok.clear()
        st.rerun()

    if not API_TOKEN:
        st.error("API_TOKEN مش موجود في ملف .env")

    st.divider()

    if st.button("🗑️ محادثة جديدة"):
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    if st.session_state.messages:
        st.download_button(
            "💾 تحميل المحادثة",
            data=conversation_as_text(),
            file_name="therapisto_chat.txt",
            mime="text/plain",
        )

    st.divider()

    st.warning(
        "Therapisto مش بديل عن الطبيب أو المعالج النفسي، ومش بيشخّص حالات.\n\n"
        "لو في خطر مباشر على سلامتك، اتصل/ي بالطوارئ فورًا."
    )

    with st.expander("تفاصيل الاتصال"):
        st.caption(f"API: {API_URL}")
        st.caption(f"ملف .env: {ENV_PATH if ENV_PATH else 'مش موجود'}")


# ============================================================
# 7. Input (resolved first, so the page renders in one pass)
# ============================================================

typed = st.chat_input("اكتبي سؤالك هنا...")
prompt = typed or st.session_state.pending
st.session_state.pending = None

generate = False
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    generate = True


# ============================================================
# 8. Main chat area
# ============================================================

if not st.session_state.messages:
    st.markdown(
        """
        <div class="hero">
            <h1>مرحبًا بك في Therapisto 🧠</h1>
            <p>اسألي عن القلق، الاكتئاب، التوتر، النوم، أو العلاج النفسي،
            وهجاوب من مصادر علمية موثوقة.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("جرّبي واحد من الأسئلة دي:")
    columns = st.columns(2)
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        if columns[i % 2].button(question, key=f"example_{i}"):
            st.session_state.pending = question
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_message(message)


if generate:
    with st.chat_message("assistant"):
        started = time.perf_counter()

        with st.spinner("بفكر... (ممكن ياخد دقيقة مع الموديل المحلي)"):
            try:
                data = ask_backend(prompt)
                reply = {
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", []),
                    "route": data.get("route", ""),
                    "elapsed": time.perf_counter() - started,
                }
                st.session_state.session_id = data.get(
                    "session_id", st.session_state.session_id
                )

            except Exception as error:
                reply = {
                    "role": "assistant",
                    "content": friendly_error(error),
                    "error": True,
                }

        render_message(reply)

    st.session_state.messages.append(reply)