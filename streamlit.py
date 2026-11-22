import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "")
TIMEOUT = 600   # a local model can take a while on CPU

st.set_page_config(page_title="Therapisto", page_icon="🧠", layout="centered")

# Make Arabic and English messages each render in the right direction
st.markdown(
    """
    <style>
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        unicode-bidi: plaintext;
        text-align: start;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def backend_ok() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def ask_backend(question: str) -> dict:
    response = requests.post(
        f"{API_URL}/chat",
        json={"question": question, "session_id": st.session_state.session_id},
        headers={"Authorization": f"Bearer {API_TOKEN}"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def show_sources(sources: list[dict]):
    if not sources:
        return

    with st.expander(f"المصادر ({len(sources)})"):
        for s in sources:
            title = s.get("title") or s.get("source") or "Source"
            line = f"**{title}**"

            if s.get("page") is not None:
                line += f" — صفحة {s['page']}"
            if s.get("pmcid"):
                line += f" — {s['pmcid']}"
            if s.get("url"):
                line += f" — [رابط]({s['url']})"

            st.markdown(f"- {line}")


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.title("🧠 Therapisto")
    st.caption("مساعد للتوعية والمعلومات في الصحة النفسية")

    if backend_ok():
        st.success("السيرفر شغال")
    else:
        st.error("السيرفر مش شغال")

    if st.button("محادثة جديدة", use_container_width=True):
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.warning(
        "Therapisto مش بديل عن الطبيب أو المعالج النفسي، ومش بيشخّص حالات.\n\n"
        "لو في خطر مباشر على سلامتك، اتصل/ي بالطوارئ فورًا."
    )


# ------------------------------------------------------------
# Chat
# ------------------------------------------------------------

st.title("Therapisto")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []))

prompt = st.chat_input("اكتب سؤالك هنا...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("بفكر..."):
            try:
                data = ask_backend(prompt)
                answer, sources = data["answer"], data.get("sources", [])

            except requests.exceptions.ConnectionError:
                answer, sources = "مقدرتش أوصل للسيرفر. اتأكد إن الـ FastAPI شغال.", []

            except requests.exceptions.Timeout:
                answer, sources = "الرد أخد وقت طويل. جرّب تاني.", []

            except requests.exceptions.HTTPError as e:
                code = e.response.status_code
                if code == 401:
                    answer = "الـ API_TOKEN في الواجهة مش مطابق للي في السيرفر."
                else:
                    answer = f"حصل خطأ في السيرفر ({code})."
                sources = []

        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )