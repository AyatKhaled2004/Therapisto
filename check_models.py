import re

from google import genai

from APP.config import settings

client = genai.Client(api_key=settings.google_api_key)

CANDIDATES = [
    settings.gemini_chat_model,
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

for model in dict.fromkeys(CANDIDATES):
    try:
        r = client.models.generate_content(model=model, contents="Say OK")
        print(f"✅ {model}: works -> {r.text.strip()[:20]}")
    except Exception as e:
        msg = str(e)
        code = re.search(r"\b(4\d\d|5\d\d)\b", msg)
        print(f"❌ {model}: {code.group(1) if code else 'error'}")
        for key in ("quotaId", "quotaMetric", "retryDelay"):
            m = re.search(rf"'{key}': '([^']+)'", msg)
            if m:
                print(f"     {key}: {m.group(1)}")