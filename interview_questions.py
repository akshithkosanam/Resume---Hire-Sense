"""
interview_questions.py
—————————
A tiny wrapper around Google Gemini that returns a list of
interview questions given some free-text (resume, JD, etc.).
"""

import google.generativeai as genai

# ❶ —-–– SET YOUR KEY HERE  (keep it secret in real projects!)
GEMINI_API_KEY = "AIzaSyBDZZJ4NxgNqiQvMHY3JQF2lY-fRZQWCJs"
genai.configure(api_key=GEMINI_API_KEY)

# ❷ —-–– The model object can be reused for every call
_model = genai.GenerativeModel("gemini-1.5-flash")

def generate_interview_questions(text: str, n: int = 5) -> list[str]:
    """
    Return *n* technical interview questions tailored to the supplied text.
    `text` can be a résumé, a JD, or any description of a candidate.
    """
    prompt = (
        f"Generate exactly {n} concise, technical interview questions — "
        "number them 1-N — that are tailored to the following text:\n\n"
        f"{text.strip()}"
    )
    rsp = _model.generate_content(prompt)
    # split on newlines, strip bullets/numbering
    lines = [ln.lstrip("•- ").strip() for ln in rsp.text.splitlines() if ln.strip()]
    return lines[:n] or ["(Gemini returned no content)"]
