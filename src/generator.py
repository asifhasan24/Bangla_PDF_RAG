import os
import google.generativeai as genai

class GeminiGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        # Pull key from env (GEMINI_API_KEY or GOOGLE_API_KEY)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) must be set")
        # Configure the SDK
        genai.configure(api_key=api_key)  # :contentReference[oaicite:0]{index=0}
        self.model_name = model_name

    def generate(self, question: str, context: str, documents: list[str]) -> str:
        # Build a single prompt
        doc_block = "\n\n".join(documents)
        prompt = (
            f"Chat history:\n{context}\n\n"
            f"Documents:\n{doc_block}\n\n"
            f"Question: {question}"
        )

        # Instantiate a model object and generate
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
