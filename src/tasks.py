
from celery_app import celery_app
from generator import GeminiGenerator

generator = GeminiGenerator(model_name="gemini-2.5-flash")

@celery_app.task
def generate_answer_task(question, context, documents):
    """
    Celery task to generate an answer.
    """
    ans = generator.generate(question=question, context=context, documents=documents)
    return ans
