from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

chat_log = """
User: Hey, how can I reset my password?
Bot: Click on 'Forgot Password' on the login screen.
User: Thanks! Also, can I change my email later?
Bot: Yes, you can update it in Settings > Profile.
"""

summary = summarizer(chat_log, max_length=40, min_length=10, do_sample=False)
print(summary[0]['summary_text'])
