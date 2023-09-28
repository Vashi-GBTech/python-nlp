from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

def answer_question(question, context):
    encoding = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    start_scores, end_scores = model(input_ids, attention_mask=attention_mask)

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer_tokens = input_ids[0][start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens)

    return answer

context = "DistilBERT is a lightweight version of BERT that is efficient for various NLP tasks. It has been pre-trained on large amounts of text data."

question1 = "What is DistilBERT?"
question2 = "What is the benefit of using DistilBERT?"
question3 = "Which tasks can DistilBERT be used for?"

answer1 = answer_question(question1, context)
answer2 = answer_question(question2, context)
answer3 = answer_question(question3, context)

print(f"Q: {question1}\nA: {answer1}")
print(f"Q: {question2}\nA: {answer2}")
print(f"Q: {question3}\nA: {answer3}")

