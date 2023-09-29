# import transformer_utils

from sympy import false
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

def answer_question(question, context):
    encoding = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict = false)

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer_tokens = input_ids[0][start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens)

    return answer

context = input("Enter Your Thoughts Here :  ")

question1 = input("Ask me a question : ")
question2 = input("Next question please : ")
question3 = input("Next question please : ")

answer1 = answer_question(question1, context)
answer2 = answer_question(question2, context)
answer3 = answer_question(question3, context)

print(f"Q: {question1}\nA: {answer1}")
print(f"Q: {question2}\nA: {answer2}")
print(f"Q: {question3}\nA: {answer3}")

