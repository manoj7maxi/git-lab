Natural Language Processing (NLP) Python Programs
1. Word Sense Disambiguation using Lesk Algorithm
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

def lesk_algorithm(sentence, ambiguous_word):
    print(f"Sentence: {sentence}")
    print(f"Ambiguous Word: {ambiguous_word}\n")
    sense = lesk(sentence.split(), ambiguous_word)
    if sense:
        print("Best Sense:", sense.name())
        print("Definition:", sense.definition())
        print("Examples:", sense.examples())
    else:
        print("No suitable sense found.")

sentence = "He went to the bank to deposit his money"
ambiguous_word = "bank"
lesk_algorithm(sentence, ambiguous_word)
3. TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
d0 = 'I love machine learning'
d1 = 'I love artificial intelligence'
d2 = 'we love NLP'
string = [d0, d1, d2]
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(string).toarray()
idf = tfidf.idf_
tf_matrix = tfidf_matrix / idf
print("TF values:")
for doc_idx, row in enumerate(tf_matrix):
   print(f"Doc{doc_idx}:", dict(zip(tfidf.get_feature_names_out(), row)))

result = tfidf.fit_transform(string)

print('\nidf values:')
for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
   print(ele1, ':', ele2)
4. Information Extraction using spaCy
import spacy
from spacy.matcher import Matcher

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# === Input: Unstructured Text ===
text = """
Apple Inc. is looking at buying a UK-based AI startup for $1 billion.
Tim Cook, the CEO of Apple, said the acquisition will help enhance Siri’s intelligence.
The company has previously acquired startups in Canada and Germany.
"""

# Process the text
doc = nlp(text)

# === 1. Named Entity Recognition (NER) ===
print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text} → {ent.label_} ({spacy.explain(ent.label_)})")

# === 2. Noun Phrase Extraction ===
print("\nNoun Phrases:")
for chunk in doc.noun_chunks:
    print(chunk.text)
5. Question Answering System
from transformers import pipeline

# Load pre-trained QA pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="bert-large-uncased-whole-word-masking-finetuned-squad"
)

# === Provide context ===
context = """
Apple Inc. is a multinational technology company headquartered in Cupertino, California. 
It designs, develops, and sells consumer electronics, computer software, and online services. 
Its best-known products include the iPhone, iPad, and Mac computers. 
Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
"""

# === Ask questions ===
questions = [
    "Where is Apple Inc. headquartered?",
    "Who founded Apple?",
    "What are Apple’s popular products?"
]

# === Get answers ===
print("🔍 Answers:")
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Q: {question}\nA: {result['answer']} (Score: {result['score']:.2f})\n")

6. Information Retrieval System
import spacy
import os
from nltk.stem import PorterStemmer
from collections import defaultdict
import math

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

documents = {
    "doc1": "Information Retrieval is a technique to search relevant documents.",
    "doc2": "Retrieval systems use indexing to improve search efficiency.",
    "doc3": "Search engines use natural language processing to understand queries.",
}

def preprocess(text):
    doc = nlp(text.lower())
    return [
        stemmer.stem(token.text)
        for token in doc
        if not token.is_punct and not token.is_stop and token.is_alpha
    ]

inverted_index = defaultdict(set)
preprocessed_docs = {}

for doc_id, content in documents.items():
    tokens = preprocess(content)
    preprocessed_docs[doc_id] = tokens
    for token in tokens:
        inverted_index[token].add(doc_id)

def search(query):
    query_tokens = preprocess(query)
    matched_docs = None
    for token in query_tokens:
        docs_with_token = inverted_index.get(token, set())
        if matched_docs is None:
            matched_docs = docs_with_token
        else:
            matched_docs = matched_docs.intersection(docs_with_token)
    return matched_docs if matched_docs else set()

while True:
    query = input("\nEnter your search query (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    results = search(query)
    if results:
        print("Documents matching query:")
        for doc_id in results:
            print(f"{doc_id}: {documents[doc_id]}")
    else:
        print("No documents found.")
7. Positional Encoding in GPT
import torch, math

def positional_encoding(seq_len, d_model):
    pos_enc = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
    return pos_enc

EMB = {
    "The": torch.tensor([0.2, 0.5, 0.1, 0.7]),
    "cat": torch.tensor([0.3, 0.6, 0.8, 0.1]),
    "sleeps": torch.tensor([0.4, 0.9, 0.2, 0.5]),
}

def encode_sentence(tokens):
    X = torch.stack([EMB[t] for t in tokens])
    P = positional_encoding(seq_len=len(tokens), d_model=X.size(1))
    E = X + P
    return X, P, E

tokens_A = ["The", "cat", "sleeps"]
X_A, P_A, E_A = encode_sentence(tokens_A)

tokens_B = ["sleeps", "The", "cat"]
X_B, P_B, E_B = encode_sentence(tokens_B)

print("Tokens A:", tokens_A)
print("Query tokens in order:\n", E_A)
8. Coreference Resolution using Transformers
!pip install -q fastcoref
import re
from fastcoref import FCoref
text = "Alice met Bob after she left the office. He told her the report was ready, and Alice thanked him."
model = FCoref()
res = model.predict(texts=[text])[0]
print("Clusters:")
for c in res.get_clusters(as_strings=True):
    print(c)
resolved = text
def pick_rep(mentions):
    proper = [m for m in mentions if re.match(r"[A-Z].*", m)]
    if proper:
        return max(proper, key=len)
    return max(mentions, key=len)
for mentions in res.get_clusters(as_strings=True):
    rep = pick_rep(mentions)
    for m in sorted(set(mentions), key=len, reverse=True):
        if m == rep:
            continue
        pattern = r"\b" + re.escape(m) + r"\b"
        resolved = re.sub(pattern, rep, resolved, flags=re.IGNORECASE)
print("\nResolved text:\n", resolved)

9. Chatbot using DialoGPT
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
model.eval()
print("GPT-2 Chatbot (type 'exit' to quit)")
chat_history = ""
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat_history += f"User: {user_input}\nBot:"
    input_ids = tokenizer.encode(chat_history, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    bot_reply = response.split("Bot:")[-1].strip().split("User:")[0].strip()
    print(f"Bot: {bot_reply}")
    chat_history += f" {bot_reply}\n"
