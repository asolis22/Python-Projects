import nltk
from nltk import NaiveBayesClassifier
from nltk.classify import apply_features

# Data Preparation 
data = [
    ("hello", "English"),
    ("word", "English"),
    ("machine", "English"),
    ("hola", "Spanish"),
    ("mundo", "Spanish"),
    ("aprendizaje", "Spanish")
]

# Feature Extraction: 
def extract_featrues(word):
    return {char: (char in word) for char in set(word)}

featuresets = [(extract_featrues(word), lang) for word, lang in data]

# Training: 

classifier = NaiveBayesClassifier.train(featuresets)


# Preditiction: 

test_words = ["hello", "amigo", "machine", "amigo"]

for word in test_words: 
    features = extract_featrues(word)
    lang = classifier.classify(features)
    print(f"{word} is likely in {lang}")