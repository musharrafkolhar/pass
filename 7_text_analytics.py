import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
 
# Download Required NLTK Resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
 
# Sample Text
text = "Naturally Language Processing is very better interesting and useful."
print("\nOriginal Text:")
print(text)
 
# 1. Tokenization
tokens = word_tokenize(text)
print("\n1. Tokens:")
print(tokens)
 
# 2. POS Tagging
pos = pos_tag(tokens)
print("\n2. POS Tagging:")
print(pos)
 
# 3. Stopword Removal
stop_words = set(stopwords.words('english'))
filtered = [
    word for word in tokens
    if word.lower() not in stop_words
]
print("\n3. After Stopword Removal:")
print(filtered)
 
# 4. Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered]
print("\n4. Stemmed Words:")
print(stemmed)
 
# 5. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [
    lemmatizer.lemmatize(word)
    for word in filtered
]
print("\n5. Lemmatized Words:")
print(lemmatized)
 
# 6. Convert Processed Words to Text
processed_text = " ".join(lemmatized)
print("\n6. Processed Text:")
print(processed_text)
 
# 7. TF-IDF Vectorization
# Multiple documents for better TF-IDF
corpus = [
    processed_text,
    "Text analytics is very useful in data science.",
    "Natural language processing and text analytics are related."
]
 
# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
 
# Fit and Transform
tfidf_matrix = vectorizer.fit_transform(corpus)
 
# Extract Feature Names
feature_names = vectorizer.get_feature_names_out()
 
# Convert TF-IDF matrix to DataFrame
df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names
)
 
print("\n7. TF-IDF Matrix:")
print(df_tfidf)