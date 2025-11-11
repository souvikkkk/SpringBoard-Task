# Import necessary libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

# Sample text
text = "The hardworking students were studying and preparing for their final exams sincerely."

# Step 1: Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# Step 2: Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("\nAfter Stopword Removal:", filtered_tokens)

# Step 3: Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
print("\nAfter Stemming:", stemmed_words)

# Step 4: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\nAfter Lemmatization:", lemmatized_words)

# Step 5: POS Tagging
pos_tags = nltk.pos_tag(filtered_tokens)
print("\nPOS Tags:", pos_tags)
