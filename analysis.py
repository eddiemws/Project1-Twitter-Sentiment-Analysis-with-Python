import tweepy  # Twitter API access
from nltk.corpus import stopwords  # Stopword removal
from nltk.stem import WordNetLemmatizer  # Lemmatization
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF
from sklearn.ensemble import RandomForestClassifier  # Machine Learning model
from gensim.models import Word2Vec  # Word embeddings (optional)
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Advanced visualization




# Function to authenticate Twitter API
def authenticate_twitter():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api

# Function to clean and preprocess tweets
def preprocess_tweet(text):
    # Lowercase text
    text = text.lower()
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9_]", " ", text)
    # Remove stopwords
    stop_words = stopwords.words('english')
    text = [word for word in text.split() if word not in stop_words]
    # Lemmatization (optional)
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

# Function to extract features using TF-IDF
def extract_tfidf_features(tweets):
    vectorizer = TfidfVectorizer(max_features=2000)
    features = vectorizer.fit_transform(tweets)
    return features

# Function to train a Random Forest Classifier model (optional)
def train_sentiment_model(tweets, labels):
    features = extract_tfidf_features(tweets)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(features, labels)
    return model

# Function to perform sentiment analysis on a tweet
def analyze_tweet_sentiment(model, tweet):
    preprocessed_tweet = preprocess_tweet(tweet)
    features = extract_tfidf_features([preprocessed_tweet])
    prediction = model.predict(features)[0]
    if prediction == 0:
        return "Negative"
    elif prediction == 1:
        return "Positive"
    else:
        return "Neutral"

# Function to visualize sentiment distribution (using Seaborn)
def visualize_sentiment_distribution(labels):
    sns.countplot(labels)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Distribution of Tweets")
    plt.show()

# Function to create a word cloud (using advanced libraries)
def create_wordcloud(tweets, positive_words, negative_words):
    from wordcloud import WordCloud  # Requires additional installation

    all_words = ' '.join(tweets)
    wordcloud = WordCloud(width=800, height=600, background_color="white").generate(all_words)
    positive_frequencies = {word: wordcloud.word_frequencies.get(word, 0) for word in positive_words}
    negative_frequencies = {word: wordcloud.word_frequencies.get(word, 0) for word in negative_words}

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(wordcloud, interpolation='bilinear')
    axs[0].set_title('Overall Word Frequency')
    axs[0].axis('off')

    positive_cloud = WordCloud(width=400, height=300, background_color="white").generate(' '.join(positive_words))
    axs[1].imshow(positive_cloud, interpolation='bilinear')
    axs[1].set_title('Positive Words')
    axs[1].axis('off')

    negative_cloud = WordCloud(width=40
