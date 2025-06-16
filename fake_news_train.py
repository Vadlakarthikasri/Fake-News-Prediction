import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib


df = pd.read_csv("fake_or_real_news.csv")  


df = df[['text', 'label']]
df.dropna(inplace=True)


df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)


model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', LogisticRegression())
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))


joblib.dump(model, "fake_news_model.pkl")
print("Model saved as fake_news_model.pkl")
