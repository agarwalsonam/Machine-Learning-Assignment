import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

os.listdir(r'C:\Users\Sonam\Desktop\DA\git\ML Assign')
df_train=pd.read_csv(r'C:\Users\Sonam\Desktop\DA\git\ML Assign\train.csv')
#df_test=pd.read_csv(r'C:\Users\Sonam\Desktop\DA\intern\knight\test.csv')

print(df_train.head())

print(df_train.shape)

print(df_train.dtypes)

stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=20,
        scale = 3,
        random_state = 1).generate(str(data))
    
    fig = plt.figure(1, figsize=(15,20))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df_train['review_title'])

all_text=df_train['review_title']
rev_title_train=df_train['review_title']
y=df_train['variety']

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(rev_title_train)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(rev_title_train)

train_features = hstack([train_char_features, train_word_features])
X_train, X_test, y_train, y_test = train_test_split(train_features, y,test_size=0.3,random_state=101)


classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)
preds=classifier.predict(X_test)

variety1 = pd.DataFrame(preds)

user_name = df_train['user_name']
review_title = df_train['review_title']
review_description = df_train['review_description']
points = df_train['points']
price = df_train['price']
variety = df_train['variety']

outcome = pd.concat([user_name,review_title,review_description,points,price,variety,variety1], axis=1)
outcome.columns = ['user_name','review_title','review_description','points','price','variety','variety1']
outcome.to_csv('outcome1.csv',encoding='utf-8', columns=['user_name','review_title','review_description','points','price','variety','variety1'], index=False)

rf_accuracy=accuracy_score(preds,y_test)

print(rf_accuracy)

