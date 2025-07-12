import io
import pandas as pd
import streamlit as st
from collections import Counter
from wordcloud import WordCloud
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ページのレイアウトを設定
st.set_page_config(
    page_title="テキスト可視化",
    layout="wide", # wideにすると横長なレイアウトに
    initial_sidebar_state="expanded"
)

# タイトルの設定
st.title("テキスト可視化")
st.markdown("""
テキストの文章を読み込ませると、スペースごとに文字を分割し、出現頻度を集計します
""")

# サイドバーにアップロードファイルのウィジェットを表示
st.sidebar.markdown("# ファイルアップロード")
uploaded_file = st.sidebar.file_uploader(
    "テキストファイルをアップロードしてください", type="txt"
)

if uploaded_file is not None:
    # テキストファイルの読み込み
    text_data = uploaded_file.read().decode('utf-8')
    
    # データフレームの作成
    data = {'text': text_data.split('\n'), 'label': ['']*len(text_data.split('\n'))}
    df = pd.DataFrame(data)
    
    # データの表示
    st.write("データフレーム:")
    st.write(df)
    
    # テキストの前処理
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(df['text'])
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, df['label'], test_size=0.2, random_state=42)
    
    # モデルの学習
    clf = MultinomialNB().fit(X_train, y_train)
    
    # モデルの評価
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("モデルの正解率:", accuracy)

    
    # テキストデータの統計情報
    word_counts = X_train_counts.toarray().sum(axis=0)
    words = count_vect.get_feature_names_out()
    word_freq = pd.DataFrame({'word': words, 'count': word_counts}).sort_values(by='count', ascending=False)
    
    # ワードクラウドの作成と表示
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(dict(zip(words, word_counts)))
    
# グラフの表示  
    fig, ax = plt.subplots()
    word_freq.head(10).plot(kind='bar', x='word', y='count', ax=ax)
    st.pyplot(fig)


    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    st.pyplot(plt)

else:
    # テキスト未アップロード時の処理
    st.write("テキストファイルをアップロードしてください。")
