import streamlit as st
import streamlit.components.v1 as components

import plotly.express as px
import plotly.graph_objects as go

from collections import Counter

from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from wordcloud import WordCloud
########################################################################################################################
def get_count_top_words(df, start_date=None, last_date=None, num_words=10, name=None, sentiment = None, item = None, source = None ):
    if name is not None:
        df = df[df['name'] == name]
    if sentiment is not None:
        df = df[df['sentiment'] == sentiment]
    if item is not None:
        df = df[df['item'] == item]
    if source is not None:
        df = df[df['source'] == source]
    if start_date is None:
        start_date = df['time'].min().strftime('%Y-%m-%d')
    if last_date is None:
        last_date = df['time'].max().strftime('%Y-%m-%d')
    df = df[(df['time'] >= start_date) & (df['time'] <= last_date)]
    count_vectorizer = CountVectorizer()
    count = count_vectorizer.fit_transform(df['noun'].values)
    count_df = pd.DataFrame(count.todense(), columns=count_vectorizer.get_feature_names_out())
    count_top_words = count_df.sum().sort_values(ascending=False).head(num_words).to_dict()
    return count_top_words
########################################################################################################################
# 데이터 로드
df_리뷰_감성분석결과 = pd.read_csv('/app/streamlit/data/리뷰6차.csv')
df_리뷰_감성분석결과['time'] = pd.to_datetime(df_리뷰_감성분석결과['time'])

words = get_count_top_words(df_리뷰_감성분석결과)
########################################################################################################################
# 레이아웃
# with st.container():
#     col1, col2, col3 = st.columns([1,1,1])
# with st.container():
#     col4, col5, col6 = st.columns([1,1,1])

tab1, tab2, tab3 = st.tabs(["All", "Positive😊", "Negative😫"])

with tab1:
    st.header("모든 리뷰")


with st.expander('=== 기업선택하기 ==='):

    st.write(df_리뷰_감성분석결과['name'].unique())







########################################################################################################################
# 파이차트
with col1:
    df_파이차트 = pd.DataFrame(df_리뷰_감성분석결과['sentiment'].value_counts())
    pie_chart = go.Figure(data=[go.Pie(labels=list(df_파이차트.index), values=df_파이차트['count'])])
    st.plotly_chart(pie_chart, use_container_width=True)
########################################################################################################################
# 워드클라우드
with col2:
    cand_mask = np.array(Image.open('/app/streamlit/data/circle.png'))
    워드클라우드 = WordCloud(
        background_color="white", 
        max_words=1000,
        font_path = "/app/streamlit/font/NanumBarunGothic.ttf", 
        contour_width=3, 
        colormap='Spectral', 
        contour_color='white',
        mask=cand_mask).generate_from_frequencies(words)
    fig, ax = plt.subplots()
    ax.imshow(워드클라우드, interpolation='bilinear')
    st.pyplot(fig, use_container_width=True)
########################################################################################################################
# 바차트
with col3:
    st.bar_chart(words)
########################################################################################################################
# 바차트
with col5:
    st.bar_chart(words)
########################################################################################################################
# 네트워크 차트
from gensim.models import Word2Vec
import networkx as nx
from pyvis.network import Network

st.title('키워드 네트워크 파악')
keywords = st.text_imput('키워드를 입력해주세요', value='제라늄')


keywords = ['뿌리','제라늄', '식물', '응애']

reviews = [eval(i) for i in df_리뷰_감성분석결과['noun']]

networks = []
for review in reviews:
    network_review = [w for w in review if len(w) > 1]
    networks.append(network_review)

model = Word2Vec(networks, vector_size=100, window=5, min_count=1, workers=4, epochs=100)

G = nx.Graph(font_path='/app/streamlit/font/NanumBarunGothic.ttf')

# 중심 노드들을 노드로 추가
for keyword in keywords:
    G.add_node(keyword)
    # 주어진 키워드와 가장 유사한 20개의 단어 추출
    similar_words = model.wv.most_similar(keyword, topn=20)
    # 유사한 단어들을 노드로 추가하고, 주어진 키워드와의 연결선 추가
    for word, score in similar_words:
        G.add_node(word)
        G.add_edge(keyword, word, weight=score)
        
# 노드 크기 결정
size_dict = nx.degree_centrality(G)

# 노드 크기 설정
node_size = []
for node in G.nodes():
    if node in keywords:
        node_size.append(5000)
    else:
        node_size.append(1000)

# 클러스터링
clusters = list(nx.algorithms.community.greedy_modularity_communities(G))
cluster_labels = {}
for i, cluster in enumerate(clusters):
    for node in cluster:
        cluster_labels[node] = i
        
# 노드 색상 결정
color_palette = ["#f39c9c", "#f7b977", "#fff4c4", "#d8f4b9", "#9ed6b5", "#9ce8f4", "#a1a4f4", "#e4b8f9", "#f4a2e6", "#c2c2c2"]
node_colors = [color_palette[cluster_labels[node] % len(color_palette)] for node in G.nodes()]

# 노드에 라벨과 연결 강도 값 추가
edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]

# 선의 길이를 변경 pos
plt.figure(figsize=(15,15))
pos = nx.spring_layout(G, seed=42, k=0.15)
nx.draw(G, pos, font_family='NanumGothic', with_labels=True, node_size=node_size, node_color=node_colors, alpha=0.8, linewidths=1,
        font_size=9, font_color="black", font_weight="medium", edge_color="grey", width=edge_weights)


# 중심 노드들끼리 겹치는 단어 출력
overlapping_keywords = set()
for i, keyword1 in enumerate(keywords):
    for j, keyword2 in enumerate(keywords):
        if i < j and keyword1 in G and keyword2 in G:
            if nx.has_path(G, keyword1, keyword2):
                overlapping_keywords.add(keyword1)
                overlapping_keywords.add(keyword2)
if overlapping_keywords:
    print(f"다음 중심 키워드들끼리 연관성이 있어 중복될 가능성이 있습니다: {', '.join(overlapping_keywords)}")


net = Network(notebook=True, cdn_resources='in_line')

net.from_nx(G)

net.save_graph(f'/app/streamlit/pyvis_graph.html')
HtmlFile = open(f'/app/streamlit/pyvis_graph.html', 'r', encoding='utf-8')
components.html(HtmlFile.read(), height=435)

st.success(f'<{}>')



########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
