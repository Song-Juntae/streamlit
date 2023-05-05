import streamlit as st
import streamlit.components.v1 as components

import plotly.express as px
import plotly.graph_objects as go

# 기본 라이브러리
import os

from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

from PIL import Image

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from gensim.models import Word2Vec
import networkx as nx
from pyvis.network import Network
from wordcloud import WordCloud
########################################################################################################################
# 데이터 로드 상수
df_리뷰_감성분석결과 = pd.read_csv('/app/streamlit/data/리뷰6차.csv')
df_리뷰_감성분석결과['time'] = pd.to_datetime(df_리뷰_감성분석결과['time'])

stopwords = ['언늘', '결국', '생각', '후기', '감사', '진짜', '완전', '사용', '요즘', '정도', '이번', '달리뷰', '결과', 
             '지금', '동영상', '조금', '안테', '입제', '영상', '이번건', '며칠', '이제', '거시기', '얼듯', '처음', '다음']
########################################################################################################################
# 레이아웃
with st.container():
    col0_1, col0_2, col0_3, col0_4, col0_4 = st.columns([1,1,1,1,1])
with st.container():
    col1_1, col1_2, col1_3, col1_4 = st.columns([1,1,1,1])
with st.container():
    col2_1, col2_2 = st.columns([1,1])
with st.container():
    col3_1, col3_2, col3_3, col3_4  = st.columns([1,1,1,1])
with st.container():
    col4_1, col4_2, col4_3 = st.columns([1,1,2])
########################################################################################################################
# 사용자 입력
with col0_3:
    긍부정 = st.radio(
    "긍정 부정 선택",
    ('All', 'Positive', 'Negative'), horizontal=True)
if 긍부정 == 'All':
    긍부정마스크 = ((df_리뷰_감성분석결과['sentiment'] == '긍정') | (df_리뷰_감성분석결과['sentiment'] == '부정'))
if 긍부정 == 'Positive':
    긍부정마스크 = (df_리뷰_감성분석결과['sentiment'] == '긍정')
if 긍부정 == 'Negative':
    긍부정마스크 = (df_리뷰_감성분석결과['sentiment'] == '부정')

with col1_1:
    option = st.selectbox(
        '고르세요',
        ('카운트', 'td-idf'))
    st.write('이것: ', option)

with col1_2:
    품사옵션 = st.selectbox(
        '고르세요',
        ('명사', '동사+형용사', '명사+동사+형용사'))
    st.write('이것: ', 품사옵션)

with col1_3:
    회사종류 = st.selectbox(
        '고르세요',
        ('자사+경쟁사', '꽃피우는 시간', '경쟁사-식물영양제', '경쟁사-뿌리영양제', '경쟁사-살충제', '경쟁사-식물등', '경쟁사All'))
    st.write('이것: ', 회사종류)

with col1_4:
    ''

with col3_1:
    추가불용어 = st.text_input('불용어를 추가하세요', '')
    if 추가불용어 == '':
        st.write('예시 : 영양제, 식물, 배송')
    if 추가불용어 != '':
        st.write('추가된 불용어: ', 추가불용어)

with col3_2:
    단어수 = st.slider(
        '단어 수를 조정하세요',
        10, 300, step=1)
    st.write('단어수: ', 단어수)

if 추가불용어.find(',') != -1:
    stopwords.extend([i.strip() for i in 추가불용어.split(',')])
if 추가불용어.find(',') == -1:
    stopwords.append(추가불용어) 

with col3_3:
    키워드 = st.text_input('키워드를 입력해주세요', '제라늄')
    if 키워드.find(',') == -1:
        st.write('예시 : 뿌리, 제라늄, 식물, 응애')
        키워드 = [키워드]
    elif 키워드.find(',') != -1:
        st.write('설정된 키워드: ', 키워드)
        키워드 = [i.strip() for i in 키워드.split(',')]
    else:
        st.write('문제가 생겼어요.')
     
########################################################################################################################
def get_count_top_words(df, start_date=None, last_date=None, num_words=10, name=None, sentiment = None, item = None, source = None , 품사='noun'):
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
    count_vectorizer = CountVectorizer(stop_words=stopwords)
    count = count_vectorizer.fit_transform(df[품사].values)
    count_df = pd.DataFrame(count.todense(), columns=count_vectorizer.get_feature_names_out())
    count_top_words = count_df.sum().sort_values(ascending=False).head(num_words).to_dict()
    return count_top_words

def get_tfidf_top_words(df, start_date=None, last_date=None, num_words=10, name=None, sentiment = None, item = None, source = None, 품사='noun' ):
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
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf = tfidf_vectorizer.fit_transform(df[품사].values)
    tfidf_df = pd.DataFrame(tfidf.todense(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_top_words = tfidf_df.sum().sort_values(ascending=False).head(num_words).to_dict()
    return tfidf_top_words
########################################################################################################################
if 품사옵션 == '명사':
    품사 = 'noun'
if 품사옵션 == '동사+형용사':
    품사 = 'v_ad'
if 품사옵션 == '명사+동사+형용사':
    품사 = 'n_v_ad'

카운트 = get_count_top_words(df_리뷰_감성분석결과, num_words=단어수, 품사=품사)
tdidf = get_tfidf_top_words(df_리뷰_감성분석결과, num_words=단어수, 품사=품사)

if option == '카운트':
    words = 카운트
if option == 'td-idf':
    words = tdidf
########################################################################################################################
# 파이차트
with col4_1:
    df_파이차트 = pd.DataFrame(df_리뷰_감성분석결과['sentiment'].value_counts())
    pie_chart = go.Figure(data=[go.Pie(labels=list(df_파이차트.index), values=df_파이차트['count'])])
    st.plotly_chart(pie_chart, use_container_width=True)
with col4_2:
    # st.plotly_chart(words)
    바차트 = go.Figure([go.Bar(x=list(words.keys()),y=list(words.values()))])
    st.plotly_chart(바차트, use_container_width=True)
########################################################################################################################
# 워드클라우드
with col2_1:
    cand_mask = np.array(Image.open('/app/streamlit/data/circle.png'))
    워드클라우드 = WordCloud(
        background_color="white", 
        max_words=1000,
        font_path = "/app/streamlit/font/NanumBarunGothic.ttf", 
        contour_width=3, 
        colormap='Spectral', 
        contour_color='white',
        # mask=cand_mask,
        width=800,
        height=400
        ).generate_from_frequencies(words)

    st.image(워드클라우드.to_array(), use_column_width=True)
########################################################################################################################
with col4_3:
    st.dataframe(df_리뷰_감성분석결과[['name','sentiment']])
########################################################################################################################
# 네트워크 차트

reviews = [eval(i) for i in df_리뷰_감성분석결과['noun']]

networks = []
for review in reviews:
    network_review = [w for w in review if len(w) > 1]
    networks.append(network_review)

model = Word2Vec(networks, vector_size=100, window=5, min_count=1, workers=4, epochs=100)

G = nx.Graph(font_path='/app/streamlit/font/NanumBarunGothic.ttf')

# 중심 노드들을 노드로 추가
for keyword in 키워드:
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
    if node in 키워드:
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
overlapping_키워드 = set()
for i, keyword1 in enumerate(키워드):
    for j, keyword2 in enumerate(키워드):
        if i < j and keyword1 in G and keyword2 in G:
            if nx.has_path(G, keyword1, keyword2):
                overlapping_키워드.add(keyword1)
                overlapping_키워드.add(keyword2)
if overlapping_키워드:
    print(f"다음 중심 키워드들끼리 연관성이 있어 중복될 가능성이 있습니다: {', '.join(overlapping_키워드)}")


net = Network(notebook=True, cdn_resources='in_line')

net.from_nx(G)

net.save_graph(f'/app/streamlit/pyvis_graph.html')
HtmlFile = open(f'/app/streamlit/pyvis_graph.html', 'r', encoding='utf-8')

with col2_2:
    components.html(HtmlFile.read(), height=435)

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
import streamlit as st
from pyvis.network import Network
import networkx as nx
from gensim.models import Word2Vec


def show_network_finish(df, keywords, num_words=20, name=None, sentiment=None, item=None, source=None):
    
    if name is not None:
        df = df[df['name'] == name]
    if sentiment is not None:
        df = df[df['sentiment'] == sentiment]
    if item is not None:
        df = df[df['item'] == item]
    if source is not None:
        df = df[df['source'] == source]
    
    # 문장 리스트 생성
    reviews = [eval(i) for i in df['n_v_ad']]

    # 단어 인코딩 및 빈도수 계산
    networks = []
    for review in reviews:
        network_review = [w for w in review if len(w) > 1]
        networks.append(network_review)

    # 유사도 계산을 위한 Graph 생성
    G = nx.Graph()
    for review in networks:
        for i, word in enumerate(review):
            if word not in G:
                G.add_node(word)
            if i < len(review) - 1:
                next_word = review[i+1]
                if not G.has_edge(word, next_word):
                    G.add_edge(word, next_word, weight=0)
                G[word][next_word]['weight'] += 1
    # 중심 노드 계산
    pagerank = nx.pagerank(G)
    central_nodes = [node for node, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)][:3]

    # Word2Vec 모델 학습
    model = Word2Vec(networks, vector_size=100, window=5, min_count=1, workers=4, epochs=100)
   
    # 그래프 생성
    net = Network(width='100%', height='750px')

    # 중심 노드들을 노드로 추가
    for keyword in central_nodes:
        net.add_node(keyword, title=keyword)

        # 주어진 키워드와 가장 유사한 20개의 단어 추출
        similar_words = model.wv.most_similar(keyword, topn=num_words)

        # 유사한 단어들을 노드로 추가하고, 주어진 키워드와의 연결선 추가
        for word, score in similar_words:
            net.add_node(word, title=word)
            net.add_edge(keyword, word, value=score)

    # 노드 크기 결정
    size_dict = nx.degree_centrality(G)
    
    # 노드 크기 설정
    node_size = []
    for node in net.nodes:
        if node['id'] in central_nodes:
            node_size.append(5000)
        else:
            node_size.append(1000)

    # 클러스터링
    clusters = list(nx.algorithms.community.greedy_modularity_communities(G))
    cluster_labels = {}
    for i, cluster in enumerate(clusters):
        for node in cluster:
            cluster_labels[node] = i
            
   # 노드 크기 결정
    size_dict = nx.degree_centrality(G)

    # 노드 크기 설정
    node_size = []
    for node in net.nodes:
        if node['id'] in central_nodes:
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
    node_colors = [color_palette[cluster_labels[node['id']] % len(color_palette)] for node in net.nodes]

    # 노드에 라벨과 연결 강도 값 추가
    for node in net.nodes:
        node['label'] = node['id']
        node['title'] += f" ({node['id']})"
        node['color'] = node_colors[net.nodes.index(node)]
        node['size'] = node_size[net.nodes.index(node)]
    for edge in net.edges:
        edge['title'] = f"{edge['from']} - {edge['to']}: {edge['value']}"
        edge['color'] = "#c2c2c2"

    # 그래프 출력
    net.show(name)

show_network_finish(df_리뷰_감성분석결과, 키워드)