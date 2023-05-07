import streamlit as st
import streamlit.components.v1 as components

import plotly.express as px
import plotly.graph_objects as go

# 기본 라이브러리
import os
import ast
from datetime import datetime
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore", message="PyplotGlobalUseWarning")

from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

from PIL import Image

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from gensim.models import Word2Vec
import networkx as nx
import gensim
from pyvis.network import Network
from wordcloud import WordCloud
########################################################################################################################
# 데이터 로드 상수
df_리뷰_감성분석결과 = pd.read_csv('/app/streamlit/data/리뷰6차.csv')
df_리뷰_감성분석결과['time'] = pd.to_datetime(df_리뷰_감성분석결과['time'])
 

긍부정 = st.radio(
    "🍀리뷰 선택🍀",
    ('All', '긍정리뷰😊', '부정리뷰😫'), horizontal=True)
if 긍부정 == 'All':
    긍부정마스크 = ((df_리뷰_감성분석결과['sentiment'] == '긍정') | (df_리뷰_감성분석결과['sentiment'] == '부정'))
if 긍부정 == '긍정리뷰😊':
    긍부정마스크 = (df_리뷰_감성분석결과['sentiment'] == '긍정')
if 긍부정 == '부정리뷰😫':
    긍부정마스크 = (df_리뷰_감성분석결과['sentiment'] == '부정')

option = st.selectbox(
    '🍀단어기준선택🍀',
    ('빈도(Count)', '중요도(TF-IDF)'))
st.write('선택기준: ', option)


품사옵션 = st.selectbox(
    '🍀품사선택🍀',
    ('명사', '명사+동사+형용사'))
st.write('선택품사: ', 품사옵션)


회사종류 = st.selectbox(
    '🍀제품선택🍀',
    ('자사+경쟁사', '꽃피우는 시간', '경쟁사-식물영양제', 
        '경쟁사-뿌리영양제', 
        '경쟁사-살충제',
        '경쟁사-식물등',
        '경쟁사All',
        ))
st.write('선택제품: ', 회사종류)
if 회사종류 == '자사+경쟁사':
    회사종류마스크 = ((df_리뷰_감성분석결과['name'] == '경쟁사') | (df_리뷰_감성분석결과['name'] == '꽃피우는시간'))
if 회사종류 == '꽃피우는 시간':
    회사종류마스크 = (df_리뷰_감성분석결과['name'] == '꽃피우는시간')
if 회사종류 == '경쟁사-식물영양제':
    회사종류마스크 = ((df_리뷰_감성분석결과['name'] == '경쟁사') & (df_리뷰_감성분석결과['item'] == '식물영양제'))
if 회사종류 == '경쟁사-뿌리영양제':
    회사종류마스크 = ((df_리뷰_감성분석결과['name'] == '경쟁사') & (df_리뷰_감성분석결과['item'] == '뿌리영양제'))
if 회사종류 == '경쟁사-살충제':
    회사종류마스크 = ((df_리뷰_감성분석결과['name'] == '경쟁사') & (df_리뷰_감성분석결과['item'] == '살충제'))
if 회사종류 == '경쟁사-식물등':
    회사종류마스크 = ((df_리뷰_감성분석결과['name'] == '경쟁사') & (df_리뷰_감성분석결과['item'] == '식물등'))
if 회사종류 == '경쟁사All':
    회사종류마스크 = (df_리뷰_감성분석결과['name'] == '경쟁사')

시작날짜 = df_리뷰_감성분석결과['time'][회사종류마스크].min()
마지막날짜 = df_리뷰_감성분석결과['time'][회사종류마스크].max()


start_date = st.date_input(
    '🍀시작날짜🍀',
    value=시작날짜,
    min_value=시작날짜,
    max_value=마지막날짜
)

end_date = st.date_input(
    '🍀마지막날짜🍀',
    value=마지막날짜,
    min_value=시작날짜,
    max_value=마지막날짜
)

기간마스크 = ((df_리뷰_감성분석결과['time'] >= pd.to_datetime(start_date)) & (df_리뷰_감성분석결과['time'] <= pd.to_datetime(end_date)))
마스크된데이터프레임 = df_리뷰_감성분석결과[긍부정마스크 & 기간마스크 & 회사종류마스크]

# 키워드 = st.text_input('🍀네트워크 단어입력🍀', '제라늄')

# if 키워드.find(',') == -1:
#     st.write('예시 : 뿌리, 제라늄, 식물, 응애')
#     키워드 = [키워드]

# if 품사옵션 == '명사':
#     품사 = 'noun'
# if 품사옵션 == '명사+동사+형용사':
#     품사 = 'n_v_ad'

키워드 = ['응애', '제라늄', '스킨답서스']

reviews = [eval(i) for i in 마스크된데이터프레임[품사]]
def 네트워크(reviews):
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
    # plt.figure(figsize=(15,15))
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
    return [net, similar_words]

네트워크 = 네트워크(reviews)
네트워크

# try:
#     net = 네트워크[0]
#     net.save_graph(f'/app/streamlit/pyvis_graph.html')
#     HtmlFile = open(f'/app/streamlit/pyvis_graph.html', 'r', encoding='utf-8')
#     components.html(HtmlFile.read(), height=435)
# except:
#     st.write('존재하지 않는 키워드예요.')