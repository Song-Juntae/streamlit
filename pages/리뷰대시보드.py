import streamlit as st
import streamlit.components.v1 as components
from streamlit_tags import st_tags

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
df_리뷰_감성분석결과 = pd.read_csv('/app/streamlit/data/리뷰7차(수정).csv')
df_리뷰_감성분석결과['time'] = pd.to_datetime(df_리뷰_감성분석결과['time'])

# df_리뷰_감성분석결과['time'] = pd.to_datetime(df_리뷰_감성분석결과['time'], format='%Y-%m-%d')


stopwords = ['언늘', '결국', '생각', '후기', '감사', '진짜', '완전', '사용', '요즘', '정도', '이번', '달리뷰', '결과', 
             '지금', '동영상', '조금', '안테', '입제', '영상', '이번건', '며칠', '이제', '거시기', '얼듯', '처음', '다음',
             '합니다', '하는', '할', '하고', '한다','하다','되다','같다','자다','되다','있다','써다','않다','해보다','주다','되어다', 
             '그리고', '입니다', '그', '등', '이런', '및','제', '더','언늘','결국','생각','식물키',
             '감사','진짜','완전','요ㅎ','사용','정도','엄마','아이','원래','흐흐','하하','정말']

########################################################################################################################
# title
st.title('자사/경쟁사 리뷰 모니터링 대시보드')      
st.write("")
st.write("")
st.write("")
########################################################################################################################
# 레이아웃
#1_1 : 품사, 1_2 : 제품, 1_3 : 시작날짜, 1_4: 마지막 날짜
#2_1 : 워드 클라우드 세부 필터
#3_1,2 : 기준, 3_3,4 : 단어 수 조정
#4_1,2,3,4 : 포함X  단어
########################################################################################################################
# 워클, 넽웤 공통필터 레이아웃
# 0. 긍/부정
with st.container():
    col0_1, col0_2, col0_3 = st.columns([1,1,1])
# 1. 워클, 넽웤 공통 옵션
with st.container():
    col1_0, col1_1, col1_2, col1_3, col1_4 = st.columns([1,1,1,1,1])
########################################################################################################################
# 워클, 넽웤 공통필터


with col0_1:
    st.markdown('🎚️기본 설정')

with col1_0:
    회사종류 = st.selectbox(
        '제품',
        ('자사+경쟁사', '꽃피우는 시간', '경쟁사-식물영양제', 
         '경쟁사-뿌리영양제', 
         '경쟁사-살충제',
         '경쟁사-식물등',
         '경쟁사 전체',
         ))
    # st.write('선택제품: ', 회사종류)
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
    if 회사종류 == '경쟁사 전체':
        회사종류마스크 = (df_리뷰_감성분석결과['name'] == '경쟁사')

with col1_1:
    # st.secrets['API_KEY']
    긍부정 = st.selectbox(
    "리뷰 유형", ('전체', '긍정', '부정'))
if 긍부정 == '전체':
    긍부정마스크 = ((df_리뷰_감성분석결과['sentiment'] == '긍정') | (df_리뷰_감성분석결과['sentiment'] == '부정'))
if 긍부정 == '긍정':
    긍부정마스크 = (df_리뷰_감성분석결과['sentiment'] == '긍정')
if 긍부정 == '부정':
    긍부정마스크 = (df_리뷰_감성분석결과['sentiment'] == '부정')

with col1_2:
    품사옵션 = st.selectbox(
        '키워드 유형',
        ('명사', '명사+동사+형용사'))
    # st.write('선택품사: ', 품사옵션)


시작날짜 = df_리뷰_감성분석결과['time'][회사종류마스크].min()
마지막날짜 = df_리뷰_감성분석결과['time'][회사종류마스크].max()

with col1_3:
    start_date = st.date_input(
        '시작 날짜',
        value=시작날짜,
        min_value=시작날짜,
        max_value=마지막날짜
    )
with col1_4:
    end_date = st.date_input(
        '끝 날짜',
        value=마지막날짜,
        min_value=시작날짜,
        max_value=마지막날짜
    )

########################################################################################################################
# 워클 세부 필터
# # 2,3. 워클 세부 필터
# with st.container():
#     col2_1, col2_2= st.columns([1,1])
# # 3. 워클 세부 필터
# with st.container():
#     col3_1, col3_2= st.columns([1,1])
st.write("")
st.write("")
st.write("")

st.subheader('**🔎 중요 키워드 발굴**')
expander = st.expander('세부필터')
with expander:
    col2_1, col2_2= st.beta_columns(2)    
    with col2_1:
        option = st.selectbox(
            '기준',
            ('빈도(Count)', '상대 빈도(TF-IDF)'), help='**도움말**\n\n'
                    'Count: 단어의 빈도 순으로 크기를 설정합니다.\n\n'
                    'TF-IDF: 전체 리뷰 내 빈도와 개별 리뷰 내 빈도를 모두 고려해 크기를 설정합니다.')
        # st.write('선택기준: ', option)

    with col2_2:
        단어수 = st.slider(
            '키워드 수',
            10, 300, step=1)
        # st.write('단어수: ', 단어수)
   
    stopwords = st_tags(
        label = '제거할 키워드',
        text = '직접 입력해보세요',
        value = ['식물', '효과', '배송'],
        suggestions = ['식물', '효과', '배송'],
        key = '1')

# 4. 워클 + 바차트
with st.container():
    col4_1, col4_2 = st.columns([2,2])

# with col2_1:
#     option = st.selectbox(
#         '🍀단어기준선택🍀',
#         ('단순 빈도(Countvecterize)', '상대 빈도(TF-IDF)'))
#     st.write('선택기준: ', option)

# with col2_2:
#     단어수 = st.slider(
#         '🍀단어 수 조정하기🍀',
#         10, 300, step=1)
#     st.write('단어수: ', 단어수)

# with col3_1:
#     추가불용어 = st.text_input('🍀포함하지 않을 단어입력🍀', '')
#     if 추가불용어 == '':
#         st.write('예시 : 영양제, 식물, 배송')
#     if 추가불용어 != '':
#         st.write('제거한 단어: ', 추가불용어)

########################################################################################################################
# 워드 클라우드 
def get_count_top_words(df, start_date=None, last_date=None, num_words=200, name=None, sentiment = None, item = None, source = None , 품사='noun'):
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

def get_tfidf_top_words(df, start_date=None, last_date=None, num_words=200, name=None, sentiment = None, item = None, source = None, 품사='noun' ):
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


기간마스크 = ((df_리뷰_감성분석결과['time'] >= pd.to_datetime(start_date)) & (df_리뷰_감성분석결과['time'] <= pd.to_datetime(end_date)))


# if 추가불용어.find(',') != -1:
#     stopwords.extend([i.strip() for i in 추가불용어.split(',')])
# if 추가불용어.find(',') == -1:
#     stopwords.append(추가불용어) 

if 품사옵션 == '명사':
    품사 = 'noun'
if 품사옵션 == '명사+동사+형용사':
    품사 = 'n_v_ad'

마스크된데이터프레임 = df_리뷰_감성분석결과[긍부정마스크 & 기간마스크 & 회사종류마스크]
reviews = [eval(i) for i in 마스크된데이터프레임[품사]]

카운트 = get_count_top_words(마스크된데이터프레임, num_words=단어수, 품사=품사)
tdidf = get_tfidf_top_words(마스크된데이터프레임, num_words=단어수, 품사=품사)

if option == '빈도(Count)':
    words = 카운트
if option == '상대 빈도(TF-IDF)':
    words = tdidf

########################################################################################################################
# 워드클라우드
with col4_1:
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
        height=600
        ).generate_from_frequencies(words)

    st.image(워드클라우드.to_array(), use_column_width=True)

with col4_2:
    # st.plotly_chart(words)
    st.markdown('**키워드 빈도수**')
    바차트 = go.Figure([go.Bar(x=list(words.keys()),y=list(words.values()))])
    st.plotly_chart(바차트, use_container_width=True)
########################################################################################################################
st.write("")
st.write("")
st.write("")

st.subheader('**🔎연관 키워드 탐색**')

expander = st.expander('세부필터')
with expander:
        키워드 = st.text_input('궁금한 키워드', '식물')
        if 키워드 == '':
            키워드 = ['식물']
            st.write('키워드를 입력해주세요.')
            st.write(' 예시 : 뿌리, 제라늄, 식물, 응애')
            st.write('설정된 키워드: ', 키워드[0])
        elif 키워드.find(',') == -1:
            st.write('예시 : 뿌리, 제라늄, 식물, 응애')
            키워드 = [키워드]
        elif 키워드.find(',') != -1:
            st.write('설정된 키워드: ', 키워드)
            키워드 = [i.strip() for i in 키워드.split(',')]
        else:
            st.error('This is an error', icon="🚨")
# try:
#     키워드 = 키워드(standard_df, new_df)
# except:
#     st.warning("⚠️ 해당 기간 동안 신규 키워드가 존재하지 않습니다")

   

# # 5. 넽웤 세부필터
# with st.container():
#     col5_1, col5_2 = st.columns([1,1])
########################################################################################################################
# with col5_1:
#     키워드 = st.text_input('🍀네트워크 단어입력🍀', '제라늄')
#     if 키워드 == '':
#         키워드 = ['제라늄']
#         st.write('단어를 입력해주세요.')
#         st.write(' 예시 : 뿌리, 제라늄, 식물, 응애')
#         st.write('설정된 단어: ', 키워드[0])
#     elif 키워드.find(',') == -1:
#         st.write('예시 : 뿌리, 제라늄, 식물, 응애')
#         키워드 = [키워드]
#     elif 키워드.find(',') != -1:
#         st.write('설정된 단어: ', 키워드)
#         키워드 = [i.strip() for i in 키워드.split(',')]
#     else:
#         # st.write('문제가 생겼어요.')

########################################################################################################################
# 네트워크 차트

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
########################################################################################################################
# 6. 넽웤 + 파이차트
with st.container():
    col6_1, col6_2 = st.columns([3,1])

with col6_1:
    try:
        net = 네트워크[0]
        net.save_graph(f'/app/streamlit/pyvis_graph.html')
        HtmlFile = open(f'/app/streamlit/pyvis_graph.html', 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=435)
    except:
        st.write('존재하지 않는 키워드예요.')

########################################################################################################################
# 파이차트
with col6_2:
    st.markdown('**키워드 긍/부정 리뷰 비율**')
    if len(키워드) > 1:
        df_파이차트 = pd.DataFrame(마스크된데이터프레임['sentiment'][마스크된데이터프레임['review_sentence'].str.contains('|'.join(키워드))].value_counts())
    if len(키워드) == 1:
        df_파이차트 = pd.DataFrame(마스크된데이터프레임['sentiment'][마스크된데이터프레임['review_sentence'].str.contains(키워드[0])].value_counts())
    pie_chart = go.Figure(data=[go.Pie(labels=list(df_파이차트.index), values=df_파이차트['count'])])
    st.plotly_chart(pie_chart, use_container_width=True)
########################################################################################################################
# 7. 넽웤 데이터 프레임
# with st.container():
#     col7_1, col7_2 = st.columns([3,1])

expander = st.expander('키워드가 포함된 리뷰')
with expander:
    if len(키워드) == 1:
        보여줄df = 마스크된데이터프레임[마스크된데이터프레임['noun'].str.contains(키워드[0])]
        st.dataframe(보여줄df[['name','sentiment','review_sentence', 'noun']])
        키워드 = [키워드]
    elif len(키워드) > 1:
        보여줄df = 마스크된데이터프레임[마스크된데이터프레임['noun'].str.contains('|'.join(키워드))]
        st.dataframe(보여줄df[['name','sentiment','review_sentence']], use_container_width=True)


# with col7_1:
#     if len(키워드) == 1:
#         보여줄df = 마스크된데이터프레임[마스크된데이터프레임['noun'].str.contains(키워드[0])]
#         st.dataframe(보여줄df[['name','sentiment','review_sentence', 'noun', 'replace_slang_sentence']])
#         키워드 = [키워드]
#     elif len(키워드) > 1:
#         보여줄df = 마스크된데이터프레임[마스크된데이터프레임['noun'].str.contains('|'.join(키워드))]
#         st.dataframe(보여줄df[['name','sentiment','review_sentence']], use_container_width=True)

########################################################################################################################
import ast

fix_stop_words = [ '합니다', '하는', '할', '하고', '한다','하다','되다','같다','자다','되다','있다','써다','않다','해보다','주다','되어다', 
             '그리고', '입니다', '그', '등', '이런', '및','제', '더','언늘','결국','생각','식물키',
             '감사','진짜','완전','요ㅎ','사용','정도','엄마','아이','원래','식물','흐흐','하하','정말']

def to_list(text):
    return ast.literal_eval(text)

def lda_modeling(tokens, num_topics, passes=10):
    # word-document matrix
    dictionary = gensim.corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    # Train the LDA model
    model = gensim.models.ldamodel.LdaModel(corpus,
                                            num_topics=num_topics,
                                            id2word=dictionary, # 단어매트릭스
                                            passes=passes, # 학습반복횟수
                                            random_state=100) 
    return model, corpus, dictionary

def print_topic_model(topics, rating, key):
    topic_values = []
    for topic in topics:
        topic_value = topic[1]
        topic_values.append(topic_value)
    topic_model = pd.DataFrame({"topic_num": list(range(1, len(topics) + 1)), "word_prop": topic_values})
    
    # 토글 생성
    if st.checkbox('주제별 구성 단어 비율 확인', key=key):
    # 토글이 선택되었을 때 데이터프레임 출력
        st.dataframe(topic_model, use_container_width=True)


# 시각화1. 각 주제에서 상위 N개 키워드의 워드 클라우드
def topic_wordcloud(model,num_topics):
    cand_mask = np.array(Image.open('/app/streamlit/data/circle.png'))
    cloud = WordCloud(background_color='white',
                      font_path = "/app/streamlit/font/NanumBarunGothic.ttf",
                      width=500,
                      height=500,
                      max_words=7,
                      colormap='tab10',
                      prefer_horizontal=1.0,
                      mask=cand_mask)
    
    topics = model.show_topics(formatted=False)

    # 모델마다 토픽개수가 달라서 rows, cols이 토픽의 개수마다 바뀜주기
    fig, axes = plt.subplots(1, num_topics, figsize=(12,8), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# 명사기준 토픽분석(7개씩 나오게 한건 이전 연구자료들 참고)
def n_get_topic_model(data, topic_number, passes=10, num_words=7, key=None):
    df = pd.read_csv(data)

    # 불용어 리스트
    stopwords = stop_words

    # 문장 리스트 생성
    reviews = []
    for i in range(len(df)):
        reviews.append(to_list(df['noun'][i]))

    # 텍스트 데이터 전처리
    # 불용어 제거, 단어 인코딩 및 빈도수 계산
    tokens = []
    for review in reviews:
        token_review = [w for w in review if len(w) > 1 and w not in stopwords]
        tokens.append(token_review)

    # # LDA 모델링
    model, corpus, dictionary = lda_modeling(tokens, num_topics = topic_number, passes=passes)

    rating = 'pos' 
    topics = model.print_topics(num_words=num_words)
    print_topic_model(topics, rating, key)

    # 토픽별 워드클라우드 시각화
    topic_wordcloud(model, num_topics=topic_number)

# 명사+동사+형용사 기준 토픽분석
def nv_get_topic_model(data, topic_number, passes=10, num_words=7, key=None):
    df = pd.read_csv(data)

    # 불용어 리스트
    stopwords = stop_words

    # 문장 리스트 생성
    reviews = []
    for i in range(len(df)):
        reviews.append(to_list(df['n_v_ad'][i]))

    # 텍스트 데이터 전처리
    # 불용어 제거, 단어 인코딩 및 빈도수 계산
    tokens = []
    for review in reviews:
        token_review = [w for w in review if len(w) > 1 and w not in stopwords]
        tokens.append(token_review)

    # # LDA 모델링
    model, corpus, dictionary = lda_modeling(tokens, num_topics=topic_number, passes=passes)

    rating = 'pos' 
    topics = model.print_topics(num_words=num_words)
    print_topic_model(topics, rating, key)

    # 토픽별 워드클라우드 시각화
    topic_wordcloud(model, num_topics=topic_number)

st.write("")
st.write("")
st.write("")
########################여기서부터 streamlit 구현 #########################
st.subheader('**🔎SWOT 분석**')
tab1, tab2, tab3, tab4 = st.tabs(["**Strength(강점)**", "**Weakness(약점)**", "**Opportunity(기회)**", "**Threat(위협)**"])

with tab1:
    col1_, col2_ = st.beta_columns(2)    

    with col1_:
        n_v_type = st.selectbox('데이터 유형',['명사', '명사+동사+형용사'], key='selectbox1')
    with col2_:
        input_str = st.text_input('제거할 키워드 :', key='stopwords_input1')
        stop_words = fix_stop_words.copy()
        stopwords = stop_words.extend([x.strip() for x in input_str.split(',')])

    st.write('자사 긍정리뷰들의 주제별 키워드를 분석한 결과입니다. :sunglasses:')

    file_path = '/app/streamlit/data/자사긍정(9차).csv'

    if n_v_type =='명사':
        n_get_topic_model(file_path,9 , key='준탱이1')
    else:
        nv_get_topic_model(file_path,10, key='준탱이2')

with tab2:
    col1_2_, col2_2_ = st.beta_columns(2)    

    with col1_2_:
        n_v_type = st.selectbox('데이터 유형',['명사', '명사+동사+형용사'], key='selectbox2')
    with col2_2_:
        input_str = st.text_input('제거할 키워드 :', key='stopwords_input2')
        stop_words = fix_stop_words.copy()
        stopwords = stop_words.extend([x.strip() for x in input_str.split(',')])

    st.write('자사 부정리뷰들의 주제별 키워드를 분석한 결과입니다. :sweat:')

    file_path = '/app/streamlit/data/자사부정(9차).csv'

    if n_v_type =='명사':
        n_get_topic_model(file_path,4, key='준탱이3')
    else:
        nv_get_topic_model(file_path,8, key='준탱이4')

# 추가수정 필요############################################################################
with tab3:
    col1_3_, col2_3_, col3_3_ = st.beta_columns(3)    

    with col1_3_:
        n_v_type = st.selectbox('데이터 유형',['명사', '명사+동사+형용사'], key='selectbox3')
    with col2_3_:
        d_type = st.selectbox('제품',['경쟁사-식물영양제', '경쟁사-뿌리영양제', '경쟁사-살충제',
                                     '경쟁사-식물등', '경쟁사 전체'], key='selectbox3_1_')
    with col3_3_:
        input_str = st.text_input('제거할 키워드 :', key='stopwords_input3')
        stop_words = fix_stop_words.copy()
        stopwords = stop_words.extend([x.strip() for x in input_str.split(',')])
    
    st.write('경쟁사 부정리뷰들의 주제별 키워드를 분석한 결과입니다. :wink:')

    if n_v_type =='명사' and d_type == '경쟁사 전체' :
        file_path = '/app/streamlit/data/경쟁사부정(10차).csv'
        n_get_topic_model(file_path,10, key='준탱이5')
    elif n_v_type =='명사+동사+형용사' and d_type == '경쟁사 전체' :
        file_path = '/app/streamlit/data/경쟁사부정(10차).csv'
        nv_get_topic_model(file_path,8, key='준탱이6')

    elif n_v_type =='명사' and d_type == '경쟁사-식물영양제' :
        file_path = '/app/streamlit/data/경쟁사(식물영양제)부정(10차).csv'
        n_get_topic_model(file_path,5, key='준탱이7')
    elif n_v_type =='명사+동사+형용사' and d_type == '경쟁사-식물영양제' :
        file_path = '/app/streamlit/data/경쟁사(식물영양제)부정(10차).csv'
        nv_get_topic_model(file_path,6, key='준탱이8')
    
    elif n_v_type =='명사' and d_type == '경쟁사-뿌리영양제' :
        file_path = '/app/streamlit/data/경쟁사(뿌리영양제)부정(10차).csv'
        n_get_topic_model(file_path,5, key='준탱이9')
    elif n_v_type =='명사+동사+형용사' and d_type == '경쟁사-뿌리영양제' :
        file_path = '/app/streamlit/data/경쟁사(뿌리영양제)부정(10차).csv'
        nv_get_topic_model(file_path,8, key='준탱이10')
    
    elif n_v_type =='명사' and d_type == '경쟁사-살충제' :
        file_path = '/app/streamlit/data/경쟁사(살충제)부정(10차).csv'
        n_get_topic_model(file_path,4, key='준탱이11')
    elif n_v_type =='명사+동사+형용사' and d_type == '경쟁사-살충제' :
        file_path = '/app/streamlit/data/경쟁사(살충제)부정(10차).csv'
        nv_get_topic_model(file_path,10, key='준탱이12')
    
    elif n_v_type =='명사' and d_type == '경쟁사-식물등' :
        file_path = '/app/streamlit/data/경쟁사(식물등)부정(10차).csv'
        n_get_topic_model(file_path,9, key='준탱이13')
    elif n_v_type =='명사+동사+형용사' and d_type == '경쟁사-식물등' :
        file_path = '/app/streamlit/data/경쟁사(식물등)부정(10차).csv'
        nv_get_topic_model(file_path,10, key='준탱이14')
    

with tab4:
    col1_4_, col2_4_, col3_4_ = st.beta_columns(3)     

    with col1_4_:
        n_v_type = st.selectbox('데이터 유형',['명사', '명사+동사+형용사'], key='selectbox4')
    
    with col2_4_:
        d_type = st.selectbox('제품',['경쟁사-식물영양제', '경쟁사-뿌리영양제', '경쟁사-살충제',
                                     '경쟁사-식물등', '경쟁사 전체'], key='selectbox4_1_')
    with col3_4_:
        input_str = st.text_input('제거할 키워드 :', key='stopwords_input4')
        stop_words = fix_stop_words.copy()
        stopwords = stop_words.extend([x.strip() for x in input_str.split(',')])

    st.write('경쟁사 긍정리뷰들의 주제별 키워드를 분석한 결과입니다. :confounded:')

    if n_v_type =='명사' and d_type == '경쟁사 전체' :
        file_path = '/app/streamlit/data/경쟁사긍정(10차).csv'
        n_get_topic_model(file_path,10, key='준탱이15')
    elif n_v_type =='명사+동사+형용사' and d_type == '경쟁사 전체' :
        file_path = '/app/streamlit/data/경쟁사긍정(10차).csv'
        nv_get_topic_model(file_path,10, key='준탱이16')

    elif n_v_type =='명사' and d_type == '경쟁사-식물영양제' :
        file_path = '/app/streamlit/data/경쟁사(식물영양제)긍정(10차).csv'
        n_get_topic_model(file_path,10, key='준탱이17')
    elif n_v_type =='명사+동사+형용사' and d_type == '경쟁사-식물영양제' :
        file_path = '/app/streamlit/data/경쟁사(식물영양제)긍정(10차).csv'
        nv_get_topic_model(file_path,8, key='준탱이18')
    
    elif n_v_type =='명사' and d_type == '경쟁사-뿌리영양제' :
        file_path = '/app/streamlit/data/경쟁사(뿌리영양제)긍정(10차).csv'
        n_get_topic_model(file_path,10, key='준탱이19')
    elif n_v_type =='명사+동사+형용사' and d_type == '경쟁사-뿌리영양제' :
        file_path = '/app/streamlit/data/경쟁사(뿌리영양제)긍정(10차).csv'
        nv_get_topic_model(file_path,10, key='준탱이20')
    
    elif n_v_type =='명사' and d_type == '경쟁사-살충제' :
        file_path = '/app/streamlit/data/경쟁사(살충제)긍정(10차).csv'
        n_get_topic_model(file_path,10, key='준탱이21')
    elif n_v_type =='명사+동사+형용사' and d_type == '경쟁사-살충제' :
        file_path = '/app/streamlit/data/경쟁사(살충제)긍정(10차).csv'
        nv_get_topic_model(file_path,9, key='준탱이22')
    
    elif n_v_type =='명사' and d_type == '경쟁사-식물등' :
        file_path = '/app/streamlit/data/경쟁사(식물등)긍정(10차).csv'
        n_get_topic_model(file_path,10, key='준탱이23')
    elif n_v_type =='명사+동사+형용사' and d_type == '경쟁사-식물등' :
        file_path = '/app/streamlit/data/경쟁사(식물등)긍정(10차).csv'
        nv_get_topic_model(file_path,9, key='준탱이24')


########################################################################################################################
st.write("")
st.write("")
st.write("")
########################Tableau 구현 #########################
st.subheader('**🔎자사/경쟁사 리뷰 분류 분석**')
tab1, tab2= st.tabs(["**자사**", "**경쟁사**"])

with tab1:
    st.write('제품/배송/사용법으로 자사의 리뷰를 분류한 결과입니다. 😊')
    with st.container():
        url = "https://public.tableau.com/views/1_16835240938720/1_1?:language=ko-KR&:showVizHome=no&:embed=true"
        html = f'''
            <iframe src={url} style="width:100%; height:70vh"></iframe>
        '''
        st.markdown(html, unsafe_allow_html=True)

with tab2:
    st.write('제품/배송/사용법으로 경쟁사의 리뷰를 분류한 결과입니다. 😊')
    with st.container():
        url = "https://public.tableau.com/views/_16835258920980/1_1?:language=ko-KR&:showVizHome=no&:embed=true"
        html = f'''
            <iframe src={url} style="width:100%; height:70vh"></iframe>
        '''
        st.markdown(html, unsafe_allow_html=True)
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################