import streamlit as st
import streamlit.components.v1 as components

import plotly.express as px
import plotly.graph_objects as go

# 기본 라이브러리
import os
import ast
from datetime import datetime
from datetime import timedelta

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

stopwords = ['언늘', '결국', '생각', '후기', '감사', '진짜', '완전', '사용', '요즘', '정도', '이번', '달리뷰', '결과', 
             '지금', '동영상', '조금', '안테', '입제', '영상', '이번건', '며칠', '이제', '거시기', '얼듯', '처음', '다음']
########################################################################################################################
# 레이아웃
with st.container():
    col0_1, col0_2, col0_3, col0_4, col0_4 = st.columns([1,1,1,1,1])
with st.container():
    col1_1, col1_2, col1_3, col1_4 = st.columns([1,1,1,1])
with st.container():
    col2_1, col2_2, col2_3, col2_4 = st.columns([1,1,1,1])
with st.container():
    col3_1, col3_2 = st.columns([1,1])
with st.container():
    col4_1, col4_2, col4_3 = st.columns([1,1,2])
########################################################################################################################
# 사용자 입력
with col0_3:
    api키 = os.environ['$\{\{secrets.VARIABLE_NAME}}']
    api키
    api키 = os.getenv('secrets.API_KEY')
    api키
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
        ('명사', '명사+동사+형용사'))
    st.write('이것: ', 품사옵션)

with col1_3:
    회사종류 = st.selectbox(
        '고르세요',
        ('자사+경쟁사', '꽃피우는 시간', '경쟁사-식물영양제', 
         '경쟁사-뿌리영양제', 
         '경쟁사-살충제',
         '경쟁사-식물등',
         '경쟁사All',
         ))
    st.write('이것: ', 회사종류)
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

with col2_3:
    start_date = st.date_input(
        '시작날짜',
        value=시작날짜,
        min_value=시작날짜,
        max_value=마지막날짜
    )
with col2_4:
    end_date = st.date_input(
        '마지막날짜',
        value=마지막날짜,
        min_value=시작날짜,
        max_value=마지막날짜
    )

기간마스크 = ((df_리뷰_감성분석결과['time'] >= pd.to_datetime(start_date)) & (df_리뷰_감성분석결과['time'] <= pd.to_datetime(end_date)))

with col2_1:
    추가불용어 = st.text_input('불용어를 추가하세요', '')
    if 추가불용어 == '':
        st.write('예시 : 영양제, 식물, 배송')
    if 추가불용어 != '':
        st.write('추가된 불용어: ', 추가불용어)

with col2_2:
    단어수 = st.slider(
        '단어 수를 조정하세요',
        10, 300, step=1)
    st.write('단어수: ', 단어수)

if 추가불용어.find(',') != -1:
    stopwords.extend([i.strip() for i in 추가불용어.split(',')])
if 추가불용어.find(',') == -1:
    stopwords.append(추가불용어) 

with col1_4:
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
if 품사옵션 == '명사+동사+형용사':
    품사 = 'n_v_ad'

카운트 = get_count_top_words(df_리뷰_감성분석결과[기간마스크 & 회사종류마스크], num_words=단어수, 품사=품사)
tdidf = get_tfidf_top_words(df_리뷰_감성분석결과[기간마스크 & 회사종류마스크], num_words=단어수, 품사=품사)

if option == '카운트':
    words = 카운트
if option == 'td-idf':
    words = tdidf
########################################################################################################################
# 사용자 입력후 사용할 데이터 정리

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
with col3_1:
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
# 네트워크 차트

reviews = [eval(i) for i in df_리뷰_감성분석결과[기간마스크 & 회사종류마스크][품사]]

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


with col3_2:
    try:
        net = 네트워크[0]
        net.save_graph(f'/app/streamlit/pyvis_graph.html')
        HtmlFile = open(f'/app/streamlit/pyvis_graph.html', 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=435)
    except:
        st.write('존재하지 않는 키워드예요.')
########################################################################################################################
with col4_3:
    if len(키워드) == 1:
        보여줄df = df_리뷰_감성분석결과[df_리뷰_감성분석결과['noun'].str.contains(키워드[0])]
        st.dataframe(보여줄df[['name','sentiment','review_sentence', 'noun', 'replace_slang_sentence']])
        키워드 = [키워드]
    elif len(키워드) > 1:
        보여줄df = df_리뷰_감성분석결과[df_리뷰_감성분석결과['noun'].str.contains('|'.join(키워드))]
        st.dataframe(보여줄df[['name','sentiment','review_sentence']], use_container_width=True)
########################################################################################################################
import ast

fix_stop_words = [ '합니다', '하는', '할', '하고', '한다','하다','되다','같다','자다','되다','있다','써다','않다','해보다','주다','되어다', 
             '그리고', '입니다', '그', '등', '이런', '및','제', '더','언늘','결국','생각','식물키',
             '감사','ㅋㅋ','진짜','완전','요ㅎ','사용','정도','엄마','아이','원래','식물']

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
    if st.checkbox('토픽별 구성 단어 비율 확인하기', key=key):
    # 토글이 선택되었을 때 데이터프레임 출력
        st.dataframe(topic_model, use_container_width=True)


# 시각화1. 각 주제에서 상위 N개 키워드의 워드 클라우드
def topic_wordcloud(model,num_topics):
    cand_mask = np.array(Image.open('/app/streamlit/data/circle.png'))
    cloud = WordCloud(background_color='white',
                      font_path = "/app/streamlit/font/NanumBarunGothic.ttf",
                      width=500,
                      height=500,
                      max_words=15,
                      colormap='tab10',
                      prefer_horizontal=1.0,
                      mask=cand_mask)
    
    topics = model.show_topics(formatted=False)

    # 모델마다 토픽개수가 달라서 rows, cols이 토픽의 개수마다 바뀜
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

# 명사기준 토픽분석(6개씩 나오게 한건 이전 연구자료들 참고)
def n_get_topic_model(data, topic_number, passes=10, num_words=6, key=None):
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
def nv_get_topic_model(data, topic_number, passes=10, num_words=6, key=None):
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


########################여기서부터 streamlit 구현 #########################

st.title('리뷰_토픽모델링')



tab1, tab2, tab3, tab4 = st.tabs(["**S**", "**W**", "**O**", "**T**"])

with tab1:
    col1_, col2_ = st.beta_columns(2)    

    with col1_:
        n_v_type = st.selectbox('데이터 타입',['명사', '명사+동사+형용사'], key='selectbox1')
    with col2_:
        input_str = st.text_input('불용어를 추가하실 수 있습니다.', key='stopwords_input1')
        stop_words = fix_stop_words.copy()
        stopwords = stop_words.extend([x.strip() for x in input_str.split(',')])

    st.header("Strength(강점)")
    st.write('자사의 긍정리뷰들을 토픽모델링한 결과입니다. :sunglasses:')

    file_path = '/app/streamlit/data/자사긍정(6차).csv'

    if n_v_type =='명사':
        n_get_topic_model(file_path,8 , key='준탱이1')
    else:
        nv_get_topic_model(file_path,10, key='준탱이2')

with tab2:
    col1_2_, col2_2_ = st.beta_columns(2)    

    with col1_2_:
        n_v_type = st.selectbox('데이터 타입',['명사', '명사+동사+형용사'], key='selectbox2')
    with col2_2_:
        input_str = st.text_input('불용어를 추가하실 수 있습니다.', key='stopwords_input2')
        stop_words = fix_stop_words.copy()
        stopwords = stop_words.extend([x.strip() for x in input_str.split(',')])

    st.header("Weakness(약점)")
    st.write('자사의 부정리뷰들을 토픽모델링한 결과입니다. :sweat:')

    file_path = '/app/streamlit/data/자사부정(6차).csv'

    if n_v_type =='명사':
        n_get_topic_model(file_path,4, key='준탱이3')
    else:
        nv_get_topic_model(file_path,5, key='준탱이4')

with tab3:
    col1_3_, col2_3_ = st.beta_columns(2)    

    with col1_3_:
        n_v_type = st.selectbox('데이터 타입',['명사', '명사+동사+형용사'], key='selectbox3')
    with col2_3_:
        input_str = st.text_input('불용어를 추가하실 수 있습니다.', key='stopwords_input3')
        stop_words = fix_stop_words.copy()
        stopwords = stop_words.extend([x.strip() for x in input_str.split(',')])
    
    st.header("Opportunity(기회)")
    st.write('경쟁사의 부정리뷰들을 토픽모델링한 결과입니다. :wink:')

    file_path = '/app/streamlit/data/경쟁사부정(6차).csv'

    if n_v_type =='명사':
        n_get_topic_model(file_path,10, key='준탱이5')
    else:
        nv_get_topic_model(file_path,8, key='준탱이6')

with tab4:
    col1_4_, col2_4_ = st.beta_columns(2)    

    with col1_4_:
        n_v_type = st.selectbox('데이터 타입',['명사', '명사+동사+형용사'], key='selectbox4')
    with col2_4_:
        input_str = st.text_input('불용어를 추가하실 수 있습니다.', key='stopwords_input4')
        stop_words = fix_stop_words.copy()
        stopwords = stop_words.extend([x.strip() for x in input_str.split(',')])

    st.header("Treatment(위협)")
    st.write('경쟁사의 긍정리뷰들을 토픽모델링한 결과입니다. :confounded:')

    file_path = '/app/streamlit/data/경쟁사긍정(6차).csv'

    if n_v_type =='명사':
        n_get_topic_model(file_path,9, key='준탱이7')
    else:
        nv_get_topic_model(file_path,9, key='준탱이8')
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################