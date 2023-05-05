import streamlit as st
import pandas as pd
import numpy as np
import ast
import gensim
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from PIL import Image

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
        st.write(topic_model)


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
    col1, col2 = st.beta_columns(2)    

    with col1:
        n_v_type = st.selectbox('데이터 타입',['명사', '명사+동사+형용사'], key='selectbox1')
    with col2:
        input_str = st.text_input('불용어를 추가하실 수 있습니다.', key='stopwords_input1')
        stop_words = fix_stop_words.copy()
        stopwords = stop_words.extend([x.strip() for x in input_str.split(',')])

    st.header("Strength(강점)")
    st.write('자사의 긍정리뷰들을 토픽모델링한 결과입니다. :sunglasses:')

    file_path = '/app/streamlit/data/자사긍정(6차).csv'

    if n_v_type =='명사':
        n_get_topic_model(file_path,8 , 11)
    else:
        nv_get_topic_model(file_path,10, 12)

with tab2:
    col1_2, col2_2 = st.beta_columns(2)    

    with col1_2:
        n_v_type = st.selectbox('데이터 타입',['명사', '명사+동사+형용사'], key='selectbox2')
    with col2_2:
        input_str = st.text_input('불용어를 추가하실 수 있습니다.', key='stopwords_input2')
        stop_words = fix_stop_words.copy()
        stopwords = stop_words.extend([x.strip() for x in input_str.split(',')])

    st.header("Weakness(약점)")
    st.write('자사의 부정리뷰들을 토픽모델링한 결과입니다. :sweat:')

    file_path = '/app/streamlit/data/자사부정(6차).csv'

    if n_v_type =='명사':
        n_get_topic_model(file_path,4, 22)
    else:
        nv_get_topic_model(file_path,5, 22)

with tab3:
    col1_3, col2_3 = st.beta_columns(2)    

    with col1_3:
        n_v_type = st.selectbox('데이터 타입',['명사', '명사+동사+형용사'], key='selectbox3')
    with col2_3:
        input_str = st.text_input('불용어를 추가하실 수 있습니다.', key='stopwords_input3')
        stop_words = fix_stop_words.copy()
        stopwords = stop_words.extend([x.strip() for x in input_str.split(',')])
    
    st.header("Opportunity(기회)")
    st.write('경쟁사의 부정리뷰들을 토픽모델링한 결과입니다. :wink:')

    file_path = '/app/streamlit/data/경쟁사부정(6차).csv'

    if n_v_type =='명사':
        n_get_topic_model(file_path,10)
    else:
        nv_get_topic_model(file_path,8)

with tab4:
    col1_4, col2_4 = st.beta_columns(2)    

    with col1_4:
        n_v_type = st.selectbox('데이터 타입',['명사', '명사+동사+형용사'], key='selectbox4')
    with col2_4:
        input_str = st.text_input('불용어를 추가하실 수 있습니다.', key='stopwords_input4')
        stop_words = fix_stop_words.copy()
        stopwords = stop_words.extend([x.strip() for x in input_str.split(',')])

    st.header("Treatment(위협)")
    st.write('경쟁사의 긍정리뷰들을 토픽모델링한 결과입니다. :confounded:')

    file_path = '/app/streamlit/data/경쟁사긍정(6차).csv'

    if n_v_type =='명사':
        n_get_topic_model(file_path,9)
    else:
        nv_get_topic_model(file_path,9)