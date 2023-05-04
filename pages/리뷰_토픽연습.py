import streamlit as st
import pandas as pd
import numpy as np
import ast
import gensim
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from PIL import Image

fix_stop_words = [ '합니다', '하는', '할', '하고', '한다', 
             '그리고', '입니다', '그', '등', '이런', '및','제', '더','언늘','결국','생각','식물키',
             '감사','ㅋㅋ','진짜','완전','요ㅎ','사용','정도','엄마','아이','원래','식물']
stop_words = fix_stop_words.copy()

def to_list(text):
    return ast.literal_eval(text)

def lda_modeling(tokens, num_topics=4, passes=10):
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

def print_topic_model(topics, rating):
    topic_values = []
    for topic in topics:
        topic_value = topic[1]
        topic_values.append(topic_value)
    topic_model = pd.DataFrame({"topic_num": list(range(1, len(topics) + 1)), "word_prop": topic_values})
    st.dataframe(topic_model)

# pyLDAvis 오류(일단 보류 후 2차 개선)
# def lda_visualize(model, corpus, dictionary, rating):
#     pyLDAvis.enable_notebook()
#     result_visualized = gensim_models.prepare(model, corpus, dictionary, sort_topics=False)
#     pyLDAvis.save_html(result_visualized, 'lda_result.html')
#     pyLDAvis.display(result_visualized)


# 시각화1. 각 주제에서 상위 N개 키워드의 워드 클라우드
def topic_wordcloud(model):
    cand_mask = np.array(Image.open('/app/busypeople-stramlit/data/circle.png'))
    cloud = WordCloud(background_color='white',
                      font_path = "/app/busypeople-stramlit/font/NanumBarunGothic.ttf",
                      width=500,
                      height=500,
                      max_words=15,
                      colormap='tab10',
                      prefer_horizontal=1.0,
                      mask=cand_mask)
        
    topics = model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(6,6), sharex=True, sharey=True)

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


def get_topic_model(data, num_topics=4, passes=10, num_words=15):
    df = pd.read_csv(data)

    # 불용어 리스트
    stopwords = stop_words

    # 문장 리스트 생성
    reviews = []
    for i in range(len(df)):
        reviews.append(to_list(df['kha_nng_은어전처리_sentence'][i]))

    # 텍스트 데이터 전처리
    # 불용어 제거, 단어 인코딩 및 빈도수 계산
    tokens = []
    for review in reviews:
        token_review = [w for w in review if len(w) > 1 and w not in stopwords]
        tokens.append(token_review)

    # # LDA 모델링
    model, corpus, dictionary = lda_modeling(tokens, num_topics=num_topics, passes=passes)

    rating = 'pos' 
    topics = model.print_topics(num_words=num_words)
    print_topic_model(topics, rating)

    # 시각화 결과(pyLDAvis 오류(일단 보류 후 2차 개선)) 
    # lda_visualize(model, corpus, dictionary, rating)

    # 토픽별 워드클라우드 시각화
    topic_wordcloud(model)


############streamlit 구현 ##################
st.title('리뷰_토픽모델링')

col1, col2, col3 = st.beta_columns(3)    
with col1:
    type = st.selectbox('회사',['자사', '경쟁사'])
with col2:
    pos_neg = st.radio('리뷰타입',['긍정', '부정'],horizontal=True)
with col3:
    input_str = st.text_input('제거할 키워드')
    stopwords = stop_words.extend([x.strip() for x in input_str.split(',')])

# 선택한 타입에 따라 함수 호출
if type == '자사' and pos_neg == '긍정':
    file_path = '/app/busypeople-stramlit/data/자사긍정리뷰.csv'
elif type == '자사' and pos_neg == '부정':
    file_path = '/app/busypeople-stramlit/data/자사부정리뷰.csv'
elif type == '경쟁사' and pos_neg == '긍정':
    file_path = '/app/busypeople-stramlit/data/경쟁사긍정리뷰.csv'
else:
    file_path = '/app/busypeople-stramlit/data/경쟁사부정리뷰.csv'

get_topic_model(file_path)


###################################################

st.header("ChatGPT-3 (Demo)")
st.markdown("해당 결과를 보고 궁금한 점은 ChatGPT에게 물어보세요.")

# import openai
# import streamlit as st
# from streamlit_chat import message

# # chatGPT
# openai.api_key = ''
 
# def generate_response(prompt):
#     completions = openai.Completion.create (
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=1024,
#         stop=None,
#         temperature=0,
#         top_p=1,
#     )
 
#     message = completions["choices"][0]["text"].replace("\n", "")
#     return message
 
# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []
 
# if 'past' not in st.session_state:
#     st.session_state['past'] = []
 
# with st.form('form', clear_on_submit=True):
#     user_input = st.text_input('You: ', '', key='input')
#     submitted = st.form_submit_button('Send')
 
# if submitted and user_input:
#     output = generate_response(user_input)
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(output)
 
# if st.session_state['generated']:
#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
#         message(st.session_state["generated"][i], key=str(i))