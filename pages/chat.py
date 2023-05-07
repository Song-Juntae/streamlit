import openai
import streamlit as st
from streamlit_chat import message

st.header("ChatGPT-3 (Demo)")
st.markdown("해당 결과를 보고 궁금한 점은 ChatGPT에게 물어보세요.")

openai.api_key = st.secrets["MIN_API_KEY"]
openai.organization= st.secrets["MIN_ORG_ID"]

# messages = []
# while True:
#     user_content = input("user : ")
#     messages.append({"role": "user", "content": f"{user_content}"})

#     completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

#     assistant_content = completion.choices[0].message["content"].strip()

#     messages.append({"role": "assistant", "content": f"{assistant_content}"})

#     print(f"GPT : {assistant_content}")

    #코드출처: https://blog.naver.com/kimflstudio/223045193407


#####이전코드#####
# API = st.secrets["JUN_API_KEY"]
# API
# # chatGPT
# openai.api_key = str(API)

# completion = openai.Completion.create(model="ada", prompt="Hello world")
# completion

# completions
 
# def generate_response(prompt):
#     completions = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=1024,
#         stop=None,
#         temperature=0,
#         top_p=1
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