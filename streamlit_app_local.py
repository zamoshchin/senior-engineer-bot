"""Python file to serve as the frontend"""
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
###
import streamlit as st
from streamlit_chat import message
import os
import time
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
import openai
import re

# define Agent class
class CAMELAgent:

    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model(messages)
        self.update_messages(output_message)

        return output_message

def start_state():
    st.session_state["started"] = True

def get_text():
    input_text = st.text_input("e.g. 'uber for dogs'", "", key="input", disabled=st.session_state.started, on_change=start_state())
    return input_text

def split_text(text, response_num, prefix, is_user):
    sentences = text.split(". ")
    i = 0
    for sentence in sentences:
        unique_key = prefix+str(response_num)+str(i)
        sentence = sentence.strip()
        if(sentence == "" or sentence == "<TASK_DONE>"):
            continue
        message(sentence, key=unique_key, is_user=is_user)
        i+=1

def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str, task2: str, tech_spec: str, assistant_inception_prompt: str, user_inception_prompt: str):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task2, tech_spec=tech_spec)[0]
    
    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = user_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task, tech_spec=tech_spec)[0]

    return assistant_sys_msg, user_sys_msg

def extract_sentence(text, start_phrase):
    # Define a regular expression pattern to match the sentence starting with the specified phrase
    # The pattern uses a non-greedy match (.*?) to capture everything until the first sentence-ending punctuation (., !, ?)
    pattern = re.compile(r'(' + re.escape(start_phrase) + r'.*?[.!?])', re.IGNORECASE)
    
    # Search for the pattern in the text
    match = pattern.search(text)
    
    # If a match is found, return the matched sentence; otherwise, return None
    return match.group(1) if match else None

# TODO: fill this in
os.environ["OPENAI_API_KEY"] = ""

# Define UI
st.set_page_config(page_title="Senior Engineer Bot", page_icon=":robot:")
st.header("Engineering discussion around:")

# Replace with the ID of the Google Doc you want to access
document_id = "1OI-1TbPjnJ-aLCnSYd7BDL4-06Axp8u9p9ikhXrD_1g"

# Replace with the path to your service account credentials file
credentials_file = "credentials.json"

# Authenticate and build the Drive API client
creds = service_account.Credentials.from_service_account_file(credentials_file, scopes=['https://www.googleapis.com/auth/drive'])
drive_service = build('drive', 'v3', credentials=creds)


def engineering_discussion(comment_id, section):
    st.markdown(section)

    #initialize session states
    if "vc_messages" not in st.session_state:
        st.session_state["vc_messages"] = []

    if "founder_messages" not in st.session_state:
        st.session_state["founder_messages"] = []

    if "started" not in st.session_state:
        st.session_state["started"] = False

    if "done" not in st.session_state:
        st.session_state["done"] = False

    # define roles
    startup_description = "Couchbase is an award-winning distributed NoSQL cloud database that delivers unmatched versatility, performance, scalability, and financial value for all of your cloud, mobile, on-premises, hybrid, distributed cloud, and edge computing applications."#user_input
    assistant_role_name = f"Senior Engineer at {startup_description} answering questions and addressing concerns about their technical design document"
    user_role_name = f"Senior Engineer at {startup_description} reviewing technical design documents, bringing up issues, and asking questions"

    # define task
    user_task = "Ask questions that will help you bring up one concern with the following section of the technical design document:f{section}. YOU MUST PRESENT ONE CONCERN AND GIVE REASONING FOR IT."
    assistant_task_info = "Answer questions, and come up with suggestions for improving the following section of the technical design document:f{section}"
    tech_spec = '''
    Summary
    It would be desirable for the SDK to be able to know the Couchbase Cluster server version it is connected with. This is would be useful for determining if a particular feature is available based on the cluster compatibility version of the cluster. It would also be useful during SDK testing to identify if a feature is available and optionally skip tests that would always fail, eg RBAC.

    There could be additional requirements for a consuming application and/or projects know the server version to modify behaviour based on the cluster version, eg start using a new feature once the cluster has been upgraded.
    Description
    Cluster compatibility can can be determined by executing a HTTP GET to the URI http://<server-ip>:8091/pools/default of a cluster node and parsing the resulting JSON. On Pre-5.0 clusters this is not an authenticated endpoint, however on 5.0 and onwards it requires authentication, however there is no specific role required as long as a valid username and password is used.

    The result of calling this URI is a Pools JSON structure (see below for example) which includes a ‘nodes’ array property and each node object contains a “version” property. The format of the property follows the Version Format section below that can be parsed into the ClusterVersion.

    <alternative to get each cluster version>
    There is a http://<server-ip>:8091/versions endpoint that returns that particular node version in the same format as ClusterFormat. However, this endpoint does not provide the cluster compatibility version.

    Both the Bucket and Cluster objects are to be extended to include a GetClusterVersion method that returns a ClusterVersion. This internally makes a call to the URI defined above to determine the cluster version and create the ClusterVersion. The ClusterVersion should be cached to prevent subsequent GetClusterVersion calls executing unnecessary HTTP requests. Cached ClusterVersion’s are invalidated when a new bucket config is processed and once invalidated, a subsequent call to GetClusterVersion would then perform another HTTP request to create a new ClusterVersion and will be cached.

    Once you have a ClusterVersion, it can be used to determine if a ClusterFeature is supported. This does not indicate if a particular node has the feature enabled, eg for Multi-dimensional-scaling services such as KV, Query or FTS, but indicates if the cluster as a whole supports the functionality. The ClusterFeature enum can be used with a ClusterVersion to determine if a feature is supported.
    IBucket & ICluster
    Both Bucket and Cluster objects, and interfaces where appropriate, should have the following methods added:

    ServerVersion GetClusterCompatibilityVersion()
    ServerVersion[] GetClusterServerVersions()

    FeatureSupported
    This method indicates if a given ClusterFeature is supported by the cluster, the method signature would look like this:

    bool FeatureSupported(ClusterFeature feature)
    Comparable
    ClusterVersion should offer language comparable options where available. This is useful when comparing version to determine if a feature is available. For example, you should be able to determine if a ClusterVersion A is less than, equal to or greater than ClusterVersion B.

    ToString()
    When the ClusterVersion cast to a string, eg using ToString(), it will use the same format that server uses. See Version Format.

    ClusterFeature (enum)
    Cluster features describe from what version a particular feature was introduced into Couchbase Server. This Version can then be used to compare with to determine if a cluster feature is supported.
    '''

    #assistant_task_info = f"Defend your tech spec from an engineer that is raising concerns, offer solutions to his concerns and possible modifications of the spec: {tech_spec}"
    #task = specified_task = f"Evaluate the following tech spec by asking questions and bringing up concerns. At the end of your questioning bring up your 5 most immidiate concerns with the tech spec. Here is the tech spec: {tech_spec}"

    # define inception prompts
    assistant_inception_prompt = (
    """Never forget you are a {assistant_role_name} and I am a {user_role_name}. Never flip roles! Never instruct me!
    We share a common interest in collaborating to successfully complete a task.
    You must help me to complete the task.
    Here is the task: {task}. Never forget our task!
    I must instruct you based on your expertise and my needs to complete the task.
    Here is the full tech spec: {tech_spec}

    I must give you one instruction at a time.
    You must write a specific solution that appropriately completes the requested instruction.
    Do not add anything else other than your solution to my instruction.
    You are never supposed to ask me any questions you only answer questions.
    You are never supposed to reply with a fake solution. Explain your solutions.
    Your solution must be declarative sentences and simple present tense.
    Unless I say the task is completed, you should always start with:

    <YOUR_SOLUTION>

    <YOUR_SOLUTION> should be specific and provide preferable implementations and examples for task-solving.

    If I give you a final score respond with a single word <TASK_DONE>.
    """
    )

    user_inception_prompt = (
    """Never forget you are a {user_role_name} and I am a {assistant_role_name}. Never flip roles! You will always instruct me.
    I must help you to complete the task.
    Here is the task: {task}. Never forget our task!
    Here is the full tech spec: {tech_spec}
    You must instruct me based on my expertise and your needs to complete the task:

    <YOUR_INSTRUCTION>

    The "Instruction" describes a task or question. 
    The "Instruction" cannot ask to generate a concern.
    Do not thank the Founder for their explanation.
    You must give me one instruction at a time.
    I must write a response that appropriately completes the requested instruction.
    Do not add anything else other than your instruction!
    Keep giving me instructions until you think the task is completed.
    When the task is completed, you must reply with <TASK_DONE>.
    """
    )

    # Used to record full transcript
    if "transcript" not in st.session_state:
        st.session_state["transcript"] = ["Company idea: " + startup_description + "\n"]

    if(st.session_state["done"] == False):
        # buy some time with welcome message
        message("Hello, I'm a Senior Engineer that will be evaluating your Design Doc", key="intro", is_user=False)

        # initialize agents
        assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, user_task, assistant_task_info, tech_spec, assistant_inception_prompt, user_inception_prompt)
        assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
        user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

        # Reset agents
        assistant_agent.reset()
        user_agent.reset()

        # Initialize chats 
        assistant_msg = HumanMessage(
            content=(f"{user_sys_msg.content}. "
                        "Now start to give me instructions one by one. "
                        "Only reply with Instruction"))

        user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
        user_msg = assistant_agent.step(user_msg)

        # Start chat
        chat_turn_limit, n = 4, 0
        vc_count = 0
        founder_count = 0
        try:
            while n < chat_turn_limit:
                n += 1
                user_ai_msg = user_agent.step(assistant_msg)
                user_msg = HumanMessage(content=user_ai_msg.content)
                st.session_state.transcript.append("Engineer 1:\n"+user_ai_msg.content+"\n")
                st.session_state.vc_messages.append(user_ai_msg.content)
                split_text(st.session_state["vc_messages"][vc_count], vc_count, "vc_", False)
                vc_count+=1

                assistant_ai_msg = assistant_agent.step(user_msg)
                assistant_msg = HumanMessage(content=assistant_ai_msg.content)
                st.session_state.transcript.append("Engineer 2:\n"+assistant_ai_msg.content+"\n")
                st.session_state.founder_messages.append(assistant_ai_msg.content)
                split_text(st.session_state["founder_messages"][founder_count], founder_count, "founder_", True)
                founder_count+=1
                    
                if "<TASK_DONE>" in user_msg.content or "Thank you!" in user_msg.content or "<TASK_DONE>" in assistant_msg.content or "Thank you!" in user_msg.content or "Thank you" in assistant_msg.content:
                    break
        except Exception as e:
            pass
    st.markdown(
    """
    ### 🎉Thanks for trying out the demo!🎉

    If you liked this project follow us on twitter [@honeyimholm](https://twitter.com/honeyimholm) and [@alexwhotweets](https://twitter.com/alexwhotweets) for more!    
    """)
    st.session_state.done = True
    transcript = "\n".join(st.session_state.transcript)
    link = '[Click Here To Try Again](http://localhost:8501/)'
    st.markdown(link, unsafe_allow_html=True)
    st.download_button('Download Transcript',data=transcript, file_name="transcript.txt", key="download_transcript")

    # pull concern from transcript
    concern_header = """You are Engineer 1, what is your concern from the following transcript, state the concern in one sentence starting with "My concern is". You MUST have a concern. 
    Transcript:"""
    concern_query = transcript + concern_header
    resp = openai.ChatCompletion.create(
                      model='gpt-4',
                      messages=[
                        {"role": "user", "content": concern_query}
                      ],
                      max_tokens=1024,
                    )

    concern_summary = resp['choices'][0]['message']['content']

    st.markdown(concern_summary)
    reply = drive_service.replies().create(fileId=document_id, commentId=comment_id, fields="*", body={'content': concern_summary}).execute()

    # pull suggestion from transcript
    suggestion_header = """You are Engineer 2, what is your solution to engineer 1's concern from the following transcript, state the suggestion in ONE SENTENCE starting with "One suggestion is". You must have a suggestion. 
    Transcript:
    """
    suggestion_query = transcript + suggestion_header
    resp = openai.ChatCompletion.create(
                      model='gpt-4',
                      messages=[
                        {"role": "user", "content": suggestion_query}
                      ],
                      max_tokens=1024,
                    )
    suggestion_summary = resp['choices'][0]['message']['content']
    suggestion_summary = extract_sentence(suggestion_summary, "One Suggestion")

    reply = drive_service.replies().create(fileId=document_id, commentId=comment_id, fields="*", body={'content': suggestion_summary}).execute()

found_comment = False
while not found_comment:
    # Call the Drive API to get the document's comments
    try:
        comments = drive_service.comments().list(fileId=document_id, fields="*").execute()
        for comment in comments.get('comments', []):
            if not comment.get('resolved', False) and not comment.get('deleted', False) and 'quotedFileContent' in comment:
                comment_id = comment['id']
                comment_context = comment['quotedFileContent']['value']
                comment_content = comment['content']
                replies = []
                last_reply = None
                for r in comment['replies']:
                    replies.append(r['content'])
                    last_reply = r['author']['displayName']
                if last_reply == 'senior-engineer@senior-engineer.iam.gserviceaccount.com':
                    continue

                engineering_discussion(comment_id, comment_context)
                found_comment = True

    except HttpError as error:
        print(f'An error occurred: {error}')

          