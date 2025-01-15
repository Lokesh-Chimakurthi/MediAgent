import asyncio
import nest_asyncio
import streamlit as st
import logfire
from pydantic_ai.usage import Usage, UsageLimits
from src.agent import search_agent

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure page and logfire
st.set_page_config(page_title="PubMed Research Assistant", layout="wide")
logfire.configure(send_to_logfire="if-token-present")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "message_history" not in st.session_state:
    st.session_state.message_history = None
if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)

# Header and clear chat button in a row
col1, col2 = st.columns([4, 1])
with col1:
    st.title("PubMed Research Assistant")
with col2:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.message_history = None
        st.rerun()


def format_citations(citations):
    """Format citations as clickable markdown links"""
    formatted = []
    for citation in citations:
        # Extract number and rest of citation
        num, rest = citation.split("] ", 1)
        title, url = rest.split(" - ", 1)
        formatted.append(f"{num}] [{title}]({url})")
    return "\n".join(formatted)


async def get_response(prompt, message_history):
    """Async function to get response from agent"""
    usage = Usage()
    usage_limits = UsageLimits(request_limit=10)

    return await search_agent.run(
        prompt, usage=usage, usage_limits=usage_limits, message_history=message_history
    )


# Chat input and conversation management
if prompt := st.chat_input("Enter your medical research question"):
    logfire.info("user_query", query=prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Searching PubMed..."):
            try:
                result = st.session_state.loop.run_until_complete(
                    get_response(prompt, st.session_state.message_history)
                )

                # Create assistant's response content
                response_content = {
                    "answer": result.data.answer,
                    "citations": format_citations(result.data.citations)
                }

                # Add assistant's response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_content
                })

                # Update message history for the agent
                st.session_state.message_history = result.all_messages()

            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {"answer": error_message, "citations": ""}
                })
                logfire.error("agent_error", error=str(e))

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            st.write(message["content"]["answer"])
            if message["content"]["citations"]:
                with st.expander("Citations"):
                    st.markdown(message["content"]["citations"])
