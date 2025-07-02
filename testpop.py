import streamlit as st

st.set_page_config(layout="wide") # Optional: Use full width

st.title("Streamlit Chat with Expandable Details")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to simulate getting a response (replace with your actual logic)
def get_chat_response(user_query):
    user_query_lower = user_query.lower()
    if "product dimensions" in user_query_lower:
        main_answer = "PRODUCT TYPES: WIRELESS, WIRELINE"
        details = """
        These are the primary product types currently supported.
        Wireless products include mobile communication services,
        while Wireline products refer to fixed-line services.
        """
    elif "service types" in user_query_lower:
        main_answer = "THE AVAILABLE SERVICE TYPES ARE: APRIL, FEBRUARY, FULL YEAR, JANUARY, JUNE, MARCH, MAY, Q1, Q2, DATA, POSTPAID, PREPAID, VIDEO, VOICE."
        details = """
        This list encompasses various service categories.
        Time-based services (April, February, etc.) relate to reporting periods.
        Data, Postpaid, Prepaid, Video, and Voice are core telecom services.
        Q1, Q2 refer to quarterly reports.
        """
    elif "services tcs company provides" in user_query_lower:
        main_answer = "I can only provide information related to telecom business metrics and KPIs. Please rephrase your question to be about these topics."
        details = "This agent is specifically scoped to telecom business metrics and KPIs and cannot answer general company service questions."
    else:
        main_answer = f"I received your query: '{user_query}'. I'm still learning. Can you rephrase or ask about telecom metrics?"
        details = "No specific details available for this general query."

    return main_answer, details

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])
        if "details" in message and message["details"]:
            with st.expander("Show Details"):
                st.write(message["details"])

# React to user input
if prompt := st.chat_input("What can I help you with?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response
    main_response, detailed_info = get_chat_response(prompt)

    # Display assistant response with optional expander
    with st.chat_message("assistant"):
        st.markdown(main_response)
        if detailed_info: # Only show expander if there are details
            with st.expander("Show Details"):
                st.write(detailed_info)
    
    # Add assistant response (and details) to chat history
    st.session_state.messages.append({"role": "assistant", "content": main_response, "details": detailed_info})