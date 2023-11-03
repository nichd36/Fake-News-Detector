import streamlit as st

padding = 20

image_path = "/Users/nichdylan/Documents/Natural Language Processing/NLP fake news/DSC_0424-Edited.jpg"

st.markdown(
    """
    <style>
   .sidebar .sidebar-content {
        background: url({image_path})
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('About me')

if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

col1, col2 = st.columns([3,3])

with col1:
        st.header("Nicholas Dylan")
        st.markdown("0133646")
with col2:
        st.image(image_path, use_column_width=True)