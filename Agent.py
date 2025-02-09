import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
from pathlib import Path
import tempfile
import os
import time

# Configure the Streamlit page
st.set_page_config(
    page_title="AI‑Powered Media Critic",
    page_icon="🎨",
    layout="wide"
)

# App title and description
st.title("AI‑Powered Media Critic 🎨")
st.write(
    """
    Welcome to the AI‑Powered Media Critic – your creative companion for multimedia analysis.
    Whether you want expert feedback on a digital artwork or a cinematic critique of your video,
    our AI agent leverages advanced Google Gemini models and real‑time web research to provide 
    detailed, actionable insights.
    """
)

# Sidebar configuration
st.sidebar.header("Configuration & Mode")
api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")
media_type = st.sidebar.radio("Select Media Type", options=["Image Critique", "Video Critique"])

# Only proceed if the API key is provided
if not api_key:
    st.error("An API key is required to initialize the AI agent. Please enter your API key above.")
    st.stop()

# Initialize the agent using a cached resource function
@st.cache_resource(show_spinner=False)
def initialize_agent(api_key: str):
    return Agent(
        name="AI Media Critic",
        model=Gemini(id="gemini-2.0-flash-exp", api_key=api_key),
        tools=[DuckDuckGo()],
        markdown=True,
    )

agent = initialize_agent(api_key)
st.sidebar.success("Agent initialized successfully!")

# Depending on the selected media type, show the appropriate uploader and input fields
if media_type == "Image Critique":
    st.subheader("Image Critique Mode")
    st.write(
        "Upload an image (JPEG or PNG) to receive creative feedback on its composition, "
        "color balance, and artistic style."
    )
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Save the uploaded image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                image_path = tmp_file.name

            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            # Input area for the user’s creative query
            task_input = st.text_area(
                "What creative feedback would you like? (e.g. Analyze the composition, lighting, and mood)",
                ""
            )
            
            # When the user clicks the button, call the agent
            if st.button("Analyze Image") and task_input:
                with st.spinner("Analyzing image..."):
                    try:
                        response = agent.run(f"Critique this image based on {task_input}. Provide a detailed analysis.", images=[image_path])
                        if response and hasattr(response, 'content'):
                            st.markdown(f"### Critique:\n{response.content}")
                        else:
                            st.error("Failed to generate a critique. The AI might not have processed the image correctly.")
                    except Exception as e:
                        st.error(f"An error occurred during image analysis: {str(e)}")
                    finally:
                        if os.path.exists(image_path):
                            os.unlink(image_path)
        except Exception as e:
            st.error(f"An error occurred while processing the image: {str(e)}")

elif media_type == "Video Critique":
    st.subheader("Video Critique Mode")
    st.write(
        "Upload a video file (MP4, MOV, or AVI) to receive an in‑depth cinematic critique "
        "that covers narrative, pacing, visual style, and technical execution."
    )
    uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file is not None:
        try:
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            # Display the uploaded video
            st.video(video_path)
            
            # Input area for the user’s query about the video
            user_prompt = st.text_area(
                "What aspects of the video would you like critiqued? (e.g. Evaluate the narrative flow, pacing, and visual style)",
                ""
            )
            
            # When the user clicks the button, process the video and perform research
            if st.button("Analyze & Critique Video") and user_prompt:
                with st.spinner("Analyzing video and researching..."):
                    try:
                        # Upload the video file to the AI service and wait until processing is complete
                        video_file = upload_file(video_path)
                        while video_file.state.name == "PROCESSING":
                            time.sleep(2)
                            video_file = get_file(video_file.name)
                        
                        result = agent.run(f"Critique this video based on {user_prompt}. Provide a detailed analysis.", videos=[video_file])
                        if result and hasattr(result, 'content'):
                            st.subheader("Critique:")
                            st.markdown(result.content)
                        else:
                            st.error("Failed to generate a critique. The AI might not have processed the video correctly.")
                    except Exception as e:
                        st.error(f"An error occurred during video analysis: {str(e)}")
                    finally:
                        Path(video_path).unlink(missing_ok=True)
        except Exception as e:
            st.error(f"An error occurred while processing the video: {str(e)}")

# Optional: Increase the height of text areas for better usability
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 150px;
    }
    </style>
    """,
    unsafe_allow_html=True
)





