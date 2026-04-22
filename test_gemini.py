"""
Run this once to diagnose Gemini API issues.
Add a button in your Streamlit app or run directly.
"""
import os
import streamlit as st

st.title("Gemini API Diagnostic")

if st.button("Run Test"):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    st.write(f"API Key found: {'YES — ' + api_key[:8] + '...' if api_key else 'NO — KEY IS MISSING'}")

    if not api_key:
        st.error("GEMINI_API_KEY is not set in Secrets!")
        st.stop()

    # Test 1: import
    try:
        from google import genai
        from google.genai import types
        import google.genai as genai_module
        st.success(f"✅ google-genai imported — version: {genai_module.__version__}")
    except Exception as e:
        st.error(f"❌ Import failed: {e}")
        st.stop()

    # Test 2: client
    try:
        client = genai.Client(api_key=api_key)
        st.success("✅ Client created")
    except Exception as e:
        st.error(f"❌ Client creation failed: {e}")
        st.stop()

    # Test 3: simple text call
    try:
        text_part = types.Part(text="Say hello in one word.")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[text_part],
            config=types.GenerateContentConfig(max_output_tokens=50)
        )
        st.success(f"✅ Text API call works — response: {response.text}")
    except Exception as e:
        st.error(f"❌ Text API call failed: {e}")
        st.stop()

    # Test 4: image call
    try:
        import base64, io
        from PIL import Image
        import numpy as np

        # Create a simple test image (gray 100x100)
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        image_part = types.Part(
            inline_data=types.Blob(mime_type="image/jpeg", data=img_bytes)
        )
        text_part2 = types.Part(text="What color is this image? One word.")

        response2 = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[text_part2, image_part],
            config=types.GenerateContentConfig(max_output_tokens=50)
        )
        st.success(f"✅ Image API call works — response: {response2.text}")
    except Exception as e:
        st.error(f"❌ Image API call failed: {e}")

    st.info("Diagnosis complete. Share the results above.")
