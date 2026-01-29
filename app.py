import streamlit as st

# Import UI components and page functions
from core.ui import (
    setup_page_config,
    render_header,
    create_sidebar_navigation,
    render_sidebar_footer,
    page_sbox_analysis,
    page_text_encryption,
    page_image_encryption,
    load_prebuilt_sboxes
)


# Setup page configuration
setup_page_config()

# Render header
render_header()


def main():
    """Main Streamlit app."""
    # Initialize session state
    if 'current_sbox' not in st.session_state:
        # Load AES Standard as default
        prebuilt_sboxes = load_prebuilt_sboxes()
        if 'AES Standard' in prebuilt_sboxes:
            sbox_data = prebuilt_sboxes['AES Standard']
            st.session_state.current_sbox = sbox_data['sbox']
            st.session_state.current_sbox_name = sbox_data['name']
        else:
            st.session_state.current_sbox = None
            st.session_state.current_sbox_name = None
    
    # Navigation
    page = create_sidebar_navigation()
    
    render_sidebar_footer()
    
    # Route pages
    if page == "Text Encryption":
        page_text_encryption()
    elif page == "Image Encryption":
        page_image_encryption()
    else:  # page == "S-box Analysis"
        page_sbox_analysis()


if __name__ == "__main__":
    main()

