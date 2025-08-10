import streamlit as st
import re
from streamlit.web.server.websocket_headers import _get_websocket_headers

def detect_device():
    """Detect if the user is on mobile or desktop based on user agent"""
    try:
        # Get headers from the current session
        headers = _get_websocket_headers()
        user_agent = headers.get('User-Agent', '').lower()
        
        # Mobile device patterns
        mobile_patterns = [
            r'mobile', r'android', r'iphone', r'ipad', r'ipod',
            r'blackberry', r'windows phone', r'opera mini'
        ]
        
        # Check if any mobile pattern matches
        is_mobile = any(re.search(pattern, user_agent) for pattern in mobile_patterns)
        
        return 'mobile' if is_mobile else 'desktop'
    
    except Exception:
        # Fallback to desktop if detection fails
        return 'desktop'

def main():
    device_type = detect_device()
    
    # Optional: Display current device detection in sidebar (for debugging)
    # st.sidebar.write(f"Detected device: {device_type}")
    
    if device_type == 'mobile':
        # Import and run mobile version
        try:
            import mobile
            mobile.run_mobile_app()
        except ImportError:
            st.error("mobile.py file not found!")
    else:
        # Import and run desktop version
        try:
            import computer
            computer.run_desktop_app()
        except ImportError:
            st.error("computer.py file not found!")

if __name__ == "__main__":
    main()
