import streamlit as st
import re

def detect_device():
    """Detect if the user is on mobile or desktop based on user agent"""
    try:
        # Method 1: Try modern Streamlit context headers
        if hasattr(st, 'context') and hasattr(st.context, 'headers'):
            headers = dict(st.context.headers)
            user_agent = headers.get('user-agent', '').lower()
            
        # Method 2: Try legacy websocket headers (for older Streamlit versions)
        elif hasattr(st.web.server, 'websocket_headers'):
            from streamlit.web.server.websocket_headers import get_websocket_headers
            headers = get_websocket_headers()
            user_agent = headers.get('User-Agent', '').lower()
            
        # Method 3: Use query parameters as manual override
        else:
            query_params = st.experimental_get_query_params()
            if 'mobile' in query_params:
                return 'mobile' if query_params['mobile'][0].lower() == 'true' else 'desktop'
            user_agent = ''
        
        # If we got a user agent, check for mobile patterns
        if user_agent:
            mobile_patterns = [
                'mobile', 'android', 'iphone', 'ipad', 'ipod',
                'blackberry', 'windows phone', 'opera mini', 'webos'
            ]
            
            is_mobile = any(pattern in user_agent for pattern in mobile_patterns)
            return 'mobile' if is_mobile else 'desktop'
        
        # If no user agent available, return desktop as default
        return 'desktop'
        
    except Exception as e:
        st.sidebar.error(f"Detection error: {str(e)}")
        return 'desktop'

def show_device_detector():
    """Show a JavaScript-based device detector that sets URL parameters"""
    st.markdown("""
    <script>
    // Check if we need to redirect with device parameter
    const urlParams = new URLSearchParams(window.location.search);
    if (!urlParams.has('device_detected')) {
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        const deviceType = isMobile ? 'mobile' : 'desktop';
        
        // Redirect with device parameter
        const newUrl = window.location.href + 
            (window.location.href.includes('?') ? '&' : '?') + 
            `device=${deviceType}&device_detected=true`;
        window.location.href = newUrl;
    }
    </script>
    """, unsafe_allow_html=True)

def main():
    # Check if we have device detection from URL parameters first
    query_params = st.experimental_get_query_params()
    
    if 'device' in query_params and 'device_detected' in query_params:
        # Use JavaScript detection result
        device_type = query_params['device'][0]
    else:
        # Try server-side detection
        device_type = detect_device()
        
        # If server-side detection failed, use JavaScript fallback
        if device_type == 'desktop':  # This might be wrong, so let's use JS
            show_device_detector()
            st.info("🔄 Detecting device type...")
            st.stop()  # Stop execution until redirect happens
    
    # Add debug info in sidebar
    with st.sidebar:
        st.write(f"**Device Type:** {device_type}")
        
        # Manual override for testing
        manual_override = st.selectbox(
            "Override (for testing)", 
            ["Use Detection", "Force Mobile", "Force Desktop"],
            key="manual_override"
        )
        
        if manual_override == "Force Mobile":
            device_type = 'mobile'
        elif manual_override == "Force Desktop":
            device_type = 'desktop'
        
        # Show debug info
        if st.checkbox("Show Debug Info"):
            st.write("**Query Params:**", dict(query_params))
            try:
                if hasattr(st, 'context') and hasattr(st.context, 'headers'):
                    headers = dict(st.context.headers)
                    st.write("**Headers Available:**", "✅")
                    st.write("**User Agent:**", headers.get('user-agent', 'Not found'))
                else:
                    st.write("**Headers Available:**", "❌")
            except Exception as e:
                st.write("**Header Error:**", str(e))
    
    # Route to appropriate app
    if device_type == 'mobile':
        st.success("📱 Mobile Version Loaded")
        try:
            import mobile
            mobile.run_mobile_app()
        except ImportError as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure mobile.py exists with run_mobile_app() function")
        except Exception as e:
            st.error(f"❌ Mobile app error: {str(e)}")
    else:
        st.success("🖥️ Desktop Version Loaded")
        try:
            import computer
            computer.run_desktop_app()
        except ImportError as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure computer.py exists with run_desktop_app() function")
        except Exception as e:
            st.error(f"❌ Desktop app error: {str(e)}")

if __name__ == "__main__":
    main()
