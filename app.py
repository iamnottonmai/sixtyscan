import streamlit as st
import re

def detect_device():
    """Detect if the user is on mobile or desktop based on user agent"""
    try:
        # Method 1: Try to get user agent from headers (newer Streamlit versions)
        if hasattr(st, 'context') and hasattr(st.context, 'headers'):
            headers = dict(st.context.headers)
            user_agent = headers.get('user-agent', '').lower()
        else:
            # Method 2: Fallback method using query params or session state
            user_agent = st.session_state.get('user_agent', '').lower()
        
        # If still no user agent, try JavaScript detection
        if not user_agent:
            return detect_with_javascript()
        
        # Mobile device patterns
        mobile_patterns = [
            r'mobile', r'android', r'iphone', r'ipad', r'ipod',
            r'blackberry', r'windows phone', r'opera mini', r'webos'
        ]
        
        # Check if any mobile pattern matches
        is_mobile = any(re.search(pattern, user_agent) for pattern in mobile_patterns)
        
        return 'mobile' if is_mobile else 'desktop'
    
    except Exception as e:
        # Show the actual error for debugging
        st.sidebar.error(f"Device detection error: {str(e)}")
        return 'desktop'  # Fallback to desktop

def detect_with_javascript():
    """Alternative detection using JavaScript"""
    # Check if we've already detected the device
    if 'device_type' not in st.session_state:
        # Use JavaScript to detect mobile
        js_code = """
        <script>
        function detectDevice() {
            const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
            const deviceType = isMobile ? 'mobile' : 'desktop';
            
            // Try to communicate with Streamlit
            if (window.parent && window.parent.postMessage) {
                window.parent.postMessage({
                    type: 'streamlit:componentReady',
                    deviceType: deviceType
                }, '*');
            }
            
            return deviceType;
        }
        
        // Set a cookie that we can read from Python
        document.cookie = `device_type=${detectDevice()}; path=/`;
        </script>
        """
        
        st.components.v1.html(js_code, height=0)
        
        # For now, return desktop and let user refresh
        st.info("🔄 Please refresh the page to properly detect your device type.")
        return 'desktop'
    
    return st.session_state.device_type

def main():
    # Add debug mode toggle
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    # Manual override for testing
    manual_override = st.sidebar.selectbox(
        "Manual Device Override (for testing)", 
        ["Auto-detect", "Force Mobile", "Force Desktop"]
    )
    
    if manual_override == "Force Mobile":
        device_type = 'mobile'
    elif manual_override == "Force Desktop":
        device_type = 'desktop'
    else:
        device_type = detect_device()
    
    # Debug information
    if debug_mode:
        st.sidebar.write("**Debug Info:**")
        st.sidebar.write(f"Detected device: **{device_type}**")
        
        try:
            if hasattr(st, 'context') and hasattr(st.context, 'headers'):
                headers = dict(st.context.headers)
                user_agent = headers.get('user-agent', 'Not found')
                st.sidebar.write(f"User Agent: `{user_agent}`")
        except:
            st.sidebar.write("Could not access headers")
    
    # Route to appropriate app
    if device_type == 'mobile':
        st.sidebar.success("📱 Loading Mobile Version")
        try:
            import mobile
            mobile.run_mobile_app()
        except ImportError as e:
            st.error(f"❌ mobile.py file not found! Error: {str(e)}")
            st.info("Make sure mobile.py exists and contains a run_mobile_app() function")
        except Exception as e:
            st.error(f"❌ Error loading mobile app: {str(e)}")
    else:
        st.sidebar.success("🖥️ Loading Desktop Version")
        try:
            import computer
            computer.run_desktop_app()
        except ImportError as e:
            st.error(f"❌ computer.py file not found! Error: {str(e)}")
            st.info("Make sure computer.py exists and contains a run_desktop_app() function")
        except Exception as e:
            st.error(f"❌ Error loading desktop app: {str(e)}")

if __name__ == "__main__":
    main()
