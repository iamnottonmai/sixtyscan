import streamlit as st
import re
from typing import Literal, Optional

DeviceType = Literal['mobile', 'desktop']

class DeviceDetector:
    """Enhanced device detection for Streamlit applications"""
    
    @staticmethod
    def detect_from_headers() -> Optional[DeviceType]:
        """Try to detect device from HTTP headers (multiple methods)"""
        try:
            # Method 1: Modern Streamlit context (v1.28+)
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                ctx = get_script_run_ctx()
                if ctx and hasattr(ctx, 'session_state'):
                    # This approach may vary by Streamlit version
                    pass
            except ImportError:
                pass
            
            # Method 2: Try experimental headers access
            try:
                # This is version-dependent and may not work in all environments
                headers = st.context.headers if hasattr(st, 'context') else None
                if headers:
                    user_agent = headers.get('user-agent', '').lower()
                    return DeviceDetector._classify_user_agent(user_agent)
            except (AttributeError, NameError):
                pass
            
            # Method 3: Check if running in cloud environment
            # Some cloud platforms provide device info differently
            return None
            
        except Exception:
            return None
    
    @staticmethod
    def detect_from_url_params() -> Optional[DeviceType]:
        """Get device type from URL parameters"""
        try:
            query_params = st.query_params
            if 'device' in query_params:
                device = query_params['device'].lower()
                return 'mobile' if device == 'mobile' else 'desktop'
            return None
        except Exception:
            return None
    
    @staticmethod
    def _classify_user_agent(user_agent: str) -> DeviceType:
        """Classify device type based on user agent string"""
        mobile_patterns = [
            'mobile', 'android', 'iphone', 'ipad', 'ipod',
            'blackberry', 'windows phone', 'opera mini', 'webos',
            'kindle', 'silk', 'fennec', 'maemo', 'meego'
        ]
        
        # More specific tablet detection
        tablet_patterns = ['ipad', 'tablet', 'kindle', 'silk']
        
        user_agent = user_agent.lower()
        
        # Check for tablets first (treat as mobile for responsive design)
        if any(pattern in user_agent for pattern in tablet_patterns):
            return 'mobile'
        
        # Check for mobile patterns
        if any(pattern in user_agent for pattern in mobile_patterns):
            return 'mobile'
        
        return 'desktop'

def create_js_detector() -> str:
    """Create JavaScript code for client-side device detection"""
    return """
    <script>
    function detectAndRedirect() {
        const urlParams = new URLSearchParams(window.location.search);
        
        // Skip if already detected
        if (urlParams.has('device_detected')) {
            return;
        }
        
        // Enhanced mobile detection
        function isMobileDevice() {
            // Check user agent
            const userAgent = navigator.userAgent || navigator.vendor || window.opera;
            const mobileRegex = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile|mobile|CriOS/i;
            
            // Check screen size as additional indicator
            const isSmallScreen = window.innerWidth <= 768 || window.innerHeight <= 1024;
            
            // Check for touch capability
            const isTouchDevice = ('ontouchstart' in window) || 
                                 (navigator.maxTouchPoints > 0) || 
                                 (navigator.msMaxTouchPoints > 0);
            
            // Combine checks for better accuracy
            const agentMatch = mobileRegex.test(userAgent);
            const likelyMobile = agentMatch || (isSmallScreen && isTouchDevice);
            
            return likelyMobile;
        }
        
        const deviceType = isMobileDevice() ? 'mobile' : 'desktop';
        
        // Add parameters and redirect
        urlParams.set('device', deviceType);
        urlParams.set('device_detected', 'true');
        
        const newUrl = window.location.pathname + '?' + urlParams.toString();
        
        // Only redirect if URL will actually change
        if (window.location.search !== '?' + urlParams.toString()) {
            window.location.href = newUrl;
        }
    }
    
    // Run detection when page loads
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', detectAndRedirect);
    } else {
        detectAndRedirect();
    }
    </script>
    """

def show_device_info(device_type: DeviceType, detection_method: str):
    """Display device detection information"""
    device_emoji = "📱" if device_type == 'mobile' else "🖥️"
    device_name = "Mobile" if device_type == 'mobile' else "Desktop"
    
    st.success(f"{device_emoji} {device_name} Version Loaded")
    
    with st.expander("ℹ️ Detection Details"):
        st.info(f"**Device Type:** {device_name}")
        st.info(f"**Detection Method:** {detection_method}")
        
        # Show current viewport info using JavaScript
        st.markdown("""
        <div id="viewport-info"></div>
        <script>
        document.getElementById('viewport-info').innerHTML = `
            <strong>Screen Info:</strong><br>
            Width: ${window.innerWidth}px<br>
            Height: ${window.innerHeight}px<br>
            User Agent: ${navigator.userAgent.substring(0, 100)}...
        `;
        </script>
        """, unsafe_allow_html=True)

def create_debug_panel():
    """Create a comprehensive debug panel"""
    with st.sidebar:
        st.header("🔧 Debug Panel")
        
        # Manual override
        manual_override = st.selectbox(
            "Manual Override", 
            ["Use Auto-Detection", "Force Mobile", "Force Desktop"],
            key="device_override"
        )
        
        override_device = None
        if manual_override == "Force Mobile":
            override_device = 'mobile'
        elif manual_override == "Force Desktop":
            override_device = 'desktop'
        
        # Debug information
        if st.checkbox("Show Debug Info", key="show_debug"):
            st.subheader("Debug Information")
            
            # URL parameters
            try:
                query_params = st.query_params
                st.json({
                    "Query Parameters": dict(query_params),
                    "Has Device Param": 'device' in query_params,
                    "Has Detection Flag": 'device_detected' in query_params
                })
            except Exception as e:
                st.error(f"Query params error: {e}")
            
            # Detection attempts
            header_detection = DeviceDetector.detect_from_headers()
            url_detection = DeviceDetector.detect_from_url_params()
            
            st.write("**Detection Results:**")
            st.write(f"- Header Detection: {header_detection or 'Failed'}")
            st.write(f"- URL Detection: {url_detection or 'Not set'}")
            
            # Browser info (client-side)
            st.markdown("""
            <div id="browser-info"></div>
            <script>
            document.getElementById('browser-info').innerHTML = `
                <strong>Browser Info:</strong><br>
                Platform: ${navigator.platform}<br>
                Language: ${navigator.language}<br>
                Cookies Enabled: ${navigator.cookieEnabled}<br>
                Online: ${navigator.onLine}
            `;
            </script>
            """, unsafe_allow_html=True)
        
        return override_device

def load_app_module(device_type: DeviceType):
    """Load and run the appropriate app module"""
    try:
        if device_type == 'mobile':
            try:
                import mobile
                if hasattr(mobile, 'run_mobile_app'):
                    mobile.run_mobile_app()
                else:
                    st.error("❌ mobile.py exists but missing run_mobile_app() function")
            except ImportError:
                st.error("❌ mobile.py not found")
                st.info("Create mobile.py with a run_mobile_app() function")
                create_sample_mobile_app()
        else:
            try:
                import computer
                if hasattr(computer, 'run_desktop_app'):
                    computer.run_desktop_app()
                else:
                    st.error("❌ computer.py exists but missing run_desktop_app() function")
            except ImportError:
                st.error("❌ computer.py not found")
                st.info("Create computer.py with a run_desktop_app() function")
                create_sample_desktop_app()
                
    except Exception as e:
        st.error(f"❌ Error loading app: {str(e)}")
        st.exception(e)

def create_sample_mobile_app():
    """Show a sample mobile app when mobile.py is missing"""
    st.subheader("📱 Sample Mobile App")
    st.write("This is a placeholder. Create `mobile.py` with your mobile-optimized interface.")
    
    # Mobile-friendly layout example
    col1, col2 = st.columns(2)
    with col1:
        st.button("🏠 Home", use_container_width=True)
        st.button("📊 Stats", use_container_width=True)
    with col2:
        st.button("⚙️ Settings", use_container_width=True)
        st.button("ℹ️ About", use_container_width=True)

def create_sample_desktop_app():
    """Show a sample desktop app when computer.py is missing"""
    st.subheader("🖥️ Sample Desktop App")
    st.write("This is a placeholder. Create `computer.py` with your desktop interface.")
    
    # Desktop layout example
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button("Home")
    with col2:
        st.button("Dashboard")
    with col3:
        st.button("Analytics")
    with col4:
        st.button("Settings")

def main():
    """Main application logic with enhanced device detection"""
    
    st.set_page_config(
        page_title="Device-Aware App",
        page_icon="📱",
        layout="wide"
    )
    
    # Create debug panel and get manual override
    override_device = create_debug_panel()
    
    # Determine device type through multiple methods
    device_type = None
    detection_method = "Unknown"
    
    # 1. Check manual override first
    if override_device:
        device_type = override_device
        detection_method = "Manual Override"
    
    # 2. Check URL parameters (from JavaScript detection)
    elif DeviceDetector.detect_from_url_params():
        device_type = DeviceDetector.detect_from_url_params()
        detection_method = "JavaScript Detection"
    
    # 3. Try header-based detection
    elif DeviceDetector.detect_from_headers():
        device_type = DeviceDetector.detect_from_headers()
        detection_method = "Server Headers"
    
    # 4. If no detection successful, use JavaScript fallback
    else:
        st.markdown(create_js_detector(), unsafe_allow_html=True)
        st.info("🔄 Detecting device type... Please wait a moment.")
        st.markdown("""
        <div style="text-align: center; margin: 20px;">
            <div style="display: inline-block; width: 40px; height: 40px; border: 3px solid #f3f3f3; border-radius: 50%; border-top: 3px solid #3498db; animation: spin 1s linear infinite;"></div>
        </div>
        <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Show device info
    show_device_info(device_type, detection_method)
    
    # Load appropriate app
    load_app_module(device_type)

if __name__ == "__main__":
    main()
