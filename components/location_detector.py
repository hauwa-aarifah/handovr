# components/location_detector.py
import streamlit as st
import streamlit.components.v1 as components

def get_location():
    """
    Component to get user's current location using browser geolocation API
    """
    
    location_html = """
    <script>
    const getLocation = () => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    // Send location to Streamlit
                    const locationData = {
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude,
                        accuracy: position.coords.accuracy
                    };
                    
                    // Create a custom event to pass data to Streamlit
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        key: 'location_data',
                        value: locationData
                    }, '*');
                    
                    // Also update the display
                    document.getElementById('location-status').innerHTML = 
                        `✅ Location detected: ${position.coords.latitude.toFixed(4)}, ${position.coords.longitude.toFixed(4)}`;
                },
                (error) => {
                    let errorMsg = 'Unable to get location: ';
                    switch(error.code) {
                        case error.PERMISSION_DENIED:
                            errorMsg += "Permission denied. Please allow location access.";
                            break;
                        case error.POSITION_UNAVAILABLE:
                            errorMsg += "Location unavailable.";
                            break;
                        case error.TIMEOUT:
                            errorMsg += "Request timed out.";
                            break;
                        default:
                            errorMsg += "Unknown error.";
                    }
                    document.getElementById('location-status').innerHTML = `❌ ${errorMsg}`;
                },
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 0
                }
            );
        } else {
            document.getElementById('location-status').innerHTML = 
                '❌ Geolocation is not supported by your browser.';
        }
    };
    
    // Auto-detect on load
    window.addEventListener('load', getLocation);
    </script>
    
    <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
        <button onclick="getLocation()" style="
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-bottom: 10px;
        ">
            📍 Detect My Location
        </button>
        <div id="location-status" style="
            color: #333;
            font-size: 14px;
            text-align: center;
            margin-top: 10px;
        ">
            Waiting for location...
        </div>
    </div>
    """
    
    return components.html(location_html, height=120)

def get_location_from_address(address):
    """
    Fallback method to get approximate coordinates from an address
    Using a simple mapping for London areas
    """
    
    # London area coordinates
    london_areas = {
        "central london": (51.5074, -0.1278),
        "westminster": (51.4975, -0.1357),
        "camden": (51.5290, -0.1225),
        "islington": (51.5416, -0.1025),
        "hackney": (51.5450, -0.0553),
        "tower hamlets": (51.5150, -0.0172),
        "greenwich": (51.4825, 0.0000),
        "southwark": (51.5030, -0.0900),
        "lambeth": (51.4607, -0.1160),
        "wandsworth": (51.4567, -0.1910),
        "hammersmith": (51.4927, -0.2240),
        "kensington": (51.5021, -0.1916),
        "city of london": (51.5155, -0.0922),
        "barking": (51.5362, 0.0798),
        "brent": (51.5586, -0.2636),
        "ealing": (51.5130, -0.3089),
        "haringey": (51.5906, -0.1110),
        "newham": (51.5255, 0.0352),
        "redbridge": (51.5784, 0.0465),
        "richmond": (51.4479, -0.3260),
        "croydon": (51.3714, -0.0977),
        "bromley": (51.4039, 0.0198),
        "lewisham": (51.4452, -0.0209),
        "merton": (51.4098, -0.1949),
        "sutton": (51.3618, -0.1945),
        "hounslow": (51.4668, -0.3615),
        "hillingdon": (51.5441, -0.4760),
        "havering": (51.5779, 0.2120),
        "bexley": (51.4411, 0.1486),
        "enfield": (51.6522, -0.0808),
        "waltham forest": (51.5908, -0.0127),
        "barnet": (51.6444, -0.1997),
        "harrow": (51.5792, -0.3415),
        "kingston": (51.4085, -0.2681)
    }
    
    # Try to match the address to an area
    address_lower = address.lower()
    
    for area, coords in london_areas.items():
        if area in address_lower:
            return coords
    
    # Default to central London if no match
    return (51.5074, -0.1278)