

import requests
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Replace with your own API key
# -------------------------------
API_KEY = "5ba966907beb865aedfa2ac41f69d237"
CITY = "Pune"
URL = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

# -------------------------------
# 1. Fetch data from OpenWeatherMap
# -------------------------------
response = requests.get(URL)
data = response.json()

# Check full response (optional)
print("Raw API response:", data)

# Handle missing keys safely
if data.get("cod") != 200:
    print("❌ Error fetching weather data:", data.get("message"))
else:
    weather_info = {
        "City": data["name"],
        "Temperature (°C)": data["main"]["temp"],
        "Feels Like (°C)": data["main"]["feels_like"],
        "Humidity (%)": data["main"]["humidity"],
        "Pressure (hPa)": data["main"]["pressure"],
        "Wind Speed (m/s)": data["wind"]["speed"],
        "Weather": data["weather"][0]["description"].title(),
        "Date/Time (UTC)": datetime.utcfromtimestamp(data["dt"])
    }

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([weather_info])
    print("\n✅ Weather Data Retrieved:\n", df)

    # -------------------------------
    # 2. Simple Visualization
    # -------------------------------
    df[["Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)"]].plot(
        kind="bar",
        title=f"Current Weather in {CITY}",
        figsize=(8, 5)
    )
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.show()
