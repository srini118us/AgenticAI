# =============================
# Imports and Environment Setup
# =============================
import streamlit as st
import os
from datetime import datetime
import pandas as pd
import requests
from langchain_community.utilities import GooglePlacesAPIWrapper, SerpAPIWrapper
from langchain_google_community import GooglePlacesTool
# from langchain_core.messages import HumanMessage, SystemMessage
from forex_python.converter import CurrencyRates
from langgraph.graph import MessagesState, StateGraph, END, START
import networkx as nx
import matplotlib.pyplot as plt
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from IPython.display import display, Image
from amadeus import Client, ResponseError
load_dotenv()

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =============================
# Service Classes
# (Config, WeatherService, CurrencyService, ExpenseCalculator, ItineraryService, SummaryService, TravelAgent)
# These classes encapsulate the business logic for each travel agent feature.
# =============================


class Config:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY')
        self.exchange_rate_api_key = os.getenv('EXCHANGE_RATE_API_KEY')
        self.google_places_api_key = os.getenv('GPLACES_API_KEY')
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        self.serper_api_key = os.getenv('SERPER_API_KEY')


class WeatherService:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_current_weather(self, city):
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            weather = {
                "city": city,
                "temperature": data["main"]["temp"],
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"]
            }
            return weather
        else:
            return {"error": f"Failed to get weather for {city}: {response.text}"}

    def get_forecast(self, city, days=3):
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
        # The forecast API returns data in 3-hour intervals for 5 days
        # You can process this to get daily summaries
            forecasts = []
            for entry in data["list"]:
                forecast = {
                    "datetime": entry["dt_txt"],
                    "temperature": entry["main"]["temp"],
                    "description": entry["weather"][0]["description"]
                }
            forecasts.append(forecast)
            return forecasts[:days*8]  # 8 intervals per day
        else:
            return {"error": f"Failed to get forecast for {city}: {response.text}"}


class CurrencyService:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_exchange_rate(self, from_currency, to_currency):
        url = f"https://v6.exchangerate-api.com/v6/{self.api_key}/latest/{from_currency}"
        response = requests.get(url)
        data = response.json()
        if data["result"] == "success":
            return data["conversion_rates"].get(to_currency)
        return None

    def convert_currency(self, amount, from_currency, to_currency):
        rate = self.get_exchange_rate(from_currency, to_currency)
        if rate:
            return amount * rate
        return None


class ExpenseCalculator:
    def calculate_total_cost(self, hotel_cost, attraction_cost, food_cost, transport_cost, misc_cost):
        """
        Calculate the total trip cost by summing all provided costs.
        """
        try:
            return sum([
                float(hotel_cost or 0),
                float(attraction_cost or 0),
                float(food_cost or 0),
                float(transport_cost or 0),
                float(misc_cost or 0)
            ])
        except Exception as e:
            return f"Error calculating total cost: {e}"

    def calculate_daily_budget(self, total_cost, days):
        """
        Calculate the daily budget given total cost and number of days.
        """
        try:
            days = int(days)
            if days <= 0:
                return 0
            return float(total_cost) / days
        except Exception as e:
            return f"Error calculating daily budget: {e}"


class ItineraryService:
    def create_day_plan(self, city, attractions, weather):
        """
        Create a plan for a single day.
        """
        return {
            "city": city,
            "attractions": attractions,
            "weather": weather
        }

    def create_full_itinerary(self, days, day_plans):
        """
        Combine day plans into a full itinerary.
        """
        itinerary = []
        for i in range(int(days)):
            if i < len(day_plans):
                itinerary.append(day_plans[i])
            else:
                itinerary.append({"day": i+1, "note": "No plan available"})
        return itinerary


class SummaryService:
    def generate_summary(self, all_data):
        """
        Generate a detailed summary of the trip, including all key details for each day.
        """
        summary_lines = []
        city = all_data.get('city', 'Unknown')
        days = int(all_data.get('days', 1))
        summary_lines.append(f"Trip Summary for {city} ({days} days)")
        summary_lines.append("-" * 40)
        # Multi-day itinerary
        for i in range(days):
            summary_lines.append(f"\nDay {i+1} in {city}:")
            # Weather
            weather = None
            if 'weather' in all_data:
                if isinstance(all_data['weather'], list) and i < len(all_data['weather']):
                    weather = all_data['weather'][i]
                else:
                    weather = all_data['weather']
            if weather:
                summary_lines.append(f"  - Weather: {weather}")
            # Attractions
            if 'attractions' in all_data:
                summary_lines.append(f"  - Attractions: {all_data['attractions']}")
            # Hotels
            if 'hotels' in all_data:
                summary_lines.append(f"  - Hotels: {all_data['hotels']}")
            # Activities
            if 'activities' in all_data:
                summary_lines.append(f"  - Activities: {all_data['activities']}")
            # Events
            if 'events' in all_data:
                summary_lines.append(f"  - Events: {all_data['events']}")
            # Place Details
            if 'place_details' in all_data:
                summary_lines.append(f"  - Place Details: {all_data['place_details']}")
            # Itinerary
            if 'itinerary' in all_data and isinstance(all_data['itinerary'], list) and i < len(all_data['itinerary']):
                summary_lines.append(f"  - Itinerary: {all_data['itinerary'][i]}")
        # Costs
        if 'total_cost' in all_data:
            summary_lines.append(f"\nEstimated Total Cost: {all_data['total_cost']}")
        if 'daily_budget' in all_data:
            summary_lines.append(f"Estimated Daily Budget: {all_data['daily_budget']}")
        # Fallback info (if present)
        if all_data.get('attractions_fallback_used'):
            summary_lines.append("Attractions fallback used: Yes")
        if all_data.get('hotels_fallback_used'):
            summary_lines.append("Hotels fallback used: Yes")
        if all_data.get('use_activities_fallback'):
            summary_lines.append("Activities fallback used: Yes")
        if all_data.get('use_events_fallback'):
            summary_lines.append("Events fallback used: Yes")
        return "\n".join(summary_lines)


# =============================
# AttractionService and HotelService
# These classes wrap GooglePlacesTool and SerpAPIWrapper for attractions and hotels.
# =============================

class AttractionService:
    def __init__(self, google_places_api_key, serpapi_key):
        self.places_tool = GooglePlacesTool(
            gplaces_api_key=google_places_api_key)
        self.serpapi_wrapper = SerpAPIWrapper(serpapi_api_key=serpapi_key)

    def search_attractions(self, city):
        try:
            result = self.places_tool.run(f"tourist attractions in {city}")
            if result and len(result) > 0:
                return result
        except Exception:
            pass
        return self.serpapi_wrapper.run(f"top attractions in {city}")


class HotelService:
    def __init__(self, google_places_api_key, serpapi_key):
        self.places_tool = GooglePlacesTool(
            gplaces_api_key=google_places_api_key)
        self.serpapi_wrapper = SerpAPIWrapper(serpapi_api_key=serpapi_key)

    def search_hotels(self, city, budget="mid-range"):
        try:
            result = self.places_tool.run(f"{budget} hotels in {city}")
            if result and len(result) > 0:
                return result
        except Exception:
            pass
        return self.serpapi_wrapper.run(f"{budget} hotels in {city}")

    def estimate_cost(self, price_per_night, days):
        try:
            return float(price_per_night) * int(days)
        except Exception as e:
            return f"Error estimating hotel cost: {e}"


class TravelAgent:
    def __init__(self, config):
        self.weather = WeatherService(config.openweather_api_key)
        self.attractions = AttractionService(
            config.google_places_api_key, config.serpapi_key)
        self.hotels = HotelService(
            config.google_places_api_key, config.serpapi_key)
        self.currency = CurrencyService(config.exchange_rate_api_key)
        self.expense = ExpenseCalculator()
        self.itinerary = ItineraryService()
        self.summary = SummaryService()

    def plan_trip(self, city, days, budget, currency="USD"):
        # 1. Get weather
        weather = self.weather.get_forecast(city, days)
        # 2. Get attractions
        attractions = self.attractions.search_attractions(city)
        # 3. Get hotels
        hotels = self.hotels.search_hotels(city, budget)
        # 4. Estimate hotel cost (example: use first hotel, dummy price)
        price_per_night = 100  # You could extract this from hotel data if available
        hotel_cost = self.hotels.estimate_cost(price_per_night, days)
        # 5. Currency conversion (if needed)
        # 6. Calculate total expense (dummy values for now)
        total_cost = self.expense.calculate_total_cost(
            hotel_cost, 100, 100, 50, 50)
        daily_budget = self.expense.calculate_daily_budget(total_cost, days)
        # 7. Create itinerary
        itinerary = self.itinerary.create_full_itinerary(days, [
            self.itinerary.create_day_plan(city, attractions, weather)
            for _ in range(days)
        ])
        # 8. Generate summary (include all info)
        summary = self.summary.generate_summary({
            "city": city,
            "days": days,
            "weather": weather,
            "attractions": attractions,
            "hotels": hotels,
            "itinerary": itinerary,
            "total_cost": total_cost,
            "daily_budget": daily_budget
        })
        return summary


config = Config()
weather_service = WeatherService(config.openweather_api_key)
currency_service = CurrencyService(config.exchange_rate_api_key)
expense_service = ExpenseCalculator()
itinerary_service = ItineraryService()
summary_service = SummaryService()
places_tool = GooglePlacesTool(gplaces_api_key=config.google_places_api_key)
serpapi_wrapper = SerpAPIWrapper(serpapi_api_key=config.serpapi_key)
# =============================
# Amadeus API Client Initialization
# =============================
amadeus = Client(
    client_id=os.getenv("AMADEUS_CLIENT_ID"),
    client_secret=os.getenv("AMADEUS_CLIENT_SECRET")
)

# =============================
# Tool Functions
# These are LangChain tool wrappers for searching attractions, hotels, weather, currency, etc.
# =============================


@tool
def search_attractions(city: str):
    """Search for top attractions in a city using Google Places, fallback to SerpAPI."""
    try:
        result = places_tool.run(f"tourist attractions in {city}")
        if result and len(result) > 0:
            return result
    except Exception:
        pass
    return serpapi_wrapper.run(f"top attractions in {city}")


@tool
def search_restaurants(city: str):
    """Search for restaurants in a city using Google Places, fallback to SerpAPI."""
    try:
        result = places_tool.run(f"restaurants in {city}")
        if result and len(result) > 0:
            return result
    except Exception:
        pass
    return serpapi_wrapper.run(f"best restaurants in {city}")


@tool
def search_activities(city: str):
    """Search for activities in a city using Google Places, fallback to SerpAPI."""
    try:
        result = places_tool.run(f"activities in {city}")
        if result and len(result) > 0:
            return result
    except Exception:
        pass
    return serpapi_wrapper.run(f"things to do in {city}")


@tool
def search_transportation(city: str):
    """Search for transportation options in a city using Google Places, fallback to SerpAPI."""
    try:
        result = places_tool.run(f"transportation in {city}")
        if result and len(result) > 0:
            return result
    except Exception:
        pass
    return serpapi_wrapper.run(f"public transport in {city}")


@tool
def search_hotels(city: str, budget: str = "mid-range"):
    """Search for hotels in a city using Google Places, fallback to SerpAPI."""
    try:
        result = places_tool.run(f"{budget} hotels in {city}")
        if result and len(result) > 0:
            return result
    except Exception:
        pass
    return serpapi_wrapper.run(f"{budget} hotels in {city}")

# =============================
# Weather Tools
# =============================


@tool
def get_current_weather(city: str):
    """Get current weather for a city."""
    return weather_service.get_current_weather(city)


@tool
def get_forecast(city: str, days: int = 3):
    """Get weather forecast for a city."""
    return weather_service.get_forecast(city, days)

# =============================
# Hotel Cost Tool
# =============================


@tool
def estimate_hotel_cost(price_per_night: float, total_nights: int):
    """Estimate the total cost for a hotel stay."""
    return price_per_night * total_nights

# =============================
# Currency Tools
# =============================


@tool
def get_exchange_rate(from_currency: str, to_currency: str):
    """Get exchange rate between two currencies."""
    return currency_service.get_exchange_rate(from_currency, to_currency)


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str):
    """Convert amount from one currency to another."""
    return currency_service.convert_currency(amount, from_currency, to_currency)

# =============================
# Expense Tools
# =============================


@tool
def calculate_total_cost(hotel_cost, attraction_cost, food_cost, transport_cost, misc_cost):
    """Calculate the total trip cost."""
    return expense_service.calculate_total_cost(hotel_cost, attraction_cost, food_cost, transport_cost, misc_cost)


@tool
def calculate_daily_budget(total_cost, days):
    """Calculate daily budget."""
    return expense_service.calculate_daily_budget(total_cost, days)

# =============================
# Itinerary Tools
# =============================


@tool
def create_day_plan(city, attractions, weather):
    """Create a plan for a single day."""
    return itinerary_service.create_day_plan(city, attractions, weather)


@tool
def create_full_itinerary(days, day_plans):
    """Combine day plans into a full itinerary."""
    return itinerary_service.create_full_itinerary(days, day_plans)

# =============================
# Summary Tool
# =============================


@tool
def generate_summary(all_data):
    """Generate a summary of the trip."""
    return summary_service.generate_summary(all_data)

# =============================
# New ReAct Agent Tools (with Fallbacks)
# =============================

@tool
def search_hotels_amadeus(city: str, checkin_date: str, checkout_date: str, guests: int = 1):
    """Search for hotels using Amadeus API, fallback to Google Places/SerpAPI."""
    try:
        city_search = amadeus.reference_data.locations.get(keyword=city, subType='CITY')
        city_code = city_search.data[0]['iataCode']
        response = amadeus.shopping.hotel_offers.get(cityCode=city_code, checkInDate=checkin_date, checkOutDate=checkout_date, adults=guests)
        hotels = response.data
        if hotels:
            return hotels
    except Exception:
        pass
    # Fallback to Google Places/SerpAPI
    try:
        result = places_tool.run(f"hotels in {city}")
        if result and len(result) > 0:
            return result
    except Exception:
        pass
    return serpapi_wrapper.run(f"hotels in {city}")

@tool
def search_city_airport(query: str):
    """Find city or airport codes using Amadeus API."""
    try:
        response = amadeus.reference_data.locations.get(keyword=query, subType='CITY,AIRPORT')
        return response.data
    except Exception as e:
        return f"Error: {e}"

@tool
def get_activities(city: str, start_date: str, end_date: str):
    """Get tours/activities using Amadeus API, fallback to Google/SerpAPI."""
    try:
        geo = places_tool.run(f"latitude and longitude of {city}")
        if isinstance(geo, dict):
            lat, lng = geo.get("lat"), geo.get("lng")
        else:
            lat, lng = 0, 0
        response = amadeus.shopping.activities.get(latitude=lat, longitude=lng, startDate=start_date, endDate=end_date)
        activities = response.data
        if activities:
            return activities
    except Exception:
        pass
    try:
        result = places_tool.run(f"things to do in {city}")
        if result and len(result) > 0:
            return result
    except Exception:
        pass
    return serpapi_wrapper.run(f"things to do in {city}")

@tool
def search_events(city: str, dates: str = ""):
    """Search for local events using SerpAPI, fallback to Google."""
    try:
        result = serpapi_wrapper.run(f"events in {city} {dates}")
        if result and len(result) > 0:
            return result
    except Exception:
        pass
    try:
        result = places_tool.run(f"events in {city} {dates}")
        if result and len(result) > 0:
            return result
    except Exception:
        pass
    return f"No events found for {city}."

@tool
def get_place_details(place_id: str):
    """Get place details (photos, reviews, info) using SerpAPI, fallback to Google Places."""
    try:
        result = serpapi_wrapper.run(f"place details for {place_id}")
        if result:
            return result
    except Exception:
        pass
    try:
        result = places_tool.run(f"details for place id {place_id}")
        if result:
            return result
    except Exception:
        pass
    return f"No details found for place {place_id}."

# =============================
# Register all tools with the agent
# =============================
# (Add new tools to the tools list)
tools = [
    get_current_weather,
    get_forecast,
    search_attractions,
    search_hotels,
    get_exchange_rate,
    convert_currency,
    search_restaurants,
    search_activities,
    search_transportation,
    estimate_hotel_cost,
    calculate_total_cost,
    calculate_daily_budget,
    create_day_plan,
    create_full_itinerary,
    generate_summary,
    # New tools:
    search_hotels_amadeus,
    search_city_airport,
    get_activities,
    search_events,
    get_place_details,
]

# =============================
# ReAct Agent Initialization (with System Message)
# =============================

system_message = (
    "You are an expert travel planner. For every trip, always provide: "
    "weather, hotel options, attractions, activities, events, transportation, currency info, and a total cost estimate. "
    "Use all available tools to gather this information. Summarize the results in a clear, day-by-day itinerary. "
    "If the user asks for a multi-day trip, provide details for each day."
)

config = Config()
llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    openai_api_key=config.openai_api_key,
    temperature=0
)

# If initialize_agent supports system_message, use it. Otherwise, prepend to conversation.
try:
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        system_message=system_message
    )
except TypeError:
    # Fallback: prepend system message to conversation in the chat UI
    def agent_with_system_message(conversation):
        return agent(f"System: {system_message}\n{conversation}")
    agent = agent_with_system_message


# =============================
# LangGraph Workflow with Fallbacks for New Tools
# =============================
from langgraph.graph import StateGraph, START, END

def weather_node(state):
    city = state.get("city")
    days = state.get("days", 3)
    weather = get_forecast.invoke({"city": city, "days": days})
    state["weather"] = weather
    return state

def attractions_node(state):
    city = state.get("city")
    attractions = search_attractions.invoke({"city": city})
    state["attractions"] = attractions
    return state

def hotels_node(state):
    city = state.get("city")
    checkin = state.get("checkin_date", "2024-07-01")
    checkout = state.get("checkout_date", "2024-07-03")
    guests = state.get("guests", 1)
    hotels = search_hotels_amadeus.invoke({"city": city, "checkin_date": checkin, "checkout_date": checkout, "guests": guests})
    if not hotels or "error" in str(hotels).lower():
        state["use_hotels_fallback"] = True
    else:
        state["hotels"] = hotels
        state["use_hotels_fallback"] = False
    return state

def hotels_fallback_node(state):
    city = state.get("city")
    budget = state.get("budget", "mid-range")
    hotels = search_hotels.invoke({"city": city, "budget": budget})
    state["hotels"] = hotels
    return state

def hotels_edge_condition(state):
    if state.get("use_hotels_fallback", False):
        return "hotels_fallback"
    return "success"

def city_airport_node(state):
    query = state.get("city")
    city_airport = search_city_airport.invoke({"query": query})
    state["city_airport"] = city_airport
    return state

def activities_node(state):
    city = state.get("city")
    start_date = state.get("checkin_date", "2024-07-01")
    end_date = state.get("checkout_date", "2024-07-03")
    activities = get_activities.invoke({"city": city, "start_date": start_date, "end_date": end_date})
    if not activities or "error" in str(activities).lower():
        state["use_activities_fallback"] = True
    else:
        state["activities"] = activities
        state["use_activities_fallback"] = False
    return state

def activities_fallback_node(state):
    city = state.get("city")
    activities = search_activities.invoke({"city": city})
    state["activities"] = activities
    return state

def activities_edge_condition(state):
    if state.get("use_activities_fallback", False):
        return "activities_fallback"
    return "success"

def events_node(state):
    city = state.get("city")
    dates = state.get("checkin_date", "")
    events = search_events.invoke({"city": city, "dates": dates})
    if not events or "No events found" in str(events):
        state["use_events_fallback"] = True
    else:
        state["events"] = events
        state["use_events_fallback"] = False
    return state

def events_fallback_node(state):
    city = state.get("city")
    activities = search_activities.invoke({"city": city})
    state["events"] = activities
    return state

def events_edge_condition(state):
    if state.get("use_events_fallback", False):
        return "events_fallback"
    return "success"

def place_details_node(state):
    place_id = state.get("place_id")
    if place_id:
        details = get_place_details.invoke({"place_id": place_id})
        state["place_details"] = details
    return state

def itinerary_node(state):
    city = state.get("city")
    days = state.get("days", 3)
    attractions = state.get("attractions")
    weather = state.get("weather")
    day_plans = [create_day_plan.invoke({"city": city, "attractions": attractions, "weather": weather}) for _ in range(days)]
    itinerary = create_full_itinerary.invoke({"days": days, "day_plans": day_plans})
    state["itinerary"] = itinerary
    return state

def summary_node(state):
    summary = generate_summary.invoke({"all_data": state})
    state["summary"] = summary
    return state

graph = StateGraph(dict)
graph.add_node("weather", weather_node)
graph.add_node("attractions", attractions_node)
graph.add_node("hotels", hotels_node)
graph.add_node("hotels_fallback", hotels_fallback_node)
graph.add_node("city_airport", city_airport_node)
graph.add_node("activities", activities_node)
graph.add_node("activities_fallback", activities_fallback_node)
graph.add_node("events", events_node)
graph.add_node("events_fallback", events_fallback_node)
graph.add_node("place_details", place_details_node)
graph.add_node("itinerary", itinerary_node)
graph.add_node("summary", summary_node)

graph.add_edge(START, "weather")
graph.add_edge("weather", "attractions")
graph.add_edge("attractions", "hotels")
graph.add_conditional_edges("hotels", {"hotels_fallback": hotels_edge_condition})
graph.add_edge("hotels", "city_airport")
graph.add_edge("hotels_fallback", "city_airport")
graph.add_edge("city_airport", "activities")
graph.add_conditional_edges("activities", {"activities_fallback": activities_edge_condition})
graph.add_edge("activities", "events")
graph.add_edge("activities_fallback", "events")
graph.add_conditional_edges("events", {"events_fallback": events_edge_condition})
graph.add_edge("events", "place_details")
graph.add_edge("events_fallback", "place_details")
graph.add_edge("place_details", "itinerary")
graph.add_edge("itinerary", "summary")
graph.add_edge("summary", END)

print("Graph nodes and their functions:")
for name, func in graph.nodes.items():
    print(f"{name}: {func}")

# Visualize the workflow by saving to a file
try:
    image_bytes = graph.compile().get_graph(xray=True).draw_mermaid_png()
    with open("travel_workflow_graph.png", "wb") as f:
        f.write(image_bytes)
    print("\nWorkflow graph saved to travel_workflow_graph.png")
except Exception as e:
    print(f"\nCould not generate workflow graph image: {e}")


# =============================
# Streamlit Chatbot UI for ReAct Agent
# =============================

# Example prompts
example_prompts = [
    "Plan a 5-day trip to Paris for two people with a $2000 budget.",
    "I want to visit Rome for 4 days with a focus on history, art, and food. My budget is 800 EUR.",
    "What will the weather be like in Tokyo next week? Suggest some hotels.",
    "Are there any local events in Barcelona this weekend?",
    "Create a 3-day itinerary for New York City including museums and top restaurants.",
    "How much is 100 USD in EUR?",
]

st.title("AI Travel Agent Chatbot")
st.markdown("#### Example Prompts:")
cols = st.columns(len(example_prompts))
for i, prompt in enumerate(example_prompts):
    if cols[i].button(prompt, key=f"ex_prompt_{i}"):
        st.session_state.user_input = prompt

# Always render the input box immediately after handling prompt clicks
user_input = st.text_input(
    "Type your message...",
    value=st.session_state.get("user_input", ""),
    key="user_input"
)

if st.button("Send") and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # If the user prompt is a trip planning request, run the workflow graph
    if "plan a trip" in user_input.lower() or "plan" in user_input.lower():
        # Extract city and days from the prompt (very basic extraction for demo)
        import re
        city_match = re.search(r'trip to ([a-zA-Z ]+)', user_input, re.IGNORECASE)
        days_match = re.search(r'(\d+)[- ]?day', user_input, re.IGNORECASE)
        city = city_match.group(1).strip() if city_match else "Paris"
        days = int(days_match.group(1)) if days_match else 3
        state = {"city": city, "days": days}
        # Run the workflow
        compiled_graph = graph.compile()
        result = compiled_graph.invoke(state)
        summary = result.get("summary", "Sorry, could not generate a full summary.")
        st.session_state.chat_history.append({"role": "agent", "content": summary})
    else:
        # Prepare prompt for agent (full conversation as context)
        conversation = "\n".join([
            f"User: {msg['content']}" if msg["role"] == "user" else f"Agent: {msg['content']}"
            for msg in st.session_state.chat_history
        ])
        agent_reply = agent(conversation)
        st.session_state.chat_history.append({"role": "agent", "content": agent_reply})

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Agent:** {msg['content']}")

