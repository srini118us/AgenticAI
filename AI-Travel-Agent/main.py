# =============================
# Imports and Environment Setup
# =============================
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
load_dotenv()

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
        Generate a detailed summary of the trip, including all key details.
        """
        summary_lines = []
        summary_lines.append(f"Trip Summary for {all_data.get('city', 'Unknown')} ({all_data.get('days', '?')} days)")
        summary_lines.append("-" * 40)
        if 'weather' in all_data:
            summary_lines.append(f"Weather Forecast: {all_data['weather']}")
        if 'attractions' in all_data:
            summary_lines.append(f"Top Attractions: {all_data['attractions']}")
        if 'hotels' in all_data:
            summary_lines.append(f"Hotel Suggestions: {all_data['hotels']}")
        if 'itinerary' in all_data:
            summary_lines.append("Itinerary:")
            for i, day in enumerate(all_data['itinerary'], 1):
                summary_lines.append(f"  Day {i}: {day}")
        if 'total_cost' in all_data:
            summary_lines.append(f"Estimated Total Cost: {all_data['total_cost']}")
        if 'daily_budget' in all_data:
            summary_lines.append(f"Estimated Daily Budget: {all_data['daily_budget']}")
        # Fallback info (if present)
        if all_data.get('attractions_fallback_used'):
            summary_lines.append("Attractions fallback used: Yes")
        if all_data.get('hotels_fallback_used'):
            summary_lines.append("Hotels fallback used: Yes")
        return "\n".join(summary_lines)


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
# Bind all tools to the LLM
# =============================
tools = [
    get_current_weather, get_forecast,
    search_attractions, search_restaurants, search_activities, search_transportation,
    search_hotels, estimate_hotel_cost,
    get_exchange_rate, convert_currency,
    calculate_total_cost, calculate_daily_budget,
    create_day_plan, create_full_itinerary,
    generate_summary
]
# llm_with_tools = llm.bind_tools(tools)

# =============================
# Workflow Node Functions
# These functions are used to build the workflow graph for orchestrating the agent's logic.
# =============================


def weather_node(state):
    print("[weather_node] state:", state)
    city = state.get("city")
    days = state.get("days", 3)
    # Use .invoke() and dict input for tool call
    weather = get_forecast.invoke({"city": city, "days": days})
    state["weather"] = weather
    return state


def attractions_node(state):
    print("[attractions_node] state:", state)
    city = state.get("city")
    try:
        # Use .invoke() and dict input for tool call
        attractions = search_attractions.invoke({"city": city})
        if not attractions or "error" in str(attractions).lower():
            state["use_fallback"] = True
        else:
            state["attractions"] = attractions
            state["use_fallback"] = False
    except Exception:
        state["use_fallback"] = True
    return state


def attractions_fallback_node(state):
    print("\n--- ENTERING ATTRACTIONS FALLBACK NODE ---")
    city = state.get("city")
    attractions = serpapi_wrapper.run(f"top attractions in {city}")
    state["attractions"] = attractions
    return state


def attractions_edge_condition(state):
    # return "fallback" if you want to go to fallback, else "success"
    if state.get("use_fallback", False):
        return "fallback"
    return "success"


def hotels_node(state):
    print("[hotels_node] state:", state)
    city = state.get("city")
    budget = state.get("budget", "mid-range")
    # Use .invoke() and dict input for tool call
    hotels = search_hotels.invoke({"city": city, "budget": budget})
    state["hotels"] = hotels
    return state


def expenses_node(state):
    print("[expenses_node] state:", state)
    days = state.get("days", 3)
    hotel_cost = 100 * days
    attraction_cost = 50 * days
    food_cost = 30 * days
    transport_cost = 20 * days
    misc_cost = 10 * days
    # Use .invoke() and dict input for tool call
    total = calculate_total_cost.invoke({
        "hotel_cost": hotel_cost,
        "attraction_cost": attraction_cost,
        "food_cost": food_cost,
        "transport_cost": transport_cost,
        "misc_cost": misc_cost
    })
    state["total_cost"] = total
    return state


def currency_node(state):
    print("[currency_node] state:", state)
    total_cost = state.get("total_cost", 0)
    from_currency = "USD"
    to_currency = state.get("currency", "EUR")
    # Use .invoke() and dict input for tool call
    converted = convert_currency.invoke({
        "amount": total_cost,
        "from_currency": from_currency,
        "to_currency": to_currency
    })
    state["converted_cost"] = converted
    return state


def itinerary_node(state):
    print("[itinerary_node] state:", state)
    city = state.get("city")
    days = state.get("days", 3)
    attractions = state.get("attractions")
    weather = state.get("weather")
    # Use .invoke() and dict input for tool call
    day_plans = [create_day_plan.invoke(
        {"city": city, "attractions": attractions, "weather": weather}) for _ in range(days)]
    itinerary = create_full_itinerary.invoke(
        {"days": days, "day_plans": day_plans})
    state["itinerary"] = itinerary
    return state


def summary_node(state):
    print("[summary_node] REACHED! state:", state)
    # Use .invoke() and dict input for tool call
    summary = generate_summary.invoke({"all_data": state})
    state["summary"] = summary
    return state

# =============================
# Build and connect the workflow graph as you did above
# =============================


graph = StateGraph(dict)
graph.add_node("weather", weather_node)
graph.add_node("attractions", attractions_node)
graph.add_node("attractions_fallback", attractions_fallback_node)
graph.add_node("hotels", hotels_node)
graph.add_node("expenses", expenses_node)
graph.add_node("currency", currency_node)
graph.add_node("itinerary", itinerary_node)
graph.add_node("summary", summary_node)


graph.add_edge(START, "weather")
graph.add_edge("weather", "attractions")
graph.add_conditional_edges(
    "attractions",
    {
        "attractions_fallback": attractions_edge_condition,
    }
)
graph.add_edge("attractions", "hotels")
graph.add_edge("attractions_fallback", "hotels")  # <-- ADD THIS LINE HERE
graph.add_edge("hotels", "expenses")
graph.add_edge("expenses", "currency")
graph.add_edge("currency", "itinerary")
graph.add_edge("itinerary", "summary")
graph.add_edge("summary", END)

print("Graph nodes and their functions:")
for name, func in graph.nodes.items():
    print(f"{name}: {func}")
# =============================
# Compile the workflow
# =============================
initial_state = {"city": "Paris", "days": 3,
                 "budget": "mid-range", "currency": "EUR"}
print("Initial state type:", type(initial_state), initial_state)
workflow = graph.compile()

# =============================
# Visualize the workflow by saving to a file
# =============================
try:
    image_bytes = workflow.get_graph(xray=True).draw_mermaid_png()
    with open("travel_workflow_graph.png", "wb") as f:
        f.write(image_bytes)
    print("\nWorkflow graph saved to travel_workflow_graph.png")
except Exception as e:
    print(f"\nCould not generate workflow graph image: {e}")

# =============================
# Run the workflow
# =============================
# initial_state = MessagesState({"city": "Paris", "days": 3, "budget": "mid-range", "currency": "EUR"})
result = workflow.invoke(initial_state)
print("=== Final State ===")
print(result)
print("=== Trip Summary ===")
print(result.get("summary"))

# =============================
# Streamlit App (UI Section)
# =============================
import streamlit as st

st.title("AI Travel Agent")

city = st.text_input("Destination City", "Paris")
days = st.number_input("Number of Days", min_value=1, value=3)
budget = st.number_input("Budget (USD)", min_value=100, value=1000)
currency = st.text_input("Currency", "USD")

if st.button("Plan My Trip"):
    # Main trip planning logic using TravelAgent class
    config = Config()
    agent = TravelAgent(config)
    summary = agent.plan_trip(city, days, budget, currency)
    st.markdown("## Your Travel Summary")
    st.write(summary)

# =============================
# AttractionService and HotelService
# These classes wrap GooglePlacesTool and SerpAPIWrapper for attractions and hotels.
# =============================

class AttractionService:
    def __init__(self, google_places_api_key, serpapi_key):
        self.places_tool = GooglePlacesTool(gplaces_api_key=google_places_api_key)
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
        self.places_tool = GooglePlacesTool(gplaces_api_key=google_places_api_key)
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
