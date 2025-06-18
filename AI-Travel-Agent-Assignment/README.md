# 🌍 AI Travel Agent & Expense Planner

A comprehensive multi-agent travel planning system built with LangGraph and LangChain that coordinates specialized agents for weather, hotels, attractions, currency, itinerary planning, and expense calculation.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/srini118us/AgenticAI)
[![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-orange?style=for-the-badge)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-purple?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)

## 📋 Table of Contents

- [🏗️ Project Structure](#️-project-structure)
- [🚀 Quick Start](#-quick-start)
- [🤖 Agent Overview](#-agent-overview)
- [🔄 Workflow Diagram](#-workflow-diagram)
- [📊 Output Example](#-output-example)
- [🛠️ Technologies](#️-technologies)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)

## 🏗️ Project Structure

```
AI-Travel-Agent-Assignment/
├── agents/                 # Specialized AI agents
│   ├── weather_agent.py    # Weather information and forecasts
│   ├── attractions_agent.py # Tourist attractions and activities
│   ├── hotel_agent.py      # Hotel search and accommodation
│   ├── currency_agent.py   # Currency conversion and exchange rates
│   ├── itinerary_agent.py  # Day-by-day trip planning
│   ├── expense_agent.py    # Cost calculation and budgeting
│   ├── summary_agent.py    # Trip summary generation
│   ├── online_hotel_agent.py # Online booking platform search
│   ├── wikipedia_agent.py  # Wikipedia information retrieval
│   ├── search_agent.py     # Web search for attractions
│   ├── supervisor_agent.py # Workflow orchestration
│   ├── cost_agent.py       # Cost analysis and recommendations
│   └── __init__.py
├── tools/                  # Tool wrappers and utilities
│   ├── weather_tools.py    # Weather API tools
│   ├── hotel_tools.py      # Hotel search tools
│   ├── cost_tools.py       # Cost calculation tools
│   ├── currency_tools.py   # Currency conversion tools
│   └── __init__.py
├── models/                 # Data models and schemas
│   ├── weather_models.py   # Weather data structures
│   ├── hotel_models.py     # Hotel and accommodation models
│   ├── itinerary_models.py # Itinerary planning models
│   ├── travel_models.py    # General travel data models
│   └── __init__.py
├── workflows/              # Workflow orchestration
│   ├── travel_workflow.py  # Main LangGraph workflow
│   └── __init__.py
├── config/                 # Configuration and settings
│   ├── settings.py         # Environment and API settings
│   └── __init__.py
├── venv/                   # Python virtual environment
├── demo.py                 # Main demo script
├── main.py                 # Alternative entry point
├── travel_workflow_graph.py # Workflow diagram generator
├── CODEBASE_GUIDE.md       # Complete codebase analysis
├── requirements.txt        # Python dependencies
├── env_template.txt        # Environment variables template
└── README.md              # This file
```

## 🚀 Quick Start

### 1. **Clone Repository**
```bash
git clone https://github.com/srini118us/AgenticAI.git
cd AgenticAI/AI-Travel-Agent-Assignment
```

### 2. **Setup Environment**
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. **Configure API Keys**
```bash
# Copy environment template
copy env_template.txt .env

# Edit .env with your API keys
# - OPENAI_API_KEY
# - OPENWEATHER_API_KEY  
# - AMADEUS_CLIENT_ID
# - AMADEUS_CLIENT_SECRET
```

### 4. **Run Demo**
```bash
python demo.py
```

### 5. **Generate Workflow Diagram** (Optional)
```bash
python travel_workflow_graph.py
```

## 🤖 Agent Overview

### 🌟 Core Agents
- **🌤️ Weather Agent**: Real-time weather data and forecasts
- **🏨 Hotel Agent**: Accommodation search with budget options
- **💱 Currency Agent**: Exchange rates and currency advice
- **📅 Itinerary Agent**: Day-by-day trip planning
- **💰 Expense Agent**: Cost breakdown and budgeting
- **📋 Summary Agent**: Comprehensive trip summary

### 🔄 Fallback Agents
- **🌐 Online Hotel Agent**: Alternative hotel booking platforms
- **📚 Wikipedia Agent**: Tourist information from Wikipedia
- **🔍 Search Agent**: Web search for attractions and activities

## 🔄 Workflow Diagram



### 🔄 Detailed Workflow Flow

1. **🌤️ Weather Analysis**: Get current weather and forecasts
2. **🎯 Attractions Search**: Find tourist attractions and activities
3. **🏨 Hotel Search**: Primary hotel search with budget options
4. **🌐 Fallback Hotel**: Alternative booking platforms if primary fails
5. **💱 Currency Conversion**: Exchange rates and financial advice
6. **📅 Itinerary Planning**: Day-by-day trip schedule
7. **💰 Expense Calculation**: Complete cost breakdown
8. **📋 Summary Generation**: Comprehensive trip summary

## 📊 Output Example

```
🌍 TRIP TO LONDON
==================================================
🌤️  WEATHER: scattered clouds (High: 16.33°C, Low: 14.27°C)
   💡 Pack accordingly!
🏨 HOTEL: Budget Inn London
   💰 $60/night | ⭐ 3.5/5
💱 CURRENCY: Exchange rate: 1 USD = 0.7420 GBP
💰 EXPENSES: Total $570
   🏨 Hotel: $360 | 🍽️ Food: $120 | 🎫 Attractions: $90
📅 ITINERARY (3 days):
   Day 1: Check-in, Explore city center
   Day 2: Visit main attractions, Try local food
   Day 3: Shopping, Check-out
```

## 🛠️ Technologies

### 🏗️ Core Framework
- **[LangGraph](https://langchain-ai.github.io/langgraph/)**: Workflow orchestration
- **[LangChain](https://langchain.com/)**: Agent framework and tools
- **[OpenAI](https://openai.com/)**: LLM for agent reasoning
- **[Pydantic](https://docs.pydantic.dev/)**: Data validation and models

### 🌐 External APIs
- **[OpenWeatherMap](https://openweathermap.org/)**: Weather data
- **[Exchange Rate API](https://exchangerate-api.com/)**: Currency conversion
- **[Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)**: Tourist information

### 🎨 Visualization
- **[Graphviz](https://graphviz.org/)**: Workflow diagram generation
- **[Mermaid](https://mermaid.js.org/)**: Documentation diagrams

## 📚 Documentation

### 📖 Complete Codebase Analysis
For a detailed breakdown of all classes, functions, and their relationships, see:
- **[CODEBASE_GUIDE.md](CODEBASE_GUIDE.md)** - Comprehensive codebase analysis

### 🔍 Key Features
- **Multi-Agent Architecture**: 12 specialized agents working together
- **State-Driven Workflow**: Shared state management across agents
- **Fallback Mechanisms**: Graceful degradation when APIs fail
- **Real API Integration**: Weather, currency, and hotel data
- **Mock Data Support**: Development and demo capabilities
- **Type Safety**: Pydantic models throughout
- **Error Handling**: Comprehensive error management

### 🎯 Agent Capabilities
- **Weather Forecasting**: Current conditions and trip-period forecasts
- **Hotel Search**: Budget, moderate, and luxury options
- **Currency Conversion**: Real-time exchange rates and advice
- **Attraction Discovery**: Tourist spots and activities
- **Itinerary Planning**: Day-by-day trip scheduling
- **Cost Analysis**: Complete expense breakdown
- **Trip Summaries**: Comprehensive travel plans

## 🤝 Contributing

### 🚀 Getting Started
1. Fork the repository: [https://github.com/srini118us/AgenticAI](https://github.com/srini118us/AgenticAI)
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### 🛠️ Development Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/AgenticAI.git
cd AgenticAI/AI-Travel-Agent-Assignment

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

### 📝 Code Style
- Follow PEP 8 guidelines
- Use type hints throughout
- Add docstrings to all functions and classes
- Use Pydantic models for data validation

## 📄 License

This project is part of the [AgenticAI](https://github.com/srini118us/AgenticAI) repository.

## 🙏 Acknowledgments

- **LangChain Team**: For the excellent agent framework
- **OpenAI**: For powerful LLM capabilities
- **OpenWeatherMap**: For reliable weather data
- **Exchange Rate API**: For currency conversion services

---

**⭐ Star this repository if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/srini118us/AgenticAI?style=social)](https://github.com/srini118us/AgenticAI)
[![GitHub forks](https://img.shields.io/github/forks/srini118us/AgenticAI?style=social)](https://github.com/srini118us/AgenticAI)
[![GitHub issues](https://img.shields.io/github/issues/srini118us/AgenticAI)](https://github.com/srini118us/AgenticAI/issues)

![AI Travel Agent Workflow](docs/ai_travel_workflow.png) 