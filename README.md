<div align="center">
  <h1>🔬 LangChain Research Agent</h1>
  <p><strong>A Local-First, Autonomous Research Assistant with Human-in-the-Loop (HITL) Capabilities</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)](https://python.langchain.com/)
  [![LangGraph](https://img.shields.io/badge/LangGraph-State%20Machine-orange.svg)](https://python.langchain.com/docs/langgraph)
  [![Ollama](https://img.shields.io/badge/Ollama-Local%20LLMs-black.svg)](https://ollama.com/)
  
  <br />
  <br />
  <img src="https://i.ibb.co/TMt3mq9x/image.png" alt="LangChain Research Agent" width="800">
</div>

<br />

## 🌟 Overview

The **LangChain Research Agent** is a terminal-based AI assistant designed to conduct deep, accurate research autonomously while keeping you in the driver's seat. Powered by **LangChain**, **LangGraph**, and **Ollama**, this agent leverages multiple data sources to fetch contextual, real-time information—all while rigorously adhering to a **Human-in-the-Loop (HITL)** architecture for critical operations like internet searches.

## ✨ Key Features

- **🧑‍💻 Human-In-The-Loop (HITL)**: Take control of the agent's actions. The agent pauses execution before making web searches, requesting your explicit approval, ensuring safety and precision.
- **🛠️ Robust Tool Integration**: Natively integrates with:
  - `DuckDuckGo` (Web Search)
  - `Wikipedia` (Encyclopedic knowledge)
  - `ArXiv` (Academic and scientific research)
  - `Datetime` (Real-time awareness)
- **🧠 Persistent Memory**: Uses SQLite-backed checkpointing to remember conversation context across interactions.
- **🔄 Fault Tolerance**: Built-in middleware for conditional tool retries, model fallbacks, and token reduction (Summarization Middleware).
- **🔒 Local-First AI**: Optimized for execution via local **Ollama** models, maximizing privacy and lowering inference costs.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/langchain-research-agent.git
   cd langchain-research-agent
   ```

2. **Set up the virtual environment & install dependencies:**
   *(This project supports `uv` for lightning-fast package management)*
   ```bash
   uv venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   
   uv pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Copy the example environment file and adjust your settings:
   ```bash
   cp .env.example .env
   ```
   *Make sure to set your preferred `MODEL_NAME` and `MODEL_TEMP` in `.env`.*

### Usage

Launch the agent by running the main script:
```bash
python main.py
```

You'll be greeted by a beautiful CLI dashboard. Start typing your queries and watch the agent navigate tools, pausing whenever your approval is required!

## 💡 How the HITL Feature Works

By utilizing `HumanInTheLoopMiddleware` from LangChain, the agent intercepts requests mapped to critical tools like `search_tool`. When a search tool is invoked, the execution halts, and a prompt is displayed in your terminal:
```text
⚠️  Tool: search_tool
   Args: {'query': 'latest AI news'}
Approve? (y/n): 
```

<div align="center">
  <img src="https://i.ibb.co/svQBNJMb/image.png" alt="HITL Terminal Example" width="450">
</div>

Approving the request resumes execution; rejecting it denies the tool call, giving you unprecedented control over your AI's behavior.

---

<div align="center">
  <p>Crafted with ❤️ by <b>Muhammad Aliyan</b></p>
</div>
