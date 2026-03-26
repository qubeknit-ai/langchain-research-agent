import os
import sys
import json
from dotenv import load_dotenv
from datetime import datetime
import sqlite3

# LangChain
from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_tool_call,
    ToolRetryMiddleware,
    ToolCallLimitMiddleware,
    ModelRetryMiddleware,
    ModelFallbackMiddleware,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool
from langchain_community.tools import (
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
    ArxivQueryRun
)
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
    ArxivAPIWrapper
)

# LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver # For Production
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

# Ollama Model
from langchain_ollama import ChatOllama

# Load Enviromanetal variables
load_dotenv(".env")

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_TEMP = float(os.getenv("MODEL_TEMP", "0.7"))
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "research_assistant.db")

ddgs_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
search_tool = DuckDuckGoSearchResults(
    api_wrapper=ddgs_wrapper,
    name="search_tool",
    description="Search the internet for real-time information."
)

wiki_wrapper= WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_wrapper,
    name="wiki_tool",
    description="Search Wikipedia for well-established, encyclopedic knowledge."
)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
arxiv_tool = ArxivQueryRun(
    api_wrapper=arxiv_wrapper,
    name="arxiv_tool",
    description= "Search arXiv for peer-reviewed academic papers and preprints."
)

@tool
def get_current_datetime():
    """Get the current date and time."""
    now_datetime = datetime.now()
    return now_datetime.strftime("%Y-%m-%d %H:%M:%S")


# Middleware
@wrap_tool_call
def handle_tool_call_error(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return f"Tool error: {str(e)}"

tool_retry = ToolRetryMiddleware(
    max_retries=2,
    tools=["search_tool"],
    on_failure="continue",
    max_delay=60,
    backoff_factor=1.5
    )

tool_call_midlw = ToolCallLimitMiddleware(
    thread_limit=3,
    run_limit=3,
    exit_behavior="continue",
    tool_name="search_tool"
)

model_retry = ModelRetryMiddleware(
    max_retries=2,
    on_failure="continue",
    max_delay=60,
    backoff_factor=2
)

model_fallback_midlw = ModelFallbackMiddleware(
    "ollama:gemini-3-flash-preview:cloud",
    "ollama:gpt-oss:20b-cloud",
)

summ_midlw = SummarizationMiddleware(
    model="ollama:minimax-m2.5:cloud",
    trigger=("tokens", 4000),
    keep=("messages", 20),
)

hitl_midlw = HumanInTheLoopMiddleware(
    interrupt_on={
        "search_tool": True,   # All decisions (approve, edit, reject) allowed
        "wiki_tool": False,    # Safe operation, no approval needed
        "arxiv_tool": False,   # Safe operation, no approval needed
    },
    description_prefix="Tool execution pending approval",
)


# Tools
custom_tools = [search_tool, wiki_tool, arxiv_tool, get_current_datetime]

# System Prompt
custom_system_prompt = f"""
    You are a precise and thorough research assistant. Your job is to investigate topics deeply and return well-structured, accurate answers grounded in real sources.

    ## IMPORTANT:
    Today is {datetime.today()}
    Now is {datetime.now()}
    
    ## Your Tools
    - **web_search** — use for current events, news, and anything time-sensitive
    - **wikipedia** — use for definitions, background context, and established facts
    - **arxiv** — use for academic papers, technical research, and scientific claims
    - **get_current_datetime** — use when the user asks about the current time or date

    ## How You Work
    1. Analyze the user's question and identify what kind of information is needed.
    2. Choose the right tool(s) — you may call multiple tools if needed.
    3. Synthesize results into a clear, structured response.
    4. Always cite where information came from (web, Wikipedia, paper title, etc.).
    5. If a tool returns an error or empty result, try rephrasing the query or use a different tool.

    ## Rules
    - Never fabricate facts, citations, or paper titles.
    - If you don't know something and can't find it via tools, say so clearly.
    - Keep responses focused — don't pad with filler.
    - For technical topics, prefer arxiv over web_search.
    - For recent events (< 1 year), always use web_search.
"""

# Agent Setup
def run_research_agent():
    
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=MODEL_TEMP
    )

    # memory = MemorySaver()
    db_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    memory = SqliteSaver(conn=db_conn)

    tool_middleware = [
        handle_tool_call_error,
        tool_retry,
        tool_call_midlw,
        model_retry,
        model_fallback_midlw,
        summ_midlw,
        hitl_midlw
    ]

    # Agent
    agent = create_agent(
        model=llm,
        tools=custom_tools,
        system_prompt= custom_system_prompt,
        middleware= tool_middleware,
        checkpointer=memory,
        name= "Research Assistant"
    )

    return agent

def banner():
    """Display the CLI banner on startup."""
    banner_art = """
        \033[36m
        ██████╗ ███████╗███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗
        ██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║
        ██████╔╝█████╗  ███████╗█████╗  ███████║██████╔╝██║     ███████║
        ██╔══██╗██╔══╝  ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║
        ██║  ██║███████╗███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║
        ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
        \033[0m\033[90m
         █████╗ ███████╗███████╗██╗███████╗████████╗ █████╗ ███╗   ██╗████████╗
        ██╔══██╗██╔════╝██╔════╝██║██╔════╝╚══██╔══╝██╔══██╗████╗  ██║╚══██╔══╝
        ███████║███████╗███████╗██║███████╗   ██║   ███████║██╔██╗ ██║   ██║   
        ██╔══██║╚════██║╚════██║██║╚════██║   ██║   ██╔══██║██║╚██╗██║   ██║   
        ██║  ██║███████║███████║██║███████║   ██║   ██║  ██║██║ ╚████║   ██║   
        ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   
        \033[0m"""

    print(banner_art)

    print("\033[90m" + "─" * 72 + "\033[0m")
    print(f"  \033[36m🔬 Model   :\033[0m \033[97m{MODEL_NAME}\033[0m")
    print(f"  \033[36m🌡  Temp    :\033[0m \033[97m{MODEL_TEMP}\033[0m")
    print(f"  \033[36m📅 Date    :\033[0m \033[97m{datetime.now().strftime('%A, %B %d %Y  %H:%M')}\033[0m")
    print(f"  \033[36m🛠  Tools   :\033[0m \033[97mweb_search · wikipedia · arxiv · datetime\033[0m")
    print(f"  \033[36m💾 Memory  :\033[0m \033[97mSqlite3 (SqliteSaver)\033[0m")
    print("\033[90m" + "─" * 72 + "\033[0m")
    print(f"  \033[90mType your question below. Commands: \033[0m\033[33mexit · quit · q\033[0m")
    print("\033[90m" + "─" * 72 + "\033[0m\n")



def stream_response(agent, query, config: dict):
    """Stream agent response with human-in-the-loop support."""
    input_data = {"messages": [HumanMessage(content=query)]}
    seen_tool_calls = set()

    while True:
        interrupted = False

        for chunk in agent.stream(
            input_data,
            config=config,
            stream_mode="values"
        ):
            # Check for HITL interrupts
            if "__interrupt__" in chunk:
                interrupted = True
                interrupt_info = chunk["__interrupt__"][0]
                payload = interrupt_info.value
                action_requests = payload.get("action_requests", [])

                # Show each pending tool call
                for req in action_requests:
                    print(f"\n\033[33m⚠️  Tool: {req['name']}\033[0m")
                    print(f"\033[33m   Args: {req['args']}\033[0m")

                decision = input("\033[36mApprove? (y/n): \033[0m").strip().lower()

                # Build one decision per action_request
                if decision in ("y", "yes"):
                    decisions = [{"type": "approve"} for _ in action_requests]
                else:
                    decisions = [{"type": "reject"} for _ in action_requests]

                input_data = Command(resume={"decisions": decisions})
                break

            # Normal message handling
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                if isinstance(latest_message, HumanMessage):
                    pass
                elif isinstance(latest_message, AIMessage):
                    print(f"\nAgent: {latest_message.content}")
            elif latest_message.tool_calls:
                # Deduplicate tool call displays after interrupt resume
                call_ids = tuple(tc.get("id", tc["name"]) for tc in latest_message.tool_calls)
                if call_ids not in seen_tool_calls:
                    seen_tool_calls.add(call_ids)
                    print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")

        if not interrupted:
            break



def main():
    banner()

    agent = run_research_agent()
    config={"configurable": {"thread_id": "my_bucket"}}

    while True:
        try:
            query = input("\nYou: ").strip()
        except KeyboardInterrupt as err:
            print("\nGoodBye!")
            sys.exit(0)
            
        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodBye! Happy Researching!")
            sys.exit(0) 

        try:
            stream_response(agent, query, config)
        except Exception as err:
            print(f'Error: {err}')


if __name__ == "__main__":
    main()
