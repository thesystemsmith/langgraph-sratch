# the tool is gonna be web search
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# llm
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama
    
# tool import
from langchain_community.tools.tavily_search import TavilySearchResults

# state
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# tools
tool = TavilySearchResults(max_results=2)
tools = [tool]

# llm with tools
llm = ChatOllama(model="qwen2.5:3b-instruct")  
llm_with_tools = llm.bind_tools(tools)

# node
def chatbot(state: State):
    # Let the LLM decide to call Tavily or not based on the conversation
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# graph wiring
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[tool]))

# After chatbot runs, route to tools if a tool call was requested
graph_builder.add_conditional_edges("chatbot", tools_condition)
# After any tool call, go back to chatbot
graph_builder.add_edge("tools", "chatbot")


# create memory 
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

# compile
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer = memory)

config = {"configurable": {"thread_id": "1"}}

# ----- Simple REPL -----
if __name__ == "__main__":
    print("Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Bye")
            break
        # LangGraph expects tuples like ("user", text)
        for event in graph.stream({"messages": [("user", user_input)]}, config):
            for value in event.values():
                msg = value["messages"][-1]
                # msg is a BaseMessage; print its content
                try:
                    print("Assistant:", msg.content)
                except Exception:
                    print("Assistant:", msg)