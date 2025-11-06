from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# import llm
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama
    
# import tool
from langchain_tavily import TavilySearch

# state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    
graph_builder = StateGraph(State)

# tools
tool = TavilySearch(max_results = 2)
tools = [tool]

# llm with tools
llm = ChatOllama(model = 'qwen2.5:3b-instruct')
llm_with_tools = llm.bind_tools(tools)

# graph node
def chatbot(state: State):
    return {'messages': [llm_with_tools.invoke(state['messages'])]}

# graph wiring
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

# memory plus interrupt
memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer = memory,
    interrupt_before = ['tools']
)

# ----- Example REPL with a fixed thread (memory) -----
if __name__ == "__main__":
    print("Type 'exit' to quit.")
    config = {"configurable": {"thread_id": "1"}}  # keep same thread to retain memory
    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Bye")
            break

        # Stream events (will interrupt before tools if triggered)
        # Stream events (may interrupt before tools)
        for event in graph.stream({"messages": [("user", user_input)]}, config):
            
            for node_name, value in event.items():
                msg = value["messages"][-1]

                # If the assistant proposed a tool call, you'll see it on msg.tool_calls
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    print("\nProposed tool call(s):")
                    for tc in tool_calls:
                        print(f"- tool: {tc.get('name')}  args: {tc.get('args')}")
                    # Ask user to approve
                    choice = input("Run the tool? (y/n): ").strip().lower()
                    if choice.startswith("y"):
                        # Continue from the interrupt to actually run the tool
                        for ev2 in graph.stream(None, config):
                            for v2 in ev2.values():
                                m2 = v2["messages"][-1]
                                if hasattr(m2, "content"):
                                    print("Assistant:", m2.content)
                    else:
                        print("Skipped tool.")
                        # Optionally clear the proposed tool use by sending a follow-up message:
                        # for ev3 in graph.stream({"messages":[("user","Don't use tools. Answer directly.")]} , config):
                        #     ...
                    # IMPORTANT: break out so we don't double-print
                    break
                else:
                    # Normal assistant response (no tool call)
                    if hasattr(msg, "content"):
                        print("Assistant:", msg.content)