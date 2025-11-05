# 1 necessary imports
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 2 define the state structure
class State(TypedDict):
    # 'messages' will store the chatbot conversation history.
    # The 'add_messages' function ensures new messages are appended to the list.
    messages: Annotated[list, add_messages]
    
# 3 Create an instance of the StateGraph, passing in the State class
graph_builder = StateGraph(State)


# 4 initialize the llm
from langchain_community.chat_models import ChatOllama
llm = ChatOllama(model="phi3:mini")  # or "mistral", "llama3"

# 5 create a chatbot node
def chatbot(state: State):
    # Use the LLM to generate a response based on the current conversation history.
    response = llm.invoke(state["messages"])
    # Return the updated state with the new message appended
    return {"messages": [response]}
# 6 Add the 'chatbot' node to the graph,
graph_builder.add_node("chatbot", chatbot)

# 7 define start and end of the graph
# For this basic chatbot, the 'chatbot' node is both the entry and finish point
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 8 compile the graph
graph = graph_builder.compile()

# 9 visualize the graph
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# 10 run the chatbot
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    # Process user input through the LangGraph
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)