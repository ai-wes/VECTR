import os
from typing import Annotated
from serpapi import GoogleSearch
import autogen
from dotenv import load_dotenv

load_dotenv()

# 1. Configuration
# Ensure you have OPENAI_API_KEY and SERPAPI_API_KEY in your .env or environment variables
llm_config = {
    "config_list": [{"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}],
    "temperature": 0.8,
}

# 2. Define the Search Tool (Swapped for SerpAPI)
def search_web(query: Annotated[str, "The search query string"]) -> Annotated[str, "The search results"]:
    """
    Performs a Google search using SerpAPI and returns the top results.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY not found in environment variables."

    try:
        search = GoogleSearch({
            "q": query,
            "api_key": api_key,
            "num": 20,      # Number of results to return
            "hl": "en",    # Language
            "gl": "us",    # Country
        })
        
        results = search.get_dict()
        organic_results = results.get("organic_results", [])

        if not organic_results:
            return "No results found."

        # Format results for the agent
        formatted_results = []
        for r in organic_results:
            title = r.get('title', 'No Title')
            link = r.get('link', 'No Link')
            snippet = r.get('snippet', 'No Snippet')
            formatted_results.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")

        return "\n---\n".join(formatted_results)

    except Exception as e:
        return f"Error performing search: {str(e)}"

# 3. Create the Agents
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    # FIXED: Added safe check for content before calling rstrip()
    is_termination_msg=lambda x: x.get("content") is not None and x.get("content").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
)

# The AssistantAgent
researcher = autogen.AssistantAgent(
    name="Researcher",
    system_message="""You are a helpful research assistant. 
    You have access to a 'search_web' tool. 
    1. Search for information to answer the user's request.
    2. Cite your sources using the URLs provided in the search results.
    3. When you have found the answer, summarize it concisely and end with 'TERMINATE'. 
    It is IMPERATIVE that you include specific citations in your response.""",
    llm_config=llm_config,
)

# 4. Register the Tool
autogen.register_function(
    search_web,
    caller=researcher,
    executor=user_proxy,
    name="search_web",
    description="A Google search engine. Use this to find information on the internet.",
)

# 5. Initiate the Research Task
task = "Find the specific IC50 validation values for the generated small molecules reported in the 'AlphaFold 3' (or latest 2025 equivalent) supplementary materials for the HER2 target. Then, identify the drug target and the specific drug that was used to validate the IC50 values. Insure that you cite your sources in your response and call out the specific page numbers of the supplementary materials in your response. THe user is extremely specific and will not accept any other answer than the one you find. It must be of the highest quality no matter what"

print(f"Starting research on: {task}")
user_proxy.initiate_chat(
    researcher,
    message=task
)
