import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

# Set API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
# Tool: Google Search via Serper


search_tool = SerperDevTool(api_key=SERPER_API_KEY)
topic = "Forkhead box protein O4 (FOXO4-p53 interaction)"





# Define the research agent
research_agent = Agent(
    role="Technology Researcher",
    goal="Explore emerging trends and innovations in the assigned topic",
    backstory=(
        "You are a tech-savvy researcher passionate about discovering "
        "cutting-edge technologies and their implications on the industry."
    ),
    tools=[search_tool],
    verbose=True,
    llm=LLM(model="deepseek/deepseek-v3.2")
)

# Define the research task
research_task = Task(
    description=(
        f"Use online resources to conduct thorough research on the topic  Be specific, concise and insightful. Insure you satisfy the user's request completely and to the best of your ability. ?"
    ),
    expected_output="A 3-paragraph summary highlighting the most relevant findings. YOU MUST INCLUDE CITATIONS FOR YOUR ANSWER TO BE ACCEPTED",
    tools=[search_tool],
    agent=research_agent
)

# Create the crew with one agent and task
crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    verbose=True,
    process=Process.sequential  # Just one agent, but still needed
)

# Run the crew with a topic input
if __name__ == "__main__":
    result = crew.kickoff(inputs={"topic": topic})
    print(result)
