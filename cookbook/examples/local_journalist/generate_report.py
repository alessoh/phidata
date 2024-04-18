from typing import List
from textwrap import dedent
from rich.pretty import pprint

from pydantic import BaseModel, Field
from phi.llm.ollama import OllamaTools
from phi.assistant.team import Assistant
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k

topic = "Llama 3 release by Meta AI"
print(f"-*- Writing a report on: '{topic}' -*-")

#######################################################
# Search the web for a topic
#######################################################
searcher = Assistant(
    llm=OllamaTools(model="llama3:70b"),
    description=dedent("""\
    You are a world-class journalist for the New York Times.
    Given a topic, search the web for 10 most relevant articles on the topic.
    Your goal is to return the top 3 most relevant articles.
    Provide a title, summary, and links for each article.
    Remember: you are writing for the New York Times, so the quality of the sources is important.
    """),
    tools=[DuckDuckGo()],
    add_datetime_to_instructions=True,
    debug_mode=True,
)

web_search_results = f"## Topic: {topic}\n\n"
web_search_results += "## Search Results\n\n"

search_results = searcher.run(topic, stream=False)  # type: ignore
web_search_results += f"{search_results}\n\n"
print(f"============= Search Results for {topic} =============")
print(web_search_results)
print("===========================================================")

########################################################
# Read each article and write a report
########################################################
writer = Assistant(
    llm=OllamaTools(model="llama3:70b"),
    description=dedent("""\
    You are a senior writer for the New York Times. Given a topic and relevant content from the web,
    your goal is to write a high-quality NYT-worthy article on the topic.
    """),
    instructions=[
        "Given a topic and a list of URLs, first read the article using `get_article_text`."
        "Then write a high-quality NYT-worthy article on the topic."
        "The article should be well-structured, informative, and engaging",
        "Remember: you are writing for the New York Times, so the quality of the article is important.",
        "Never make up facts or plagiarize. Always provide proper attribution.",
    ],
    tools=[Newspaper4k()],
    # debug_mode=True,
)

draft = writer.run(web_search_results, stream=False)  # type: ignore
print(f"============= Draft article for {topic} =============")
print(draft)
print("===========================================================")

# ########################################################
# # Write a high-quality article
# ########################################################
#
# report_input = ""
# report_input += f"# Topic: {topic}\n\n"
# report_input += "## Search Terms\n\n"
# report_input += f"{search_terms}\n\n"
# if web_content:
#     report_input += "## Web Content\n\n"
#     report_input += "<web_content>\n\n"
#     report_input += f"{web_content}\n\n"
#     report_input += "</web_content>\n\n"
#
# writer = Assistant(
#     llm=OllamaTools(model="llama3"),
#     description=dedent("""\
#     You are a senior writer for the New York Times. Given a topic and a content from the web,
#     your goal is to write a high-quality NYT-worthy article on the topic.
#     """),
#     instructions=[
#         "You will be provided with a topic and relevant content from the web.",
#         "You must write a high-quality NYT-worthy article on the topic."
#         "The article should be well-structured, informative, and engaging",
#         "Remember: you are writing for the New York Times, so the quality of the article is important.",
#         "Never make up facts or plagiarize. Always provide proper attribution.",
#     ],
#     add_datetime_to_instructions=True,
# )
# writer.print_response(report_input, show_message=False)
