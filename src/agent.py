from typing import List, Optional
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage, UsageLimits
from .functions import fetch_articles, fetch_clinical_trails, fetch_medline_plus

# Update logfire configuration to track everything
logfire.configure(send_to_logfire="if-token-present")


class Article(BaseModel):
    """Structure for article data"""

    title: str
    abstract: str
    authors: Optional[List[str]] = None
    url: str 
    primary_outcomes: Optional[List[dict]] = None  # Make primary_outcomes optional with None as default


class SearchResponse(BaseModel):
    """Agent's response with citations"""

    answer: str = Field(
        description="Detailed Verbose answer to the user's query with clickable numbered citations like [1], [2]"
    )
    citations: List[str] = Field(description="List of citations in format: '[number] Title - URL'")


search_agent = Agent[None, SearchResponse](
    "google-gla:gemini-2.0-flash-exp",
    # result_type=SearchResponse,
    result_type = str,
    retries=4,  # Add retries for reliability
    system_prompt=(
        """You are a medical research assistant. Your job is to answer only medical questions based on evidence. 
        You are provided with 3 tools: PubMed, ClinicalTrials.gov, and Medline Plus. You can use all tools at once parallelly.
        You should not use your knowledge or experience to answer the questions. Answer only based on evidence from tools.
        *You have to call tools for all medical questions.* Deny answering non-medical questions.
        Based on evidence provided, you can call tools again to get more information.
        Use numbered citations in your answer like [1], [2] etc at the end of relevant sentences. 
        In the citations section, format each citation as: "[number] Title - URL". 
        Example format:\n
        Answer: This is a finding [1]. Another finding [2].\n"
        Citations:\n"
        [1] First Article Title - http://url1\n"
        [2] Second Article Title - http://url2"
        Examples: Query: 'What is the treatment for diabetes?' keyword: 'diabetes treatment'
        Query: 'what causes hairfall?' keyword: 'hair fall'
        Be as detailed and verbose as possible in your answer by considering all evidences from tools. 
        Provide citations to support your answer."""
    ),
)


@search_agent.tool
async def search_pubmed(ctx: RunContext[None], keyword: str) -> List[Article]:
    """Search PubMed for articles related to the keyword"""
    articles = fetch_articles(keyword)
    logfire.info("found {article_count} articles from pubmed", article_count=len(articles))
    return [Article(**article) for article in articles]

@search_agent.tool
async def search_clinical_trials(ctx: RunContext[None], keyword: str) -> List[Article]:
    """Search ClinicalTrials.gov for articles related to the keyword"""
    articles = fetch_clinical_trails(keyword)
    logfire.info("found {article_count} articles from clinical trails", article_count=len(articles))
    return [Article(**article) for article in articles]

@search_agent.tool
async def search_medline_plus(ctx: RunContext[None], keyword: str) -> List[Article]:
    """Search Medline Plus for articles related to the keyword"""
    articles = fetch_medline_plus(keyword)
    logfire.info("found {article_count} articles from medline plus", article_count=len(articles))
    return [Article(**article) for article in articles]


# @search_agent.result_validator
# async def validate_result(ctx: RunContext[None], result: SearchResponse) -> SearchResponse:
#     """Validate that the response includes citations"""
#     if not result.citations:
#         raise ModelRetry("Response must include at least one citation")
#     return result


async def main():
    message_history: list[ModelMessage] | None = None

    while True:
        query = input("\nEnter your medical research question (or 'quit' to exit): ")
        if query.lower() == "quit":
            break

        print("\nGenerating answer...")
        result = await search_agent.run(query, message_history=message_history)

        print("\nAnswer:")
        print(result.data)
        # print(result.data.answer)
        # if result.data.citations:
        #     print("\nCitations:")
        #     for citation in result.data.citations:
        #         print(f"- {citation}")
        
        message_history = result.all_messages()

        # Print token usage for this response
        print("\nToken Usage:")
        usage_info = result.usage()
        print(f"Request tokens: {usage_info.request_tokens}")
        print(f"Response tokens: {usage_info.response_tokens}")
        print(f"Total tokens: {usage_info.total_tokens}")

        

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
