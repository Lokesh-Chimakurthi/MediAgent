from typing import List
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage, UsageLimits
from .functions import fetch_articles

# Update logfire configuration to track everything
logfire.configure(send_to_logfire="if-token-present")


class Article(BaseModel):
    """Structure for article data"""

    title: str
    abstract: str
    authors: List[str]
    url: str


class SearchResponse(BaseModel):
    """Agent's response with citations"""

    answer: str = Field(
        description="Detailed answer to the user's query with clickable numbered citations like [1], [2]"
    )
    citations: List[str] = Field(description="List of citations in format: '[number] Title - URL'")


search_agent = Agent[None, SearchResponse](
    "google-gla:gemini-2.0-flash-exp",
    result_type=SearchResponse,
    retries=4,  # Add retries for reliability
    system_prompt=(
        """You are a medical research assistant. Your job is to answer questions only based on 
        PubMed articles abstract provided. "
        Use numbered citations in your answer like [1], [2] at the end of relevant sentences. 
        In the citations section, format each citation as: "[number] Title - URL". 
        Example format:\n
        Answer: This is a finding [1]. Another finding [2].\n"
        Citations:\n"
        [1] First Article Title - http://url1\n"
        [2] Second Article Title - http://url2"
        Examples: Query: 'What is the treatment for diabetes?' keyword: 'diabetes treatment'
        Query: 'what causes hairfall?' keyword: 'hairfall'"""
    ),
)


@search_agent.tool
async def search_pubmed(ctx: RunContext[None], keyword: str) -> List[Article]:
    """Search PubMed for articles related to the keyword"""
    articles = fetch_articles(keyword)
    logfire.info("found {article_count} articles", article_count=len(articles))
    return [Article(**article) for article in articles]


@search_agent.result_validator
async def validate_result(ctx: RunContext[None], result: SearchResponse) -> SearchResponse:
    """Validate that the response includes citations"""
    if not result.citations:
        raise ModelRetry("Response must include at least one citation")
    return result


async def main():
    usage_limits = UsageLimits(request_limit=10)
    usage = Usage()
    message_history: list[ModelMessage] | None = None

    while True:
        query = input("\nEnter your medical research question (or 'quit' to exit): ")
        if query.lower() == "quit":
            break

        result = await search_agent.run(
            query, usage=usage, usage_limits=usage_limits, message_history=message_history
        )

        print("\nAnswer:")
        print(result.data.answer)
        print("\nCitations:")
        for citation in result.data.citations:
            print(f"- {citation}")

        message_history = result.all_messages()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
