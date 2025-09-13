"""
Graph-based medical research agent using pydantic-graph.

- Gathers evidence concurrently from 3 sub-agents (MedlinePlus, PubMed, ClinicalTrials)
- Summarizes into a single evidence-based answer with citations
- Prints a Mermaid diagram of the graph

Run directly to try it:
  python -m src.graph_agent "your medical question"
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from src.tools import (  # type: ignore
    fetch_clinical_trails,
    fetch_medline_plus,
    fetch_pubmed_articles,
)

# Update logfire configuration to track everything
logfire.configure(send_to_logfire=False, local=True)

GEMINI_FLASH_MODEL = "google-gla:gemini-2.5-flash"
GEMINI_PRO_MODEL = "google-gla:gemini-2.5-pro"
SUB_SYSTEM_PROMPT = (
    "You are a helpful medical research assistant. "
    "You will be provided with a user's query, and you will research using the tools provided to help user with evidence based answer. "
    "You will do a query fanout with multiple relevant search terms to get the best results. "
    "You will provide multiple articles with summarized relevant content and source. "
    "Provide a comprehensive answer. Ensure that you do not miss any detail or piece of information, even if its contribution seems minor, unless it is completely irrelevant to the topic. "
    "You always answer with inline citations. "
    "When using a tool, retrieve data and analyze it. If the information is insufficient or further details are needed, call the tool again with refined queries to gather more evidence. "
    "If you do not have enough information, you will say 'I don't know'. You will not make up an answer."
)


# Models and state
class SourceItem(BaseModel):
    """Normalized research item for summarization."""

    title: str = Field(default="")
    content: str = Field(default="")
    url: str = Field(default="")
    source: str = Field(default="")


@dataclass
class GraphState:
    """Graph run state: user question + gathered sources."""

    question: str
    medline: list[SourceItem] = field(default_factory=list)
    pubmed: list[SourceItem] = field(default_factory=list)
    clinical: list[SourceItem] = field(default_factory=list)


class SubResearch(BaseModel):
    """Sub-research task for a specific source"""

    content: str = Field(description="The research content with citation numbers")
    source: str = Field(
        description="list of jsons having citation number as key and value with the source of the research - link to article or study"
    )


medline_agent = Agent(
    model=GEMINI_FLASH_MODEL,
    system_prompt=SUB_SYSTEM_PROMPT,
    tools=[fetch_medline_plus],
    output_type=list[SubResearch],
    model_settings=ModelSettings(temperature=0.3),
)

pubmed_agent = Agent(
    model=GEMINI_FLASH_MODEL,
    system_prompt=SUB_SYSTEM_PROMPT,
    tools=[fetch_pubmed_articles],
    output_type=list[SubResearch],
    model_settings=ModelSettings(temperature=0.3),
)

clinical_trials_agent = Agent(
    model=GEMINI_FLASH_MODEL,
    system_prompt=SUB_SYSTEM_PROMPT,
    tools=[fetch_clinical_trails],
    output_type=list[SubResearch],
    model_settings=ModelSettings(temperature=0.3),
)


def _result_data(result: Any) -> Any:
    """Return data from pydantic-ai RunResult regardless of property name."""
    return getattr(result, "data", getattr(result, "output", None))


class SummarizerInput(BaseModel):
    question: str
    sources: list[SourceItem]


summarizer_agent = Agent(
    model=GEMINI_PRO_MODEL,
    system_prompt=(
        "You are an evidence synthesis assistant. "
        "Given a user's medical question and a list of sources, write a comprehensive, evidence-based answer. "
        "Use inline numeric citations like [1], [2] with links embedded that correspond to the numbered list of sources. "
        "If evidence is insufficient, say you don't know. Do not fabricate."
    ),
    output_type=str,
)


# Graph nodes
@dataclass
class GatherResearch(BaseNode[GraphState]):
    """Run three sub-agents concurrently and store results in state."""

    async def run(self, ctx: GraphRunContext[GraphState]) -> "Summarize":
        q = ctx.state.question

        # Run the three agents concurrently
        medline_task = medline_agent.run(q)
        pubmed_task = pubmed_agent.run(q)
        clinical_task = clinical_trials_agent.run(q)

        results = await asyncio.gather(
            medline_task, pubmed_task, clinical_task, return_exceptions=True
        )

        medline_res, pubmed_res, clinical_res = results

        def safe_items(res: Any, source_name: str) -> list[SourceItem]:
            if isinstance(res, Exception):
                logfire.warning("sub-agent failed", source=source_name, error=str(res))
                return []
            data = _result_data(res) or []
            # Ensure normalization to SourceItem
            items: list[SourceItem] = []
            for r in data:
                try:
                    # Map common dict shapes to SourceItem
                    if isinstance(r, dict):
                        items.append(
                            SourceItem(
                                title=r.get("title", ""),
                                content=r.get("content", ""),
                                url=r.get("url", ""),
                                source=source_name,
                            )
                        )
                    elif isinstance(r, SourceItem):
                        # Ensure source tag
                        if not r.source:
                            r.source = source_name
                        items.append(r)
                except Exception as e:  # Minimal defensive handling
                    logfire.warning("item-normalization-failed", source=source_name, error=str(e))
            return items

        ctx.state.medline = safe_items(medline_res, "medlineplus")
        ctx.state.pubmed = safe_items(pubmed_res, "pubmed")
        ctx.state.clinical = safe_items(clinical_res, "clinicaltrials")

        return Summarize()


@dataclass
class Summarize(BaseNode[GraphState, None, str]):
    """Summarize aggregated sources and end the run."""

    async def run(self, ctx: GraphRunContext[GraphState]) -> End[str]:
        sources = ctx.state.medline + ctx.state.pubmed + ctx.state.clinical

        if not sources:
            return End("Answer: I don't know. Citations: No sources found.")

        # Build compact, deterministic prompt for summarizer
        # Deduplicate by URL to avoid repeats
        seen = set()
        uniq: list[SourceItem] = []
        for s in sources:
            key = s.url or (s.title, s.source)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)

        lines = [
            f"Question: {ctx.state.question}",
            "",
            "Sources:",
        ]
        for i, s in enumerate(uniq, 1):
            title = s.title.strip() or "Untitled"
            url = s.url.strip()
            abstract = (s.content or "").strip()
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            lines.append(f"[{i}] {title} - {url}")
            if abstract:
                lines.append(f"Abstract: {abstract}")
        lines += [
            "",
            "Write a comprehensive evidence-based answer with inline numeric citations [1]..[n].",
            "Finish with a Citations section listing each as: [n] Title - URL.",
            "If evidence is insufficient, say you don't know.",
        ]

        prompt = "\n".join(lines)
        result = await summarizer_agent.run(prompt)
        answer = _result_data(result) or "Answer: I don't know."
        return End(answer)


# Graph definition
research_graph: Graph[GraphState, None, str] = Graph(nodes=[GatherResearch, Summarize])


async def run_graph(question: str) -> str:
    """Convenience runner for external callers."""
    state = GraphState(question=question)
    result = await research_graph.run(start_node=GatherResearch(), state=state)
    return result.output


def print_mermaid() -> None:
    code = research_graph.mermaid_code(start_node=GatherResearch)
    print("\nMermaid diagram:\n")
    print(code)


if __name__ == "__main__":
    # import sys

    # question = "What are effective treatments for migraine prevention?"
    # if len(sys.argv) > 1:
    #     question = " ".join(sys.argv[1:]).strip()

    # print("Question:", question)

    # # Show diagram
    # print_mermaid()

    # # Run the graph
    # print("\nRunning graph...\n")
    try:
        answer = asyncio.run(run_graph("what causes weakness of hair?"))
        print("\nAnswer:\n")
        print(answer)
    except Exception as e:
        print(f"Error: {e}")
    # print_mermaid()
    # result = medline_agent.run_sync("What are effective treatments for migraine prevention?")
    # print(result.output)
