import os
import re
import time
from typing import TypedDict

import streamlit as st
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
]
REQUIRED_COVERAGE = ["positive", "negative", "edge", "boundary"]
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 15

DEFAULT_SNIPPET = """\
def process(data):
    # TODO: handle empty input
    result = []
    for item in data:
        result.append(item * 2)
    return result
"""


@tool
def static_code_check(code: str) -> str:
    """Run a lightweight static check on a Python code snippet."""
    issues = []
    if '"""' not in code and "'''" not in code:
        issues.append("No docstring found.")
    if "TODO" in code:
        issues.append("Contains TODO comment(s) left in the code.")
    if code.count("\n") > 40:
        issues.append("Function/file may be too long - consider splitting it.")
    return "; ".join(issues) if issues else "No obvious issues found."


@tool
def check_test_coverage(test_cases_text: str) -> str:
    """Check test cases for positive, negative, edge, and boundary coverage."""
    lower = test_cases_text.lower()
    missing = [category for category in REQUIRED_COVERAGE if category not in lower]
    if not missing:
        return "Coverage looks complete: all required categories present."
    return (
        f"Missing coverage for: {', '.join(missing)}. "
        "Please add cases for these."
    )


def configure_langsmith(api_key: str, project_name: str) -> bool:
    """Enable or disable LangSmith tracing for the current Streamlit process."""
    if not api_key:
        os.environ.pop("LANGSMITH_API_KEY", None)
        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        return False

    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project_name or "developer-qa-review"
    return True


@st.cache_resource(show_spinner=False)
def build_developer_agent(api_key: str, model_name: str):
    llm = ChatOpenAI(model=model_name, api_key=api_key)
    return create_agent(
        model=llm,
        tools=[static_code_check],
        system_prompt=(
            "You are the Developer Agent. Review the submitted Python code. "
            "Use static_code_check, then write a concise review with findings, "
            "severity, and recommended fixes."
        ),
    )


@st.cache_resource(show_spinner=False)
def build_qa_agent(api_key: str, model_name: str):
    llm = ChatOpenAI(model=model_name, api_key=api_key)
    return create_agent(
        model=llm,
        tools=[check_test_coverage],
        system_prompt=(
            "You are the QA Agent. The Developer Agent has already reviewed "
            "the code. Draft concise test cases based on the code and developer "
            "review. Label every case Positive, Negative, Edge, or Boundary. "
            "Call check_test_coverage on your draft, then revise it if coverage "
            "is missing. End with a brief QA recommendation."
        ),
    )


class ReviewState(TypedDict, total=False):
    code: str
    attempt: int
    max_attempts: int
    developer_review: str
    qa_review: str
    static_result: str
    coverage_result: str
    passed: bool
    reports: list[dict]
    developer_seconds: float
    qa_seconds: float
    total_tool_calls: int
    total_usage: dict
    fixer_usage: dict


def extract_code(response: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return response.strip()


def empty_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def usage_from_result(result: dict) -> dict[str, int]:
    usage = empty_usage()
    for message in result.get("messages", []):
        metadata = getattr(message, "usage_metadata", None) or {}
        if not metadata and isinstance(message, dict):
            metadata = message.get("usage_metadata", {}) or {}
        usage["input_tokens"] += int(
            metadata.get("input_tokens", metadata.get("prompt_tokens", 0)) or 0
        )
        usage["output_tokens"] += int(
            metadata.get("output_tokens", metadata.get("completion_tokens", 0)) or 0
        )
        usage["total_tokens"] += int(metadata.get("total_tokens", 0) or 0)
    if usage["total_tokens"] == 0:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def usage_from_message(message) -> dict[str, int]:
    metadata = getattr(message, "usage_metadata", None) or {}
    usage = empty_usage()
    usage["input_tokens"] = int(metadata.get("input_tokens", 0) or 0)
    usage["output_tokens"] = int(metadata.get("output_tokens", 0) or 0)
    usage["total_tokens"] = int(metadata.get("total_tokens", 0) or 0)
    if usage["total_tokens"] == 0:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def add_usage(*usages: dict) -> dict[str, int]:
    combined = empty_usage()
    for usage in usages:
        for key in combined:
            combined[key] += int(usage.get(key, 0) or 0)
    return combined


def format_usage(usage: dict) -> str:
    return (
        f"Input: {usage.get('input_tokens', 0):,} | "
        f"Output: {usage.get('output_tokens', 0):,} | "
        f"Total: {usage.get('total_tokens', 0):,}"
    )


def build_review_graph(
    api_key: str, model_name: str, max_attempts: int = MAX_RETRIES + 1
):
    developer_agent = build_developer_agent(api_key, model_name)
    qa_agent = build_qa_agent(api_key, model_name)
    fixer_llm = ChatOpenAI(model=model_name, api_key=api_key)

    def review_node(state: ReviewState) -> ReviewState:
        attempt = state.get("attempt", 0) + 1
        code = state["code"]
        static_result = static_code_check.invoke({"code": code})

        developer_started = time.perf_counter()
        developer_result = developer_agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Review this Python code (attempt {attempt}):\n{code}",
            }]
        })
        developer_seconds = time.perf_counter() - developer_started
        developer_review = last_agent_message(developer_result)

        qa_started = time.perf_counter()
        qa_result = qa_agent.invoke({
            "messages": [{
                "role": "user",
                "content": (
                    f"Create QA test cases for attempt {attempt}.\n\n"
                    f"CODE:\n{code}\n\n"
                    f"DEVELOPER REVIEW:\n{developer_review}"
                ),
            }]
        })
        qa_seconds = time.perf_counter() - qa_started
        qa_review = last_agent_message(qa_result)
        developer_usage = usage_from_result(developer_result)
        qa_usage = usage_from_result(qa_result)
        fixer_usage = state.get("fixer_usage", empty_usage())
        attempt_usage = add_usage(developer_usage, qa_usage, fixer_usage)
        coverage_result = check_test_coverage.invoke({"test_cases_text": qa_review})
        passed = (
            static_result == "No obvious issues found."
            and coverage_result == "Coverage looks complete: all required categories present."
        )
        report = {
            "attempt": attempt,
            "code": code,
            "developer_review": developer_review,
            "qa_review": qa_review,
            "static_result": static_result,
            "coverage_result": coverage_result,
            "passed": passed,
            "developer_seconds": developer_seconds,
            "qa_seconds": qa_seconds,
            "developer_usage": developer_usage,
            "qa_usage": qa_usage,
            "fixer_usage": fixer_usage,
            "attempt_usage": attempt_usage,
        }
        return {
            "attempt": attempt,
            "developer_review": developer_review,
            "qa_review": qa_review,
            "static_result": static_result,
            "coverage_result": coverage_result,
            "passed": passed,
            "reports": state.get("reports", []) + [report],
            "developer_seconds": developer_seconds,
            "qa_seconds": qa_seconds,
            "total_tool_calls": state.get("total_tool_calls", 0)
            + count_tool_calls(developer_result)
            + count_tool_calls(qa_result),
            "total_usage": add_usage(
                state.get("total_usage", empty_usage()),
                developer_usage,
                qa_usage,
            ),
            "fixer_usage": empty_usage(),
        }

    def publish_node(state: ReviewState) -> ReviewState:
        return state

    def route_after_publish(state: ReviewState) -> str:
        if state.get("passed") or state.get("attempt", 0) >= max_attempts:
            return "finish"
        return "fix"

    def fix_node(state: ReviewState) -> ReviewState:
        latest = state["reports"][-1]
        response = fixer_llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a code-fixing agent. Return only the corrected Python "
                    "code in one python code block. Fix the reported static issues "
                    "without changing the intended behavior."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CODE:\n{latest['code']}\n\n"
                    f"STATIC CHECK:\n{latest['static_result']}\n\n"
                    f"DEVELOPER REVIEW:\n{latest['developer_review']}\n\n"
                    f"QA COVERAGE:\n{latest['coverage_result']}"
                ),
            },
        ])
        fixer_usage = usage_from_message(response)
        return {
            "code": extract_code(response.content),
            "fixer_usage": fixer_usage,
            "total_usage": add_usage(
                state.get("total_usage", empty_usage()), fixer_usage
            ),
        }

    def wait_node(state: ReviewState) -> ReviewState:
        time.sleep(RETRY_WAIT_SECONDS)
        return state

    graph = StateGraph(ReviewState)
    graph.add_node("review", review_node)
    graph.add_node("publish", publish_node)
    graph.add_node("fix", fix_node)
    graph.add_node("wait", wait_node)
    graph.add_edge(START, "review")
    graph.add_edge("review", "publish")
    graph.add_conditional_edges(
        "publish", route_after_publish, {"finish": END, "fix": "fix"}
    )
    graph.add_edge("fix", "wait")
    graph.add_edge("wait", "review")
    return graph.compile()


def run_review_graph(code: str, api_key: str, model_name: str) -> ReviewState:
    workflow = build_review_graph(api_key, model_name)
    return workflow.invoke({
        "code": code,
        "attempt": 0,
        "max_attempts": MAX_RETRIES + 1,
        "reports": [],
        "total_tool_calls": 0,
        "total_usage": empty_usage(),
    })


def last_agent_message(result: dict) -> str:
    return result["messages"][-1].content


def count_tool_calls(result: dict) -> int:
    return sum(
        1
        for message in result.get("messages", [])
        if getattr(message, "type", None) == "tool"
        or (isinstance(message, dict) and message.get("role") == "tool")
    )


def coverage_summary(test_cases: str) -> str:
    missing = [category for category in REQUIRED_COVERAGE if category not in test_cases.lower()]
    if missing:
        return f"Missing coverage categories: {', '.join(missing)}."
    return "All required categories are covered: positive, negative, edge, and boundary."


def review_context() -> str:
    review = st.session_state.get("review_data")
    if not review:
        return "No review has been run yet. Ask the user to run Developer + QA Review first."
    metrics = review["metrics"]
    return (
        f"Developer review:\n{review['developer_review']}\n\n"
        f"QA test cases:\n{review['qa_review']}\n\n"
        f"Coverage result: {coverage_summary(review['qa_review'])}\n"
        f"Tracing enabled: {metrics['tracing_enabled']}\n"
        f"LangSmith project: {metrics['project']}\n"
        f"LangGraph attempts: {metrics['attempts']}\n"
        f"Workflow passed: {metrics['passed']}\n"
        f"Developer duration: {metrics['developer_seconds']:.2f} seconds\n"
        f"QA duration: {metrics['qa_seconds']:.2f} seconds\n"
        f"Total tool calls: {metrics['total_tool_calls']}\n"
        f"Total token usage: {format_usage(metrics['total_usage'])}"
    )


def answer_chat_question(question: str, api_key: str, model_name: str) -> str:
    lowered = question.lower()
    review = st.session_state.get("review_data")
    if not review:
        return "Run Developer + QA Review first so I have test and tracing results to discuss."
    if "coverage" in lowered or "test" in lowered:
        return (
            f"{coverage_summary(review['qa_review'])}\n\n"
            f"QA Agent output:\n{review['qa_review']}"
        )
    if "metric" in lowered or "trace" in lowered or "langsmith" in lowered:
        metrics = review["metrics"]
        return (
            "Latest run metrics:\n"
            f"- Tracing enabled: {metrics['tracing_enabled']}\n"
            f"- Project: {metrics['project']}\n"
            f"- LangGraph attempts: {metrics['attempts']}\n"
            f"- Workflow passed: {metrics['passed']}\n"
            f"- Developer duration: {metrics['developer_seconds']:.2f}s\n"
            f"- QA duration: {metrics['qa_seconds']:.2f}s\n"
            f"- Total tool calls: {metrics['total_tool_calls']}\n\n"
            f"- Total token usage: {format_usage(metrics['total_usage'])}\n\n"
            "Open the LangSmith project to inspect prompts, model responses, tool inputs/outputs, tokens, cost, and nested run timing."
        )

    chat_agent = ChatOpenAI(model=model_name, api_key=api_key)
    response = chat_agent.invoke([
        {
            "role": "system",
            "content": "Answer questions about this Developer and QA review. Use only the supplied context.",
        },
        {"role": "user", "content": f"CONTEXT:\n{review_context()}\n\nQUESTION:\n{question}"},
    ])
    return response.content


def main() -> None:
    st.set_page_config(page_title="Developer and QA Review", page_icon="review")
    st.title("Developer and QA Code Review")
    st.caption(
        "The Developer Agent reviews the code first; the QA Agent then creates "
        f"and validates test cases. Failed runs are fixed and retried up to {MAX_RETRIES} "
        f"times after a {RETRY_WAIT_SECONDS}-second wait."
    )
    if "review_data" not in st.session_state:
        st.session_state.review_data = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    with st.sidebar:
        st.header("Model settings")
        openai_api_key = st.text_input("OPENAI_API_KEY", type="password")
        model_name = st.selectbox("OpenAI model", AVAILABLE_MODELS)
        st.caption("Keys are used for this session and are not written to disk by this app.")

        st.header("LangSmith tracing")
        langsmith_api_key = st.text_input("LANGSMITH_API_KEY", type="password")
        langsmith_project = st.text_input(
            "LangSmith project", value="developer-qa-review"
        )
        st.caption("Optional. When provided, both agent runs are traced in LangSmith.")

    code = st.text_area("Python code", value=DEFAULT_SNIPPET, height=320)

    if st.button("Run LangGraph Developer + QA Review", type="primary", use_container_width=True):
        if not openai_api_key.strip():
            st.error("Enter your OPENAI_API_KEY in the sidebar.")
            return
        if not code.strip():
            st.warning("Enter Python code to review.")
            return

        tracing_enabled = configure_langsmith(
            langsmith_api_key.strip(), langsmith_project.strip()
        )
        if tracing_enabled:
            st.info(
                "LangSmith tracing enabled: "
                f"{langsmith_project.strip() or 'developer-qa-review'}"
            )

        with st.spinner(
            "LangGraph is reviewing, publishing reports, and retrying failed runs..."
        ):
            try:
                final_state = run_review_graph(
                    code, openai_api_key.strip(), model_name
                )
                latest_report = final_state["reports"][-1]
                project_name = langsmith_project.strip() or "developer-qa-review"
                st.session_state.review_data = {
                    "developer_review": latest_report["developer_review"],
                    "qa_review": latest_report["qa_review"],
                    "reports": final_state["reports"],
                    "metrics": {
                        "tracing_enabled": tracing_enabled,
                        "project": project_name,
                        "attempts": final_state["attempt"],
                        "passed": final_state["passed"],
                        "developer_seconds": final_state["developer_seconds"],
                        "qa_seconds": final_state["qa_seconds"],
                        "total_tool_calls": final_state["total_tool_calls"],
                        "total_usage": final_state.get("total_usage", empty_usage()),
                    },
                }

            except Exception as error:
                st.error(f"Multi-agent review failed: {error}")

    review = st.session_state.get("review_data")
    if review:
        st.subheader("1. Developer Agent Review")
        st.markdown(review["developer_review"])

        st.subheader("2. QA Agent Test Cases")
        st.markdown(review["qa_review"])

        with st.expander("Latest run metrics"):
            metrics = review["metrics"]
            total_usage = metrics["total_usage"]
            token_columns = st.columns(3)
            token_columns[0].metric("Input tokens", f"{total_usage['input_tokens']:,}")
            token_columns[1].metric("Output tokens", f"{total_usage['output_tokens']:,}")
            token_columns[2].metric("Total tokens", f"{total_usage['total_tokens']:,}")
            st.caption(
                "Token cost is calculated in LangSmith using the selected model's pricing."
            )
            st.write(f"LangSmith tracing enabled: {metrics['tracing_enabled']}")
            st.write(f"LangSmith project: {metrics['project']}")
            st.write(f"LangGraph attempts: {metrics['attempts']}")
            st.write(f"Workflow passed: {metrics['passed']}")
            st.write(f"Developer duration: {metrics['developer_seconds']:.2f} seconds")
            st.write(f"QA duration: {metrics['qa_seconds']:.2f} seconds")
            st.write(f"Total tool calls: {metrics['total_tool_calls']}")
            st.write(f"Total token usage: {format_usage(metrics['total_usage'])}")

        st.subheader("LangGraph attempt reports")
        for report in review.get("reports", []):
            label = "passed" if report["passed"] else "retry required"
            with st.expander(f"Attempt {report['attempt']} - {label}"):
                st.code(report["code"], language="python")
                st.write(f"Static check: {report['static_result']}")
                st.write(f"Coverage check: {report['coverage_result']}")
                st.write(
                    f"Developer tokens: {format_usage(report['developer_usage'])}"
                )
                st.write(f"QA tokens: {format_usage(report['qa_usage'])}")
                st.write(f"Fixer tokens: {format_usage(report['fixer_usage'])}")
                st.write(f"Attempt total: {format_usage(report['attempt_usage'])}")
                st.markdown("**Developer report**")
                st.markdown(report["developer_review"])
                st.markdown("**QA report**")
                st.markdown(report["qa_review"])

    st.divider()
    st.subheader("Ask about this review")
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask about test coverage, tracing metrics, or the review"
    )
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                answer = answer_chat_question(
                    question, openai_api_key.strip(), model_name
                )
                st.markdown(answer)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": answer,
                })
            except Exception as error:
                st.error(f"Chat failed: {error}")


if __name__ == "__main__":
    main()
