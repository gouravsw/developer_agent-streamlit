import os
import time

import streamlit as st
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
]
REQUIRED_COVERAGE = ["positive", "negative", "edge", "boundary"]

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
        f"Developer duration: {metrics['developer_seconds']:.2f} seconds\n"
        f"QA duration: {metrics['qa_seconds']:.2f} seconds\n"
        f"Developer tool calls: {metrics['developer_tool_calls']}\n"
        f"QA tool calls: {metrics['qa_tool_calls']}"
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
            f"- Developer duration: {metrics['developer_seconds']:.2f}s\n"
            f"- QA duration: {metrics['qa_seconds']:.2f}s\n"
            f"- Developer tool calls: {metrics['developer_tool_calls']}\n"
            f"- QA tool calls: {metrics['qa_tool_calls']}\n\n"
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
        "and validates test cases."
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

    if st.button("Run Developer + QA Review", type="primary", use_container_width=True):
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

        with st.spinner(f"Developer and QA agents are working with {model_name}..."):
            try:
                developer_agent = build_developer_agent(
                    openai_api_key.strip(), model_name
                )
                developer_started = time.perf_counter()
                developer_result = developer_agent.invoke({
                    "messages": [{
                        "role": "user",
                        "content": f"Review this code:\n{code}",
                    }]
                })
                developer_review = last_agent_message(developer_result)
                developer_seconds = time.perf_counter() - developer_started

                qa_agent = build_qa_agent(openai_api_key.strip(), model_name)
                qa_started = time.perf_counter()
                qa_result = qa_agent.invoke({
                    "messages": [{
                        "role": "user",
                        "content": (
                            "Create QA test cases for this submitted code.\n\n"
                            f"CODE:\n{code}\n\n"
                            f"DEVELOPER REVIEW:\n{developer_review}"
                        ),
                    }]
                })
                qa_seconds = time.perf_counter() - qa_started
                project_name = langsmith_project.strip() or "developer-qa-review"
                st.session_state.review_data = {
                    "developer_review": developer_review,
                    "qa_review": last_agent_message(qa_result),
                    "metrics": {
                        "tracing_enabled": tracing_enabled,
                        "project": project_name,
                        "developer_seconds": developer_seconds,
                        "qa_seconds": qa_seconds,
                        "developer_tool_calls": count_tool_calls(developer_result),
                        "qa_tool_calls": count_tool_calls(qa_result),
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
            st.write(f"LangSmith tracing enabled: {metrics['tracing_enabled']}")
            st.write(f"LangSmith project: {metrics['project']}")
            st.write(f"Developer duration: {metrics['developer_seconds']:.2f} seconds")
            st.write(f"QA duration: {metrics['qa_seconds']:.2f} seconds")
            st.write(f"Developer tool calls: {metrics['developer_tool_calls']}")
            st.write(f"QA tool calls: {metrics['qa_tool_calls']}")

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
