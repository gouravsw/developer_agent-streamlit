import os

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


def main() -> None:
    st.set_page_config(page_title="Developer and QA Review", page_icon="review")
    st.title("Developer and QA Code Review")
    st.caption(
        "The Developer Agent reviews the code first; the QA Agent then creates "
        "and validates test cases."
    )

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
                developer_result = developer_agent.invoke({
                    "messages": [{
                        "role": "user",
                        "content": f"Review this code:\n{code}",
                    }]
                })
                developer_review = last_agent_message(developer_result)

                st.subheader("1. Developer Agent Review")
                st.markdown(developer_review)

                qa_agent = build_qa_agent(openai_api_key.strip(), model_name)
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

                st.subheader("2. QA Agent Test Cases")
                st.markdown(last_agent_message(qa_result))
            except Exception as error:
                st.error(f"Multi-agent review failed: {error}")


if __name__ == "__main__":
    main()
