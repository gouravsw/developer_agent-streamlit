import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent


AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
]

@tool
def static_code_check(code: str) -> str:
    """Run a lightweight static check on a Python code snippet."""
    issues = []
    if '"""' not in code and "'''" not in code:
        issues.append("No docstring found.")
    if "TODO" in code:
        issues.append("Contains TODO comment(s) left in the code.")
    if code.count("\n") > 40:
        issues.append("Function/file may be too long — consider splitting it.")
    return "; ".join(issues) if issues else "No obvious issues found."


DEFAULT_SNIPPET = '''
def process(data):
    # TODO: handle empty input
    result = []
    for d in data:
        result.append(d * 2)
    return result
'''

@st.cache_resource(show_spinner=False)
def build_agent(api_key: str, model_name: str):
    llm = ChatOpenAI(model=model_name, api_key=api_key)
    return create_agent(
        model=llm,
        tools=[static_code_check],
        system_prompt=(
            "You are a senior developer performing a code review. Use "
            "static_code_check on the snippet, then write a short, "
            "constructive review comment covering what to fix and why."
        ),
    )


def main() -> None:
    st.set_page_config(page_title="AI Developer Code Review", page_icon="🔍")
    st.title("AI Developer Code Review")
    st.caption("Choose an OpenAI model, enter your key for this session, and review Python code.")

    with st.sidebar:
        st.header("Model settings")
        api_key = st.text_input("OPENAI_API_KEY", type="password")
        model_name = st.selectbox("Model", AVAILABLE_MODELS)
        st.caption("The key is used for this Streamlit session and is not saved by this app.")

    code = st.text_area("Python code", value=DEFAULT_SNIPPET, height=320)

    if st.button("Review Code", type="primary", use_container_width=True):
        if not api_key.strip():
            st.error("Enter your OPENAI_API_KEY in the sidebar.")
        elif not code.strip():
            st.warning("Enter Python code to review.")
        else:
            with st.spinner(f"Reviewing with {model_name}..."):
                try:
                    agent = build_agent(api_key.strip(), model_name)
                    result = agent.invoke({
                        "messages": [{
                            "role": "user",
                            "content": f"Review this code:\n{code}",
                        }]
                    })
                    st.subheader("Review")
                    st.markdown(result["messages"][-1].content)
                except Exception as error:
                    st.error(f"Review failed: {error}")


if __name__ == "__main__":
    main()
