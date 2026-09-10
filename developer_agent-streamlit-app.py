import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

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


tools = [static_code_check]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a senior developer performing a code review. Use "
        "static_code_check on the snippet, then write a short, "
        "constructive review comment covering what to fix and why."
    ),
    debug=True,
)

DEFAULT_SNIPPET = '''
def process(data):
    # TODO: handle empty input
    result = []
    for d in data:
        result.append(d * 2)
    return result
'''

class DeveloperReviewApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AI Developer Code Review")
        self.geometry("900x700")
        self.minsize(700, 520)
        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(3, weight=2)

        ttk.Label(
            self,
            text="AI Developer Code Review",
            font=("TkDefaultFont", 16, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        input_frame = ttk.LabelFrame(self, text="Python code", padding=8)
        input_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)

        self.code_input = scrolledtext.ScrolledText(
            input_frame, wrap="none", undo=True, font=("Courier New", 10)
        )
        self.code_input.grid(row=0, column=0, sticky="nsew")
        self.code_input.insert("1.0", DEFAULT_SNIPPET)

        controls = ttk.Frame(self)
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
        controls.columnconfigure(1, weight=1)

        self.review_button = ttk.Button(
            controls, text="Review Code", command=self._start_review
        )
        self.review_button.grid(row=0, column=0, padx=(0, 10))

        self.status = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.status).grid(
            row=0, column=1, sticky="w"
        )

        output_frame = ttk.LabelFrame(self, text="Review", padding=8)
        output_frame.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 12))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.review_output = scrolledtext.ScrolledText(
            output_frame, wrap="word", state="disabled", font=("TkDefaultFont", 10)
        )
        self.review_output.grid(row=0, column=0, sticky="nsew")

    def _start_review(self) -> None:
        code = self.code_input.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Missing code", "Enter Python code to review.")
            return
        if not os.getenv("OPENAI_API_KEY"):
            messagebox.showerror(
                "Missing API key",
                "Set OPENAI_API_KEY in the terminal before starting the app.",
            )
            return

        self.review_button.config(state="disabled")
        self.status.set("Reviewing code...")
        self._set_output("Running static checks and asking the AI reviewer...")
        threading.Thread(target=self._review_code, args=(code,), daemon=True).start()

    def _review_code(self, code: str) -> None:
        try:
            result = agent.invoke({
                "messages": [{
                    "role": "user",
                    "content": f"Review this code:\n{code}",
                }]
            })
            review = result["messages"][-1].content
            self.after(0, self._review_succeeded, review)
        except Exception as error:
            self.after(0, self._review_failed, error)

    def _review_succeeded(self, review: str) -> None:
        self._set_output(review)
        self.status.set("Review complete")
        self.review_button.config(state="normal")

    def _review_failed(self, error: Exception) -> None:
        self._set_output(f"Review failed:\n{error}")
        self.status.set("Review failed")
        self.review_button.config(state="normal")

    def _set_output(self, text: str) -> None:
        self.review_output.config(state="normal")
        self.review_output.delete("1.0", tk.END)
        self.review_output.insert("1.0", text)
        self.review_output.config(state="disabled")


def main() -> None:
    app = DeveloperReviewApp()
    app.mainloop()


if __name__ == "__main__":
    main()
