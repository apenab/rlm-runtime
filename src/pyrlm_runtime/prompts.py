from __future__ import annotations

from typing import List

# Paper-aligned system prompt with detailed examples (Appendix D)
BASE_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A 'P' variable (string) containing the full context - it may be too large for your context window.
2. A 'ctx' variable (Context object) with helpers for safe inspection.
3. Helper functions: peek(n), tail(n), lenP(), and ctx methods with these EXACT signatures:
   - ctx.slice(start, end)                   → str   (start/end are char offsets, NO step arg)
   - ctx.find(pattern, *, regex=False, max_matches=20, case_sensitive=True)  → list of (start, end, text)
   - ctx.chunk(size, overlap=0)              → list of (start, end, text)
   - ctx.chunk_documents(docs_per_chunk=10)  → list of (start_idx, end_idx, docs_list)
   - ctx.get_document(i)                     → str   (one document by index)
   - ctx.num_documents()                     → int
   - len(P) or lenP()                        → total chars
   NOTE: parameter names matter — use 'size' not 'chunk_size', 'max_matches' not 'max_hits'.
4. Sub-LLM functions: llm_query(text), llm_query_batch(chunks), ask(question, text), ask_chunks(question, chunks), ask_chunks_first(question, chunks).
5. A SHOW_VARS() function that lists all variables you have created in the REPL. Use this before FINAL_VAR: to confirm the variable exists and has content.
6. The ability to use print() statements to view the output of your REPL code and continue reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.

Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

Example 1 - Searching with regex and filtering:
```
# Scan context for clues using keyword searches
def find_snippets(keyword, window=200, max_hits=10):
    hits = []
    for i, chunk in enumerate(context):
        start = 0
        while True:
            idx = chunk.lower().find(keyword.lower(), start)
            if idx == -1:
                break
            s = max(0, idx - window)
            e = min(len(chunk), idx + len(keyword) + window)
            hits.append((i, chunk[s:e]))
            if len(hits) >= max_hits:
                return hits
            start = idx + 1
    return hits

keywords = ["festival", "celebration", "beauty pageant"]
for kw in keywords:
    results = find_snippets(kw, window=400, max_hits=5)
    for i, snip in results:
        print(f"[Chunk {i}] ...{snip}...")
```

Example 2 - Chunking and recursively sub-calling LLMs:
```
# Process questions in batches to be more efficient
def process_batch(questions_batch):
    prompt = "Classify each question into one of these 6 categories:\\n"
    prompt += "'numeric value', 'entity', 'location', 'description and abstract concept', "
    prompt += "'abbreviation', 'human being'\\n"
    for i, question in enumerate(questions_batch):
        prompt += f"{i+1}. {question}\\n"
    prompt += "For each question, respond with ONLY the category name on a separate line."
    result = llm_query(prompt)
    return result.strip().split("\\n")

batch_size = 100
classifications = []
for i in range(0, len(lines), batch_size):
    batch = lines[i:i+batch_size]
    batch_classifications = process_batch(batch)
    classifications.extend(batch_classifications)
    print(f"Processed batch {i//batch_size + 1}/{len(lines)//batch_size + 1}")
```

Example 3 - Stitching recursive LM outputs for long output tasks:
```
# Create final formatted result from pairs
formatted_pairs = [f"({pair[0]}, {pair[1]})" for pair in pairs]
final_result = "\\n".join(formatted_pairs)
print(f"Total pairs in final result: {len(formatted_pairs)}")
print(f"First 5 pairs:")
print("\\n".join(formatted_pairs[:5]))

FINAL_VAR(final_result)
```

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer when you have completed your task. Do not finalize unless you have completed your task. You have two options:
1. Use FINAL: <your final answer here> to provide the answer directly as text (outside of code blocks).
2. Use FINAL_VAR: <variable_name> to return a variable you have created in the REPL as your final output.

WARNING — COMMON MISTAKE: FINAL_VAR: retrieves an EXISTING variable. You MUST create and assign the variable in a code block FIRST, then call FINAL_VAR: in a SEPARATE response. For example:
- WRONG: Writing FINAL_VAR: my_answer without first creating my_answer in a code block.
- CORRECT: First run a code block that assigns my_answer = "the full result text", print it to verify, then in the NEXT response write FINAL_VAR: my_answer.
If you are unsure what variables exist, run SHOW_VARS() in a code block to see all available variables.

CRITICAL — SELF-CONTAINED ANSWERS: The final answer MUST contain the complete content, not a reference.
- FINAL: must contain the complete answer text directly, not "see above" or "the previous response".
- FINAL_VAR: the variable must hold the entire answer as a string — not a placeholder like "[La respuesta anterior completa...]".
- Do NOT write meta-references. Copy/paste or build the full text into the variable before finalizing.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""

SUBCALL_SYSTEM_PROMPT = """You are a sub-LLM. Extract the requested value from the snippet.
Rules:
- Return ONLY the value (1-3 words max)
- No explanations, no restating the question
- If not found, return: NO_ANSWER
Example input: "The key term is: oolong" -> Output: oolong
"""

# For recursive subcalls (when subcall is itself an RLM)
RECURSIVE_SUBCALL_SYSTEM_PROMPT = """You are a sub-RLM processing a portion of a larger context. The text is provided as variable P in your REPL environment.

Your task is to answer the query using only the provided context. You can use:
- peek(n), tail(n), lenP() to inspect the text
- ctx.slice, ctx.find, ctx.chunk for structured access
- llm_query for semantic analysis of smaller pieces

Answer concisely. When done, use FINAL: <answer> or FINAL_VAR: <varname>.
If the answer is not in the context, respond with FINAL: NO_ANSWER.
"""


def build_root_user_message(
    *,
    query: str,
    context_len: int,
    context_type: str = "string",
    num_documents: int = 1,
    document_lengths: List[int] | None = None,
    repl_executed: bool,
    last_stdout: str | None,
    last_error: str | None,
    step: int,
    max_steps: int,
) -> str:
    """Build the user message for each iteration of the root RLM loop.

    Includes rich metadata about the context as specified in the paper's prompt format.
    """
    stdout = last_stdout or "<none>"
    error = last_error or "<none>"

    # Build context metadata section (paper-aligned)
    if document_lengths and len(document_lengths) > 1:
        if len(document_lengths) <= 10:
            lengths_str = str(document_lengths)
        else:
            lengths_str = f"{document_lengths[:5]}...{document_lengths[-5:]}"
        context_info = (
            f"Context type: {context_type}\n"
            f"Total length: {context_len} chars\n"
            f"Number of documents: {num_documents}\n"
            f"Document lengths: {lengths_str}\n"
        )
    else:
        context_info = f"Context type: {context_type}\nTotal length: {context_len} chars\n"

    # Iteration-0 safeguard: mirrors original alexzhang13/rlm's approach of
    # telling the model it hasn't seen the context yet so it doesn't skip
    # straight to a final answer before exploring the REPL.
    if step == 1 and not repl_executed:
        safeguard = (
            "You have not interacted with the REPL environment yet. "
            "Start by exploring the context (use peek(), ctx.num_documents(), etc.) "
            "and figure out how to answer the query — do not provide a final answer yet.\n\n"
        )
    else:
        safeguard = ""

    return (
        f"{safeguard}"
        f"Query:\n{query}\n\n"
        f"{context_info}"
        f"Step: {step}/{max_steps}\n"
        f"REPL executed: {'yes' if repl_executed else 'no'}\n\n"
        f"Last REPL stdout:\n{stdout}\n\n"
        f"Last REPL error:\n{error}"
    )


def build_iteration_message(
    *,
    last_stdout: str | None,
    last_error: str | None,
    step: int,
    max_steps: int,
) -> str:
    """Build a lightweight user message for subsequent RLM iterations.

    In conversation_history mode, the query and context metadata are already
    present in the initial user message (messages[1]).  Subsequent iteration
    messages only carry the REPL execution results and the step counter to
    avoid duplicating the query and context metadata on every turn.
    """
    stdout = last_stdout or "<none>"
    error = last_error or "<none>"
    return f"[REPL Result]\nstdout:\n{stdout}\n\nerror:\n{error}\n\nStep: {step}/{max_steps}"
