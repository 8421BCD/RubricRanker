import json
import litellm
import os
import pandas as pd
from tqdm import tqdm
import requests
import argparse
import datasets
import http.client
import litellm
from litellm.caching import Cache
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# TODO add key
# os.environ["OPENAI_API_KEY"] = "dummy"
litellm.cache = Cache(type="disk", disk_cache_dir="litellm_cache/")


# --------------- Prompts ---------------
retrieval_prompt_w_snippets = """You will receive: (1) a user Question that tests literature knowledge, and (2) a list of Snippets (each with an id and text).
Your task: design a rubric — a compact set of elements ("ingredients") that a high-quality final answer should satisfy, and map each element to the most relevant snippets.

Important: You are specifying what a *good answer must contain*, not grading any existing answer. Use ONLY the provided snippets for evidence.

--------------------------------
INPUT FORMAT
--------------------------------
- Question: a single string.
- Snippets: a list of items. Each item has:
  - id: a unique identifier (e.g., S_abcd123, DOI/CorpusID, or similar).
  - text: the snippet content (the ONLY citable text).

--------------------------------
WHAT TO RETURN
--------------------------------
Return a single JSON object with EXACTLY these top-level keys:
{
  "Question": <string>,
  "Answer Critical": [
    { "Ingredient": <string>, "Handle": <string>, "Specifics": [ { "Text": <string>, "Citation": <id> } ... ] }
  ],
  "Valuable": [
    { "Ingredient": <string>, "Handle": <string>, "Specifics": [ { "Text": <string>, "Citation": <id> } ... ] }
  ],
  "Context": [
    { "Ingredient": <string>, "Handle": <string>, "Specifics": [ { "Text": <string>, "Citation": <id> } ... ] }
  ]
}

--------------------------------
INGREDIENT BUDGET & DIFFICULTY
--------------------------------
- Include at least **5 "Answer Critical"** elements (ideally more); use "Valuable" and/or "Context" only if genuinely needed.
- Make each element **detailed and challenging**: it should bundle multiple precise, testable requirements for the same capability (multi-criteria), not broad or vague checks.
- Make each element **detailed and challenging**: write it as a **multi-criteria** requirement (multiple precise, testable sub-checks for a single capability).

Examples of strong multi-criteria formulations:
- "State the **exact <value>**; **cite ≥2 independent snippets** that directly support it; **specify the relevant time/version** (e.g., year, edition, release)."
- "Identify the **primary mechanism/definition**; **contrast with the closest alternative**; **include one qualifier** that constrains interpretation."
- "Report the **numeric result with units**; **include method/source context**; **note any threshold/exclusion** that changes the value."

--------------------------------
ELEMENT REQUIREMENTS
--------------------------------
Each element is one checklist item the final answer should satisfy:
- **Ingredient**: Start with a command verb (e.g., "State", "Specify", "Identify", "Cite", "Describe", "Compare", "Quantify", "Explain", "Include", "Discuss"). Make it **multi-criteria** for one capability; use semicolons/clauses to enumerate sub-requirements. Avoid vague "and/or."
- **Handle**: Short (2-5 words, Title Case), general label that abstracts the ingredient.
- **Specifics**: Zero or more exact quotes from snippets:
  - "Text": an **exact substring** from a snippet's **text** (preserve punctuation/casing; do not paraphrase).
  - "Citation": the snippet's **id exactly as provided**.
  - Prefer **multiple, diverse** snippets per element when available; avoid repeating the same quote.

--------------------------------
GROUPING & PRIORITY
--------------------------------
- **Answer Critical**: Minimal essentials that directly determine correctness/completeness (missing any makes the answer effectively wrong/incomplete).
- **Valuable**: Substantial supports that improve reliability, precision, or traceability (e.g., stronger sourcing, key qualifiers, scope clarifications).
- **Context**: Helpful background that aids understanding but is not required for correctness.

--------------------------------
RULES
--------------------------------
1) **Evidence scope**: Cite ONLY from the provided snippets. Do NOT use outside knowledge.
2) **Exactness**: Quotes must be exact substrings from snippet text. No edits to punctuation, casing, or numbers.
3) **Maximize distinct support**: Where possible, include ≥2 unique citations in a single element's "Specifics" to satisfy its multi-criteria checks.
4) **Context breadth**: Any "Context" element must include **≥2 citations** in "Specifics".
5) **Unsupported essentials (rare)**: If an ingredient is clearly essential but no snippet supports it, include it with "Specifics": [] — use sparingly and only with high confidence.
6) **No duplication**: Do not create multiple elements that check the same requirement; merge into a single, stronger multi-criteria ingredient.
7) **Stay under the cap**: Do not exceed **4 total elements** across all categories.
8) **Raw JSON only**: Return valid JSON (no Markdown, comments, or trailing commas).

--------------------------------
RECOMMENDED DISTRIBUTIONS (guidance, not strict)
--------------------------------
- 3 "Answer Critical" + 1 "Valuable"; or
- 2 "Answer Critical" + 1 "Valuable" + 1 "Context" (ensure Context has ≥2 citations).
"""

retrieval_prompt_no_snippets = """I will provide a query that requires up-to-date, real-world knowledge. Produce a comprehensive long-form report that synthesizes all necessary information. Your task is to come back with a list of elements you would expect to see in the answer. Each element should include an "ingredient", which is a detailed descriptor of what is expected in an answer (start with a verb whenever it makes sense) and a "handle", which is a short descriptor that slightly abstracts away from the ingredient description. Finally group the elements into 3 categories: "Answer Critical": necessary elements without this you'd not have an answer, "Valuable": key supporting information are useful but not as necessary as "Answer Critical", and "Context": elaborations or background that help understanding.

Respond with a json.
Example:
QUERY: What are advantages and disadvantages of top methods for picking the right number of topics in topic modeling?
{
"Question": "What are advantages and disadvantages of top methods for picking the right number of topics in topic modeling?",
"Answer Critical": [
  {
    "Ingredient": "Mention the most important methods for topic modeling, such as elbow method, coherence metrics, and likelihood, together with their advantages and disadvantages.",
    "Handle": "Advantages and Disadvantages of Methods",
  }
],
"Valuable": [
  {
    "Ingredient": "Highlight the importance of picking the right number of topics in the performance of topic models.",
    "Handle": "Importance of Selection Methods",
  },
  {
    "Ingredient": "Mention that the choice of the method depends on different factors such as computational capability, the intended application, and maybe a mix of various methods would be required.",
    "Handle": "Constraints and Dependencies",
  }
],
"Context": [
  {
    "Ingredient": "Define topic modeling and its applications.",
    "Handle": "Background",
  },
  {
    "Ingredient": "Name LDA as one of the most important methods for topic modeling.",
    "Handle": "Most Important Method",
  }
]
}


Example 2:
QUERY: Describe what is known about overfitting in in-context learning.
{
"Question": "Describe what is known about overfitting in in-context learning.",
"Answer Critical":
[
{
"Ingredient": "Define overfitting specifically in the context of in-context learning",
"Handle": "Definition"
},
{
"Ingredient": "Discuss the causes of overfitting in in-context learning.",
"Handle": "Causes"
}
],
"Valuable":
[
{
"Ingredient": "Describe known methods to prevent or reduce ICL overfitting",
"Handle": "Mitigation Strategies"
}
],
"Context":
[
{
"Ingredient": "Provide an explanation of overfitting in the fine-tuning or training frameworks.",
"Handle": "Background"
},
{
"Ingredient": "Provide background for in-context learning",
"Handle": "Background"
}
}
####
"""


# --------------- Util for data loading and saving ---------------
def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------- S2 API ---------------
def retrieve_s2_snippet(query: str, limit: int = 10, retries: int = 2):
    """
    Call S2 snippet search. On failure, retry a few times; if it still fails, return [].
    """
    url = "https://api.semanticscholar.org/graph/v1/snippet/search"
    params = {
        "query": query,
        "limit": int(limit),
        "minCitationCount": 10,
        # include paper fields we read below
        "fields": "snippet.text,paper.title,paper.paperId,paper.corpusId",
    }
    key = os.getenv("S2_API_KEY")
    headers = {"x-api-key": key} if key else {}

    attempts = retries + 1
    for i in range(attempts):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)

            # simple retry on 429/5xx
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                time.sleep(1.0)
                continue

            if not resp.ok:
                return []

            # parse JSON safely
            try:
                data = resp.json()
            except json.JSONDecodeError:
                time.sleep(0.5)
                continue

            if isinstance(data, str):
                # rare case: JSON returned as a string
                try:
                    data = json.loads(data)
                except Exception:
                    time.sleep(0.5)
                    continue

            processed = []
            for item in data.get("data", []):
                paper = item.get("paper") or {}
                snippet = item.get("snippet") or {}
                processed.append(
                    {
                        "paperID": paper.get("corpusId")
                        or paper.get("paperId")
                        or paper.get("id"),
                        "title": paper.get("title") or "",
                        "text": (snippet.get("text") or "").strip(),
                    }
                )
            return processed

        except requests.RequestException:
            time.sleep(0.5)
            continue

    # All attempts failed
    return []


# --------------- Web search API ---------------
def serper_search(query):
    try:
        conn = http.client.HTTPSConnection("google.serper.dev", timeout=20)
        key = os.getenv("SERPER_API_KEY")
        if not key:
            return []

        payload = json.dumps({"q": query})
        headers = {"X-API-KEY": key, "Content-Type": "application/json"}

        def _once():
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = res.read()
            return res.status, data

        status, data = _once()
        if status == 429:
            time.sleep(10)
            status, data = _once()

        if not (200 <= status < 300):
            return []

        try:
            j = json.loads(data.decode("utf-8"))
        except Exception:
            return []

        # Prefer "organic"; fall back if absent
        if isinstance(j, dict):
            for key_name in ("organic", "news", "knowledgeGraph", "answerBox"):
                v = j.get(key_name)
                if isinstance(v, list):
                    return v
        return []
    except Exception:
        return []


def serper_scrape(url):
    conn = http.client.HTTPSConnection("scrape.serper.dev")
    key = os.getenv("SERPER_API_KEY")
    payload = json.dumps({"url": url})
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    conn.request("POST", "/", payload, headers)
    res = conn.getresponse()
    data = res.read()
    try:
        j = json.loads(data.decode("utf-8"))
        if "text" in j and isinstance(j["text"], str):
            return j["text"]
    except Exception:
        pass  # fall through to retry

    # Retry once after a short backoff if "text" missing or any error occurred
    time.sleep(10)
    try:
        conn.request("POST", "/", payload, headers)
        res = conn.getresponse()
        data = res.read()
        j = json.loads(data.decode("utf-8"))
        return j.get("text", "")
    except Exception:
        return ""


def accept_up_to_max_words(
    scraped_text: str, max_words: int = 1000, add_ellipsis: bool = True
) -> str:
    """
    Return scraped_text truncated to at most `max_words` whitespace-delimited words.
    Uses a streaming regex to avoid creating large intermediate lists.
    """
    if not scraped_text:
        return scraped_text

    count = 0
    end_idx = None
    for m in re.finditer(r"\S+", scraped_text):
        count += 1
        end_idx = m.end()
        if count >= max_words:
            break

    # If fewer than max_words, return as-is
    if count < max_words:
        return scraped_text

    truncated = scraped_text[:end_idx].rstrip()
    return (truncated + "...") if add_ellipsis else truncated


def retrieve_web(query, run_scrape=True):
    serper_results = serper_search(query) or []
    snippets = []

    if not serper_results:
        return snippets

    if run_scrape is True:
        for idx, s in enumerate(serper_results):
            title = s.get("title") or s.get("link") or "Untitled"
            link = s.get("link") or s.get("url") or ""
            summary = s.get("snippet") or s.get("description") or ""

            if idx < 3 and link:
                scraped_text = accept_up_to_max_words(serper_scrape(link))
                full_text = (
                    f"Summary: {summary}\nFull text:\n{scraped_text}"
                    if scraped_text
                    else f"Summary: {summary}"
                )
                snippets.append({"title": title, "text": full_text})
            else:
                snippets.append({"title": title, "text": summary})
    else:
        for s in serper_results:
            title = s.get("title") or s.get("link") or "Untitled"
            summary = s.get("snippet") or s.get("description") or ""
            snippets.append({"title": title, "text": summary})

    return snippets


def extract_json_from_response(response):
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    if json_start == -1 or json_end == -1:
        return None

    try:
        return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        try:
            return json.loads(response[json_start:json_end] + "]}")
        except json.JSONDecodeError:
            print(
                f"Could not decode JSON from response: {response[json_start:json_end]}"
            )
        return None


def call_llm_snippets(
    query, output_file, error_file, no_retrieval=False, search_tool="s2"
):
    if no_retrieval is True:
        prompt = retrieval_prompt_no_snippets + f"Query: {query}"
    else:
        if search_tool == "s2":
            snippets = retrieve_s2_snippet(query)
            time.sleep(1)
        else:
            snippets = retrieve_web(query)

        snippets_text = " ".join(
            [item["title"] + "\n" + item["text"] for item in snippets]
        )
        prompt = (
            retrieval_prompt_w_snippets + f"Query: {query}\nSnippets:\n{snippets_text}"
        )

    msgs = [{"role": "user", "content": prompt}]
    resp = litellm.completion(
        model="gpt-4.1-mini-2025-04-14",  # originally gpt-4.1
        messages=msgs,
    )

    output = resp.choices[0].message.content
    cost = litellm.completion_cost(completion_response=resp)

    output = extract_json_from_response(output)
    if output is None:
        with open(error_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "query": query,
                    }
                )
                + "\n"
            )
    else:
        with open(output_file, "a") as f:
            f.write(json.dumps(output) + "\n")

    return {"query": query, "cost": cost}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_path", type=str, default=None, help="Path to input HF")
    ap.add_argument("--input", type=str, help="Path to write input JSONL")
    ap.add_argument(
        "--output", type=str, required=True, help="Path to write output JSONL"
    )
    ap.add_argument(
        "--error", type=str, required=True, help="Path to write error JSONL"
    )
    ap.add_argument("--mode", type=str, required=True, help="Rubric generation mode")
    ap.add_argument("--start_idx", type=int, default=0, help="Start index")
    ap.add_argument("--end_idx", type=int, default=-1, help="Start index")
    ap.add_argument(
        "--search_tool",
        type=str,
        help="Search tools to be used for rubric generation modes",
    )
    ap.add_argument(
        "--no_retrieval",
        action="store_true",
        help="set true to generate retrieval-free rubrics.",
    )
    ap.add_argument("--asta", action="store_true", help="asta processed data")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.hf_path is not None:
        input_data = list(datasets.load_dataset(args.hf_path)["train"])
        input_data = [item for item in input_data]
    else:
        input_data = read_jsonl(args.input)

    print(input_data[0])
    results = []

    for item in input_data:
        if "query" in item and "question" not in item:
            item["question"] = item["query"]
        if "messages" in item:
            item["question"] = item["messages"][0]["content"]

    # this filter out search arena and healthbench
    results = []
    end_idx = args.end_idx if args.end_idx > 0 else len(input_data)
    for q in tqdm(input_data[args.start_idx : end_idx]):
        if args.mode == "snippets":
            # try:
            res = call_llm_snippets(
                q["question"],
                args.output,
                args.error,
                args.no_retrieval,
                search_tool=args.search_tool,
            )
            results.append(res)
        elif args.mode == "general":
            # try:
            res = call_llm_snippets(q["question"], args.output, args.error, True)
            results.append(res)
    costs = 0
    for result in results:
        if "cost" in result:
            costs += result["cost"]
    print(f"Total cost: {costs}")


if __name__ == "__main__":
    main()
