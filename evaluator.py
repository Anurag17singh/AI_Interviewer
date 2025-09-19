import os
from typing import Dict
from llm_model import ask_llm
import streamlit as st
from typing import List
from sentence_transformers import SentenceTransformer, util

# helper to build rubrics
def make_rubric(points, criteria):
    return {"points": points, "criteria": criteria}

# QUESTION_BANK (30 Qs) - compact rubrics
QUESTION_BANK = [
    {
        "id": "q01",
        "title": "VLOOKUP vs INDEX-MATCH",
        "prompt": "How would you use Excel to look up a value from a table on another sheet? Explain both VLOOKUP and INDEX-MATCH and when you'd prefer one over the other.",
        "rubric": make_rubric(10, [
            {"k": "vlookup", "weight": 3, "desc": "Mentions VLOOKUP syntax"},
            {"k": "index", "weight": 3, "desc": "Mentions INDEX and MATCH"},
            {"k": "left lookup", "weight": 2, "desc": "Recognizes INDEX-MATCH can look left"},
            {"k": "exact", "weight": 2, "desc": "Covers exact vs approximate matches"},
        ])
    },
    {
        "id": "q02",
        "title": "Pivot Tables - monthly sales",
        "prompt": "Describe steps to build a pivot table to calculate monthly sales by product and how to filter to top 5 products.",
        "rubric": make_rubric(8, [
            {"k":"pivot", "weight":4, "desc":"Creates pivot, drags fields"},
            {"k":"top 5","weight":2,"desc":"Uses Top N filter or Rank+filter"},
            {"k":"group", "weight":2, "desc":"Groups dates into months"}
        ])
    },
    # ... remaining Qs added programmatically below
]

additional_qs = [
    ("q03", "Percent change formula and divide-by-zero handling",
     "Explain how you'd write a formula to compute percent change between two columns, and how you'd handle divide by zero.",
     6,
     [{"k":"(new-old)/old","weight":3,"desc":"Correct percent change formula"},
      {"k":"iferror","weight":3,"desc":"Handles divide-by-zero using IFERROR or IF(old=0)"}]),

    ("q04", "Conditional formatting",
     "How would you highlight cells in a column that are above average using conditional formatting?",
     5,
     [{"k":"conditional formatting","weight":3,"desc":"Mentions conditional formatting menu"},
      {"k":"above average","weight":2,"desc":"Selects 'Above Average' or formula rule"}]),

    ("q05", "SUMIFS vs SUMIF",
     "When would you use SUMIFS instead of SUMIF? Show an example.",
     5,
     [{"k":"sumifs","weight":3,"desc":"Understands multiple criteria"},
      {"k":"example","weight":2,"desc":"Gives a clear example"}]),

    ("q06", "Text functions (LEFT, RIGHT, MID)",
     "How to extract year from YYYY-MM-DD strings and combine first and last names into full name?",
     5,
     [{"k":"left","weight":2,"desc":"Uses LEFT/MID/RIGHT correctly"},
      {"k":"concatenate","weight":3,"desc":"Uses & or CONCAT/CONCATENATE"}]),

    ("q07", "Data validation",
     "How to restrict cell inputs to a list of allowed values and show a custom error message?",
     4,
     [{"k":"data validation","weight":3,"desc":"Uses Data Validation -> List"},
      {"k":"error message","weight":1,"desc":"Custom error prompt"}]),

    ("q08", "INDEX with multiple criteria",
     "Explain how to look up a value matching two conditions (e.g., product and month).",
     7,
     [{"k":"index","weight":3,"desc":"INDEX usage"},
      {"k":"match","weight":2,"desc":"MATCH with concatenation or array"},
      {"k":"array","weight":2,"desc":"Uses array formula or newer XLOOKUP"}]),

    ("q09", "XLOOKUP basics",
     "How does XLOOKUP differ from VLOOKUP? When would you prefer it?",
     6,
     [{"k":"xlookup","weight":4,"desc":"Describes XLOOKUP advantages"},
      {"k":"default","weight":2,"desc":"Mentions default exact match and return if not found"}]),

    ("q10", "Remove duplicates vs unique",
     "How to get unique list of customers and how to remove duplicates in-place?",
     4,
     [{"k":"unique","weight":2,"desc":"Mentions UNIQUE function or Remove Duplicates"},
      {"k":"remove duplicates","weight":2,"desc":"Explains 'Remove Duplicates' action"}]),

    ("q11", "Power Query basics",
     "When would you use Power Query vs formulas? Describe a simple ETL step in Power Query.",
     6,
     [{"k":"power query","weight":3,"desc":"When to use PQ"},
      {"k":"steps","weight":3,"desc":"Transform step example"}]),

    ("q12", "Charts - combo chart",
     "How to make a chart with bars for volume and a line for moving average?",
     5,
     [{"k":"combo chart","weight":3,"desc":"Uses secondary axis or combo chart"},
      {"k":"moving average","weight":2,"desc":"Shows how to compute moving average"}]),

    ("q13", "Named ranges",
     "What are named ranges and when are they useful?",
     4,
     [{"k":"named range","weight":3,"desc":"Defines named range"},
      {"k":"useful","weight":1,"desc":"Explains use-cases"}]),

    ("q14", "ARRAY formulas",
     "Explain array formulas and give an example where they help (pre-dynamic arrays).",
     6,
     [{"k":"array","weight":3,"desc":"Mentions CSE or dynamic arrays"},
      {"k":"example","weight":3,"desc":"Shows a use-case"}]),

    ("q15", "Conditional aggregation",
     "How to compute average of sales for a category excluding zeros?",
     5,
     [{"k":"averageif","weight":3,"desc":"Uses AVERAGEIF/IFS"},
      {"k":"exclude zeros","weight":2,"desc":"Shows criteria to exclude zeros"}]),

    ("q16", "Goal Seek",
     "Describe how you'd use Goal Seek to find the required discount to hit a revenue target.",
     4,
     [{"k":"goal seek","weight":3,"desc":"Explains Goal Seek steps"},
      {"k":"target","weight":1,"desc":"Sets cell and target value"}]),

    ("q17", "Solver",
     "When to use Solver vs Goal Seek? Give an example for optimizing allocation.",
     6,
     [{"k":"solver","weight":4,"desc":"Explains constraint optimization"},
      {"k":"example","weight":2,"desc":"Allocation example"}]),

    ("q18", "DATE functions",
     "How to get end of month date from a date field?",
     4,
     [{"k":"eomonth","weight":3,"desc":"Uses EOMONTH or other approach"},
      {"k":"example","weight":1,"desc":"Shows usage"}]),

    ("q19", "TIME functions and durations",
     "How to sum durations (hh:mm) and handle >24 hours?",
     4,
     [{"k":"time format","weight":2,"desc":"Explains time serial numbers"},
      {"k":"format","weight":2,"desc":"Custom format [h]:mm"}]),

    ("q20", "IF with multiple conditions",
     "Explain how to write nested IFs vs IFS and give a simple example.",
     5,
     [{"k":"ifs","weight":3,"desc":"Uses IFS or nested IF"},
      {"k":"example","weight":2,"desc":"Shows clean example"}]),

    ("q21", "MATCH types",
     "Explain the different match types in MATCH and pitfalls with unsorted data.",
     4,
     [{"k":"match types","weight":3,"desc":"-1,0,1 explanation"},
      {"k":"pitfalls","weight":1,"desc":"Sorted vs unsorted data"}]),

    ("q22", "FILTER function",
     "How to filter records from a table where region='APAC' and sales>1000?",
     4,
     [{"k":"filter","weight":3,"desc":"Uses FILTER"},
      {"k":"criteria","weight":1,"desc":"Shows criteria composition"}]),

    ("q23", "Statistical functions",
     "How to compute median and explain when median is preferred over mean?",
     4,
     [{"k":"median","weight":2,"desc":"Uses MEDIAN"},
      {"k":"preference","weight":2,"desc":"Outlier robustness"}]),

    ("q24", "Data types and errors",
     "Explain common Excel error types (#N/A, #DIV/0!, #VALUE!) and how to handle them.",
     5,
     [{"k":"errors","weight":3,"desc":"Identifies errors"},
      {"k":"handling","weight":2,"desc":"IFERROR/IFNA examples"}]),

    ("q25", "XLSX vs CSV",
     "When would you use CSV over XLSX? Mention pros/cons.",
     3,
     [{"k":"csv","weight":2,"desc":"Lightweight, plain text"},
      {"k":"xlsx","weight":1,"desc":"Formulas and formatting preserved"}]),

    ("q26", "Performance tips",
     "How to improve performance on large Excel files?",
     5,
     [{"k":"calc mode","weight":2,"desc":"Manual calc or reduce volatile functions"},
      {"k":"tables","weight":1,"desc":"Use tables/pivot caches"},
      {"k":"remove formatting","weight":2,"desc":"Avoid whole-column formulas"}]),

    ("q27", "XLOOKUP with multiple return",
     "Explain how to return multiple columns from a lookup (spill behavior).",
     4,
     [{"k":"xlookup","weight":3,"desc":"Uses XLOOKUP with return array"},
      {"k":"spill","weight":1,"desc":"Mentions dynamic array spill"}]),

    ("q28", "Protect sheets",
     "How to protect a worksheet and allow certain cells to be editable?",
     3,
     [{"k":"protect","weight":2,"desc":"Protect sheet steps"},
      {"k":"allow","weight":1,"desc":"Unlock certain cells first"}]),

    ("q29", "Importing data",
     "How to import data from a web URL and keep it refreshed?",
     4,
     [{"k":"get & transform","weight":3,"desc":"Power Query web import"},
      {"k":"refresh","weight":1,"desc":"Refresh schedule or manual"}]),

    ("q30", "Dynamic arrays - SORT and UNIQUE",
     "How to create a sorted unique list of products?",
     4,
     [{"k":"unique","weight":2,"desc":"UNIQUE function"},
      {"k":"sort","weight":2,"desc":"Wrap with SORT"}]),
]

# append
for t in additional_qs:
    QUESTION_BANK.append({
        "id": t[0], "title": t[1], "prompt": t[2], "rubric": make_rubric(t[3], t[4])
    })

# Semantic Embedding Helper

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def semantic_score(answer: str, rubric_keywords: list[str]) -> float:
    """Return max cosine similarity between answer and rubric keywords."""
    if not answer or not rubric_keywords:
        return 0.0
    model = get_embedder()
    emb_answer = model.encode([answer], convert_to_tensor=True)
    emb_keys = model.encode(rubric_keywords, convert_to_tensor=True)
    sims = util.cos_sim(emb_answer, emb_keys)
    return float(sims.max().item())

# Rule-based scorer (keyword matching)
def rule_based_score(answer: str, rubric: Dict) -> Dict:
    total_weight = sum(c["weight"] for c in rubric["criteria"])
    matched = []
    score = 0.0
    normalized = (answer or "").lower()
    for c in rubric["criteria"]:
        key = c["k"].lower()
        present = key in normalized
        if present:
            score += c["weight"]
            matched.append({"criterion": c["desc"], "matched": True, "weight": c["weight"]})
        else:
            matched.append({"criterion": c["desc"], "matched": False, "weight": c["weight"]})
    points = rubric["points"] * (score / total_weight if total_weight else 0)
    return {"points": round(points, 2), "max_points": rubric["points"], "details": matched}

# LLM-assisted grade (uses ask_llm which will choose Gemini/HF/local)
def llm_grade(answer: str, question_prompt: str) -> Dict:
    instruction = (
        "You are an expert Excel interviewer grader. Candidate answer:\n\n"
        f"{answer}\n\n"
        "Question: " + question_prompt + "\n\n"
        "Provide a concise numeric grade out of 10 on the first line (e.g. Grade: 7/10), "
        "then a 1-2 sentence justification, and up to 2 short suggestions for improvement as bullet points."
    )
    resp = ask_llm(instruction, max_tokens=200)
    return {"llm_feedback": resp}

@st.cache_data
def cached_llm_grade(answer: str, question_prompt: str) -> dict:
    return llm_grade(answer, question_prompt)

def needs_followup(rule_score: Dict, answer: str = "", rubric: Dict = None) -> bool:
    pts = rule_score.get("points", 0)
    maxp = rule_score.get("max_points", 1)
    frac = (pts / maxp) if maxp else 0

    # semantic similarity
    keywords = [c["k"] for c in (rubric["criteria"] if rubric else [])]
    sem = semantic_score(answer, keywords) if answer else 0.0

    # threshold from .env (default 0.6)
    threshold = float(os.getenv("FOLLOWUP_THRESHOLD", "0.6"))

    # trigger follow-up only if BOTH rule-based and semantic are weak
    return (frac < threshold) and (sem < 0.5)


def generate_followup(answer: str, question_prompt: str) -> str:
    instruction = (
        "You are an interviewer assistant. A candidate gave the following short/insufficient answer:\n\n"
        f"{answer}\n\n"
        "Original question:\n"
        f"{question_prompt}\n\n"
        "Create a single, concise follow-up question (one line) that asks the candidate to clarify or expand on the missing part(s). Do not be judgmental."
    )
    resp = ask_llm(instruction, max_tokens=80)
    return (resp or "").strip().split("\n")[0]
