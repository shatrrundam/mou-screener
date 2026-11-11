"""
MoU Opportunity Screener
Takes an MoU (PDF, DOCX or TXT), extracts the key terms, assesses them against
10-16 configurable parameters, and returns a weighted opportunity score.

Run:  streamlit run app.py
"""

import io
import json
import os

import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# 1. Key terms the MoU template asks the incoming party to fill in.
#    The extractor looks for exactly these.
# ---------------------------------------------------------------------------
KEY_TERMS = [
    ("parties", "Parties", "Full legal names of all parties, registration details and authorised signatories"),
    ("partner_profile", "Partner Profile", "Partner's background, sector, size, track record and relevant past partnerships"),
    ("objectives", "Purpose and Objectives", "Why the partnership exists and what it aims to achieve"),
    ("scope", "Scope of Activities", "Specific activities, programmes or services covered"),
    ("deliverables_kpis", "Deliverables and KPIs", "Measurable outputs, outcomes, targets and timelines"),
    ("roles", "Roles and Responsibilities", "What each party commits to do, including staff and resources"),
    ("financials", "Financial Terms", "Funding, cost sharing, payments, in-kind contributions and who bears what cost"),
    ("duration", "Duration and Renewal", "Effective date, term, renewal conditions"),
    ("termination", "Termination", "Termination rights, notice periods and exit obligations"),
    ("exclusivity", "Exclusivity", "Any exclusivity, non-compete or first-right-of-refusal terms"),
    ("data_confidentiality", "Confidentiality and Data Protection", "Handling of confidential information and personal or government data"),
    ("ip", "Intellectual Property", "Ownership and licensing of existing and newly created IP"),
    ("governance", "Governance and Reporting", "Steering committee, review meetings, reporting frequency and escalation"),
    ("liability", "Liability and Indemnity", "Liability caps, indemnities, insurance"),
    ("disputes", "Dispute Resolution and Governing Law", "Governing law, jurisdiction, arbitration or mediation"),
]

# ---------------------------------------------------------------------------
# 2. Default evaluation parameters (12, weights total 100). Editable in the app.
#    Knockout = a score of 1 or below forces escalation regardless of total.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = [
    ("Strategic alignment", "Objectives clearly support the ministry's mandate and national priorities", 12, False),
    ("Value and measurable outcomes", "Clear, measurable KPIs with targets and timelines that benefit the ministry or citizens", 12, False),
    ("Scope clarity", "Activities and boundaries are specific, with no open-ended commitments", 8, False),
    ("Partner credibility", "Partner has a verifiable track record and relevant capability", 10, False),
    ("Financial terms", "Costs and contributions are defined, balanced and affordable for the ministry", 10, False),
    ("Roles and responsibilities", "Each party's obligations and resources are explicit", 6, False),
    ("Governance and reporting", "Defined steering structure, review cadence and escalation path", 8, False),
    ("Duration and exit flexibility", "Reasonable term, renewal by mutual consent, ministry can exit with fair notice", 6, False),
    ("Exclusivity", "No exclusivity, or exclusivity that is narrow, time-bound and justified", 6, False),
    ("Data protection and confidentiality", "Government and personal data protected in line with applicable law", 8, True),
    ("Intellectual property", "Ministry retains rights to IP and outputs created for it", 6, False),
    ("Liability and dispute resolution", "Ministry's liability is limited, local governing law, clear dispute mechanism", 8, True),
]

MAX_CHARS = 120_000  # keeps very long documents inside the model's context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def default_params_df() -> pd.DataFrame:
    return pd.DataFrame(
        DEFAULT_PARAMS,
        columns=["Parameter", "What good looks like", "Weight", "Knockout"],
    )


def read_document(name: str, data: bytes) -> str:
    name = name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)
    return data.decode("utf-8", errors="ignore")


def build_template_docx() -> bytes:
    doc = Document()
    doc.add_heading("Memorandum of Understanding", level=0)
    doc.add_paragraph(
        "Instructions for the proposing party: complete every section below. "
        "Write 'Not applicable' where a section does not apply. Incomplete "
        "sections are scored as not addressed."
    )
    for i, (_, label, hint) in enumerate(KEY_TERMS, start=1):
        doc.add_heading(f"{i}. {label}", level=1)
        doc.add_paragraph(f"[{hint}]")
    doc.add_heading("Signatures", level=1)
    doc.add_paragraph("Name, title, date and signature for each party.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def call_llm(api_key: str, model: str, prompt: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


def extraction_prompt(mou_text: str) -> str:
    fields = "\n".join(f'- "{key}": {label} ({hint})' for key, label, hint in KEY_TERMS)
    return f"""You are a government partnerships analyst reviewing a proposed MoU.

Extract the key terms below from the MoU. Return one JSON object whose keys are
exactly these field names:
{fields}

Each value must be an object with:
- "value": a concise factual summary of what the MoU says (max 60 words), or "Not specified"
- "clause": the clause or section reference where it appears, or ""

Only report what the document actually states. Do not infer or invent terms.

MoU TEXT:
\"\"\"
{mou_text[:MAX_CHARS]}
\"\"\"
"""


def assessment_prompt(terms: dict, params: pd.DataFrame) -> str:
    plist = "\n".join(
        f'- "{r.Parameter}": {r["What good looks like"]}' for _, r in params.iterrows()
    )
    return f"""You are a senior advisor to a government ministry evaluating whether to
sign a partnership MoU. Assess the extracted key terms against each parameter.

PARAMETERS (name: what good looks like):
{plist}

SCORING SCALE (integer):
5 = fully meets the standard, favourable to the ministry
4 = meets the standard with minor gaps
3 = adequate but with clear gaps or ambiguity
2 = weak, several gaps or terms tilted toward the partner
1 = unfavourable or risky for the ministry
0 = not addressed in the MoU

Return one JSON object:
{{
  "assessments": [
    {{"parameter": "<exact parameter name>", "score": <0-5>,
      "rationale": "<max 40 words>", "evidence": "<which key term or clause supports this>"}}
  ],
  "red_flags": ["<specific risk>", ...],
  "questions_for_partner": ["<clarification to request before signing>", ...]
}}

Include every parameter exactly once, using its exact name.

EXTRACTED KEY TERMS:
{json.dumps(terms, indent=2, ensure_ascii=False)}
"""


def score_opportunity(params: pd.DataFrame, assessments: list) -> tuple:
    by_name = {a.get("parameter", "").strip().lower(): a for a in assessments}
    rows, knockouts_failed = [], []
    for _, p in params.iterrows():
        a = by_name.get(str(p.Parameter).strip().lower(), {})
        try:
            s = int(a.get("score", 0))
        except (TypeError, ValueError):
            s = 0
        s = max(0, min(5, s))
        if bool(p.Knockout) and s <= 1:
            knockouts_failed.append(p.Parameter)
        rows.append({
            "Parameter": p.Parameter,
            "Weight": float(p.Weight),
            "Score (0-5)": s,
            "Weighted points": 0.0,
            "Rationale": a.get("rationale", "Not assessed"),
            "Evidence": a.get("evidence", ""),
        })
    df = pd.DataFrame(rows)
    total_weight = df["Weight"].sum() or 1
    df["Weighted points"] = (df["Weight"] / total_weight * df["Score (0-5)"] / 5 * 100).round(1)
    return round(df["Weighted points"].sum(), 1), df, knockouts_failed


def recommendation(score: float, knockouts_failed: list) -> tuple:
    if knockouts_failed:
        return "Escalate to legal before proceeding", "error"
    if score >= 75:
        return "Proceed", "success"
    if score >= 55:
        return "Proceed with conditions", "warning"
    return "Decline or renegotiate", "error"


def build_report(score, rec, knockouts, terms_df, score_df, red_flags, questions) -> str:
    lines = [
        "# MoU Opportunity Assessment",
        f"\n**Final score:** {score} / 100  ",
        f"**Recommendation:** {rec}",
    ]
    if knockouts:
        lines.append(f"\n**Knockout parameters failed:** {', '.join(knockouts)}")
    lines.append("\n## Scorecard\n")
    lines.append(score_df.to_markdown(index=False))
    lines.append("\n## Key terms\n")
    lines.append(terms_df.to_markdown(index=False))
    lines.append("\n## Red flags\n")
    lines += [f"- {r}" for r in red_flags] or ["- None identified"]
    lines.append("\n## Questions for the partner\n")
    lines += [f"- {q}" for q in questions] or ["- None"]
    return "\n".join(lines)


def get_default_key() -> str:
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return os.environ.get("GEMINI_API_KEY", "")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="MoU Opportunity Screener", layout="wide")
    st.title("MoU Opportunity Screener")
    st.caption(
        "Upload a partnership MoU. The agent extracts the key terms, assesses them "
        "against your evaluation parameters and returns a weighted opportunity score."
    )

    with st.sidebar:
        st.header("Settings")
        api_key = get_default_key()
        if api_key:
            st.caption("API key loaded securely from app secrets.")
        else:
            api_key = st.text_input("Gemini API key", type="password")
        model = st.text_input("Model", value="gemini-2.5-flash")
        st.divider()
        st.header("MoU template")
        st.write("Send this to the incoming party to fill in.")
        st.download_button(
            "Download MoU template (.docx)",
            data=build_template_docx(),
            file_name="MoU_Template.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    st.subheader("1. Evaluation parameters")
    st.write("Edit names, weights or knockouts. Keep between 10 and 16 parameters.")
    if "params" not in st.session_state:
        st.session_state.params = default_params_df()
    params = st.data_editor(
        st.session_state.params,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Weight": st.column_config.NumberColumn(min_value=0, max_value=100, step=1),
            "Knockout": st.column_config.CheckboxColumn(help="A score of 1 or below forces escalation"),
        },
        key="params_editor",
    ).dropna(subset=["Parameter"])
    params = params[params["Parameter"].astype(str).str.strip() != ""]
    n, total_w = len(params), params["Weight"].fillna(0).sum()
    st.caption(f"{n} parameters, total weight {total_w:g} (weights are normalised to 100).")

    st.subheader("2. Upload the MoU")
    uploaded = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"])

    if st.button("Assess opportunity", type="primary"):
        if not api_key:
            st.error("Add your Gemini API key in the sidebar.")
            return
        if not 10 <= n <= 16:
            st.error(f"Use between 10 and 16 parameters (you have {n}).")
            return
        if total_w <= 0:
            st.error("Weights must add up to more than zero.")
            return
        if not uploaded:
            st.error("Upload an MoU first.")
            return

        text = read_document(uploaded.name, uploaded.getvalue())
        if len(text.strip()) < 200:
            st.error("Could not read enough text. If this is a scanned PDF, upload a DOCX or text version.")
            return

        try:
            with st.spinner("Step 1 of 2: extracting key terms..."):
                terms = call_llm(api_key, model, extraction_prompt(text))
            with st.spinner("Step 2 of 2: assessing against parameters..."):
                result = call_llm(api_key, model, assessment_prompt(terms, params))
        except Exception as e:
            st.error(f"Model call failed: {e}")
            return

        score, score_df, knockouts = score_opportunity(params, result.get("assessments", []))
        rec, level = recommendation(score, knockouts)

        labels = {k: label for k, label, _ in KEY_TERMS}
        terms_df = pd.DataFrame([
            {"Key term": labels.get(k, k),
             "What the MoU says": (v or {}).get("value", "") if isinstance(v, dict) else str(v),
             "Clause": (v or {}).get("clause", "") if isinstance(v, dict) else ""}
            for k, v in terms.items()
        ])
        red_flags = result.get("red_flags", [])
        questions = result.get("questions_for_partner", [])

        st.subheader("3. Result")
        c1, c2 = st.columns([1, 2])
        c1.metric("Opportunity score", f"{score} / 100")
        getattr(c2, level)(f"Recommendation: {rec}")
        if knockouts:
            st.error("Knockout parameters failed: " + ", ".join(knockouts))

        st.markdown("**Scorecard**")
        st.dataframe(score_df, width="stretch", hide_index=True)

        with st.expander("Extracted key terms", expanded=False):
            st.dataframe(terms_df, width="stretch", hide_index=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Red flags**")
            for r in red_flags or ["None identified"]:
                st.write(f"- {r}")
        with c4:
            st.markdown("**Questions for the partner**")
            for q in questions or ["None"]:
                st.write(f"- {q}")

        st.download_button(
            "Download assessment report (.md)",
            data=build_report(score, rec, knockouts, terms_df, score_df, red_flags, questions),
            file_name="MoU_Assessment.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
