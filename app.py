import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator
from pydantic_ai import Agent
from typing import Optional, List
import json
from math import ceil
import time

load_dotenv()

DEV_MODE = True  # Set to False in production


# ----------------------
# Pydantic Model
# ----------------------
class Plan(BaseModel):
    experience: Optional[str] = None
    goal: str
    steps: List[str]
    total_effort_minutes: int
    fulltime_days: int
    parttime_days: int
    user_duration_minutes: Optional[int] = None
    feasibility_comment: Optional[str] = None

    @field_validator("total_effort_minutes")
    def effort_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("total_effort_minutes must be > 0")
        return v

    @field_validator("steps")
    def steps_minimum_three(cls, v):
        if len(v) < 3:
            raise ValueError("steps must contain at least 3 items")
        return v

# ----------------------
# AI Agent
# ----------------------
system_prompt = """
You are a detail-oriented mentor and coach. Your task is to generate a practical, achievable plan based on all the information provided by the user. Follow these instructions carefully:

1. Read all user input from a single input box, which may include:
   - Experience, background, and skills
   - Goal or objective
   - Constraints, preferences, or limitations
   - Target duration or timeline

2. Identify the skills, knowledge, or abilities the user already possesses and note which are relevant to the stated goal.

3. Generate a plan **only for missing skills, knowledge gaps, or steps required to achieve the goal**. Do not include steps for skills the user already has, unless they are necessary for mastery.

4. For each step in the plan:
   - Assign a realistic duration in minutes that reflects the effort required to gain competence in that skill or complete that task.
   - Adjust durations based on the user's prior experience: reduce durations for experienced users, but never increase them beyond what is reasonable.
   - Ensure each step has a **minimum duration of 120 minutes (2 hours)**.
   - Provide sufficient detail so the user understands **what exactly needs to be done in each step**.
   - Be conservative and realistic: avoid underestimating the effort required, but do not artificially inflate it.

5. Make sure the total plan is achievable, realistic, and fits within the user's target duration if possible.

6. Compute the following summary metrics:
   - total_effort_minutes = sum of all step durations
   - fulltime_days = ceil(total_effort_minutes / (8*60))
   - parttime_days = ceil(total_effort_minutes / (4*60))

7. Convert any target duration provided by the user (in days, weeks, or months) into minutes and include it as user_duration_minutes. If no target duration is provided, set this to null.

8. Provide a feasibility comment:
   - "Target duration is feasible" if user_duration_minutes >= total_effort_minutes
   - "Target duration may require more effort" if user_duration_minutes < total_effort_minutes

9. Include the user’s experience or context in the "experience" field; if none is provided, set it to null.

10. Always return **valid JSON only** following this exact schema:
{
  "experience": "<string or null>",
  "goal": "<string>",
  "steps": ["<string>", ...],
  "total_effort_minutes": <integer>,
  "fulltime_days": <integer>,
  "parttime_days": <integer>,
  "user_duration_minutes": <integer or null>,
  "feasibility_comment": "<string or null>"
}

11. Do not include any text outside the JSON. Do not explain or justify anything outside the JSON.

12. Ensure that users with relevant prior experience always have **total_effort_minutes less than or equal to beginners** for the same goal, while keeping all step durations realistic.

13. Consider all information provided by the user—including experience, constraints, preferences, and timeline—when generating the plan. Make the plan practical, achievable, and personalized to the user’s context.
"""


agent = Agent(
    model="groq:llama-3.3-70b-versatile",
    system_prompt=system_prompt
)

# ----------------------
# Helpers
# ----------------------

def build_aggregation_prompt(
    plans: list[Plan],
    original_user_prompt: str
) -> str:
    plans_json = [p.model_dump() for p in plans]

    return f"""
You are a senior mentor and reviewer.

You are given:
1) The ORIGINAL user request (ground truth)
2) Multiple alternative plans generated for the SAME request

Your task is to aggregate the plans into ONE safe, realistic, and achievable plan
that best satisfies the ORIGINAL user request.

IMPORTANT PRIORITY ORDER:
1. The user's stated goal, experience, constraints, and timeline
2. Consensus across multiple plans
3. Conservative realism over optimism

Rules:
1. Do NOT invent new steps outside the given plans unless absolutely required
   to satisfy the user's request.
2. Merge similar or overlapping steps.
3. Remove redundant or unnecessary steps.
4. Be conservative and realistic with total effort.
5. Ensure at least 3 steps.
6. Preserve the user's experience, goal, and target duration.
7. If plans disagree, choose the safer and more realistic option.
8. Return VALID JSON ONLY using the SAME schema as the original plans.
9. Do NOT include any explanation outside the JSON.

ORIGINAL USER REQUEST:
\"\"\"{original_user_prompt}\"\"\"

PLANS TO AGGREGATE:
{json.dumps(plans_json, indent=2)}
"""

def run_ai_call(agent, prompt: str, label: str, dev_logs: list):
    start_time = time.time()
    response = agent.run_sync(prompt)
    duration = round(time.time() - start_time, 2)

    output = response.output.strip()

    if output.startswith("```"):
        output = "\n".join(output.splitlines()[1:-1])

    if DEV_MODE:
        dev_logs.append({
            "label": label,
            "prompt": prompt,
            "response": output,
            "duration_seconds": duration,
        })

    return output

# ----------------------
# Streamlit UI
# ----------------------
st.title("Planimal - Your Blueprint for Goals")

user_input = st.text_area(
    "Enter your goal, experience, constraints, preferences, and any other context",
    "I am an experienced Business Analyst. I want to be AI PM in 2 months."
)

if st.button("Generate Plan"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                input_text = user_input.strip()
                dev_logs = []

                # ----------------------
                # 1) Generate multiple candidate plans (3 calls)
                # ----------------------
                plans = []
                MAX_RUNS = 3

                for i in range(MAX_RUNS):
                    raw_output = run_ai_call(
                        agent=agent,
                        prompt=f"Generate a practical plan based on this input: {input_text}",
                        label=f"Generation Call #{i+1}",
                        dev_logs=dev_logs
                    )

                    try:
                        plan_data = json.loads(raw_output)
                        validated_plan = Plan(**plan_data)
                        plans.append(validated_plan)
                    except (json.JSONDecodeError, ValidationError):
                        continue

                # ----------------------
                # 2) Safety check
                # ----------------------
                if len(plans) < 2:
                    st.error("AI responses were inconsistent. Please try again.")
                    st.stop()

                # ----------------------
                # 3) Aggregation call (4th call)
                # ----------------------
                aggregation_prompt = build_aggregation_prompt(
                    plans=plans,
                    original_user_prompt=input_text
                )

                raw_output = run_ai_call(
                    agent=agent,
                    prompt=aggregation_prompt,
                    label="Aggregation Call",
                    dev_logs=dev_logs
                )

                try:
                    aggregated_data = json.loads(raw_output)
                    plan = Plan(**aggregated_data)
                except (json.JSONDecodeError, ValidationError):
                    st.error("Failed to safely aggregate AI responses.")
                    st.stop()

                # ----------------------
                # 4) Deterministic recompute
                # ----------------------
                plan = plan.copy(
                    update={
                        "fulltime_days": ceil(plan.total_effort_minutes / (8 * 60)),
                        "parttime_days": ceil(plan.total_effort_minutes / (4 * 60)),
                        "feasibility_comment": (
                            "Target duration is feasible"
                            if plan.user_duration_minutes
                            and plan.user_duration_minutes >= plan.total_effort_minutes
                            else "Target duration may require more effort"
                        ),
                    }
                )

                # ----------------------
                # 5) Display final plan
                # ----------------------
                if plan.experience:
                    st.subheader("User Experience / Context")
                    st.write(plan.experience)

                st.subheader("Goal")
                st.write(plan.goal)

                st.subheader("Steps")
                for i, step in enumerate(plan.steps, 1):
                    st.write(f"{i}. {step}")

                st.subheader("Estimated Total Effort")
                st.write(f"{plan.total_effort_minutes:,} minutes")

                st.subheader("Estimated Timeline")
                st.write(f"Full-time (8 hrs/day): {plan.fulltime_days} days")
                st.write(f"Part-time (4 hrs/day): {plan.parttime_days} days")

                if plan.user_duration_minutes is not None:
                    st.subheader("User Suggested Duration")
                    st.write(plan.user_duration_minutes)

                if plan.feasibility_comment:
                    st.subheader("Feasibility Comment")
                    st.write(plan.feasibility_comment)

                # ----------------------
                # 6) DEV MODE: Observability UI
                # ----------------------
                if DEV_MODE:
                    st.divider()
                    st.subheader("🛠️ Dev Mode – AI Call Trace")

                    for log in dev_logs:
                        with st.expander(
                            f"{log['label']} — {log['duration_seconds']}s"
                        ):
                            st.markdown("**Prompt**")
                            st.code(log["prompt"], language="text")

                            st.markdown("**Response**")
                            st.code(log["response"], language="json")

            except ValidationError as ve:
                st.error(f"No-BS validation error: {str(ve)}")
    else:
        st.warning("Please enter your goal and context!")
