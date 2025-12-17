import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator
from pydantic_ai import Agent
from typing import Optional, List
import json
from math import ceil

load_dotenv()

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
   - If the user has prior experience in a step’s domain, **reduce the step duration by up to 50%**, but never increase durations for experienced users.
   - Ensure each step has a **minimum duration of 120 minutes (2 hours)**.
   - The total plan should realistically reflect the time and effort needed to achieve the goal.

5. Compute the following summary metrics:
   - total_effort_minutes = sum of all step durations
   - fulltime_days = ceil(total_effort_minutes / (8*60))
   - parttime_days = ceil(total_effort_minutes / (4*60))

6. Convert any target duration provided by the user (in days, weeks, or months) into minutes and include it as user_duration_minutes. If no target duration is provided, set this to null.

7. Provide a feasibility comment:
   - "Target duration is feasible" if user_duration_minutes >= total_effort_minutes
   - "Target duration may require more effort" if user_duration_minutes < total_effort_minutes

8. Include the user’s experience or context in the `"experience"` field; if none is provided, set it to null.

9. Always return **valid JSON only** following this exact schema:
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

10. Do not include any text outside the JSON. Do not explain or justify anything outside the JSON.

11. Ensure that users with relevant prior experience always have **total_effort_minutes less than or equal to beginners** for the same goal, while keeping all step durations realistic.

12. Consider all information provided by the user—including experience, constraints, preferences, and timeline—when generating the plan. Make the plan practical, achievable, and personalized to the user’s context.

"""

agent = Agent(
    model="groq:llama-3.3-70b-versatile",
    system_prompt=system_prompt
)

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
                
                # Call AI agent
                plan = None
                for attempt in range(2):
                    raw_response = agent.run_sync(
                        f"Generate a practical plan based on this input: {input_text}"
                    )
                    raw_output = raw_response.output.strip()

                    # Remove Markdown code fences if present
                    if raw_output.startswith("```"):
                        raw_output = "\n".join(raw_output.splitlines()[1:-1])

                    if raw_output:
                        try:
                            plan_data = json.loads(raw_output)
                            plan = Plan(**plan_data)
                            break
                        except json.JSONDecodeError:
                            continue  # retry once

                if not plan:
                    st.error("AI returned empty or invalid response. Please provide more context or rephrase your input.")
                else:
                    # ----------------------
                    # Display structured output
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

            except ValidationError as ve:
                st.error(f"No-BS validation error: {str(ve)}")
    else:
        st.warning("Please enter your goal and context!")
