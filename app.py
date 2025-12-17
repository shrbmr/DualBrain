import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field
from pydantic_ai import Agent
from typing import Optional, List
import json

load_dotenv()

# Pydantic model
class Plan(BaseModel):
    goal: str
    steps: List[str]

    # Total active work effort (not calendar time)
    total_effort_minutes: int = Field(..., gt=0)

    # Derived calendar estimates
    fulltime_days: int = Field(..., gt=0)   # 8 hrs/day
    parttime_days: int = Field(..., gt=0)   # 4 hrs/day avg

    # Optional user-provided duration converted to effort minutes
    user_duration_minutes: Optional[int] = Field(None, gt=0)

    feasibility_comment: Optional[str] = None

# AI Agent (no result_type!)
    
agent = Agent(
    model="groq:llama-3.3-70b-versatile",
    system_prompt="""
You are a detailed-oriented mentor and coach.
- Read the user's goal.
- Estimate TOTAL EFFORT time in minutes (not calendar duration).
- Convert that effort into:
  - fulltime days (8–10 hrs/day)
  - parttime days (2–6 hrs/day)
- If the user mentions a duration in their goal, based on defined fulltime and parttime days, convert in to minutes and compare it with your estimates.
- Mention in "feasibility_comment" whether the user's duration is realistic.
- Return ONLY a JSON object matching this schema:
{
  "goal": "<string>",
  "steps": ["<string>", ...],
  "total_effort_minutes": <integer>,
  "fulltime_days": <integer>,
  "parttime_days": <integer>,
  "user_duration_minutes": <integer or null>,
  "feasibility_comment": "<string or null>"
}
- And explicitly state:
    - fulltime_days = total_effort_minutes / (8 * 60), rounded up
    - parttime_days = total_effort_minutes / (4 * 60), rounded up
- Do not include any text outside the JSON.
"""
)

# Helpers

# Streamlit UI

st.title("Dual Brain - Plan Generator")
goal = st.text_input("Enter your goal")

if st.button("Generate Plan"):
    if goal:
        with st.spinner("Thinking..."):
            # Get raw AI response
            raw_response = agent.run_sync(f"Create a practical plan for this goal: {goal}")

            # Safer parsing
            try:
                import json
                plan_data = json.loads(raw_response.output)
                plan = Plan(**plan_data)
            except (json.JSONDecodeError, ValidationError) as e:
                st.error(f"Failed to parse AI response: {e}")
            else:
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

                if plan.user_duration_minutes:
                    st.subheader("User Suggested Duration")
                    st.write(plan.user_duration_minutes)

                if plan.feasibility_comment:
                    st.subheader("Feasibility Comment")
                    st.write(plan.feasibility_comment)
    else:
        st.warning("Please enter a goal!")
