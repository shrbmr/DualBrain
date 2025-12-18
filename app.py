import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator
from pydantic_ai import Agent
from typing import Optional, List
import json
import time
from datetime import datetime, timedelta
import re
import statistics

# ======================
# Config
# ======================

load_dotenv()
DEV_MODE = True

# ======================
# Models
# ======================

class EffortPlan(BaseModel):
    experience: Optional[str]
    goal: str
    steps: List[str]
    total_effort_minutes: int

    @field_validator("steps")
    def min_steps(cls, v):
        if len(v) < 3:
            raise ValueError("At least 3 steps required")
        return v

    @field_validator("total_effort_minutes")
    def positive_effort(cls, v):
        if v <= 0:
            raise ValueError("Effort must be positive")
        return v


class Task(BaseModel):
    date: str
    task_description: str
    allocated_minutes: int
    guidance: Optional[str] = None  # New field for guidance/materials


class FinalPlan(BaseModel):
    experience: Optional[str]
    goal: str
    steps: List[Task]
    total_effort_minutes: int
    scheduled_days: int
    user_duration_minutes: Optional[int]
    feasibility_comment: str

# ======================
# AI Agent
# ======================

agent = Agent(
    model="groq:llama-3.3-70b-versatile",
    system_prompt="""
You are an expert career planner.

You estimate effort ONLY.
You ignore deadlines completely.

Return VALID JSON ONLY.
No markdown. No commentary.
"""
)

# ======================
# Helpers
# ======================

def strip_time_constraints(text: str) -> str:
    return re.sub(
        r"in\s+\d+\s+(days?|weeks?|months?)",
        "",
        text,
        flags=re.IGNORECASE
    )


def parse_desired_duration_minutes(text: str, weekly_hours: float) -> Optional[int]:
    text = text.lower()

    if m := re.search(r"(\d+)\s*months?", text):
        return int(int(m.group(1)) * 4 * weekly_hours * 60)

    if m := re.search(r"(\d+)\s*weeks?", text):
        return int(int(m.group(1)) * weekly_hours * 60)

    if m := re.search(r"(\d+)\s*days?", text):
        return int(int(m.group(1)) * (weekly_hours / 7) * 60)

    return None


def run_ai(agent, prompt, label, logs):
    start = time.time()
    result = agent.run_sync(prompt)
    duration = round(time.time() - start, 2)
    output = result.output.strip()

    if output.startswith("```"):
        output = "\n".join(output.splitlines()[1:-1])

    if DEV_MODE:
        logs.append({
            "label": label,
            "prompt": prompt,
            "response": output,
            "duration": duration
        })

    return output


def deterministic_schedule(
    steps: List[str],
    total_minutes: int,
    start_date: datetime,
    daily_capacity_minutes: int
) -> List[Task]:
    tasks = []
    minutes_remaining = total_minutes
    current_date = start_date
    step_index = 0

    while minutes_remaining > 0:
        minutes_today = min(daily_capacity_minutes, minutes_remaining)
        tasks.append(Task(
            date=current_date.isoformat(),
            task_description=steps[step_index % len(steps)],
            allocated_minutes=minutes_today
        ))
        minutes_remaining -= minutes_today
        current_date += timedelta(days=1)
        step_index += 1

    return tasks


def generate_goal_overview(agent, user_context, logs):
    prompt = f"""
You are an expert career/goal planner.

User context:
{user_context}

Task:
1. List the essential knowledge, skills, or capabilities needed to achieve the goal.
2. Identify knowledge or skills the user likely does NOT know based on their experience.
3. Return valid JSON with two keys:
   {{
       "required_knowledge": [string],
       "skills_to_learn": [string]
   }}

Return JSON ONLY, no commentary.
"""
    raw = run_ai(agent, prompt, "Generic Goal Overview", logs)
    try:
        return json.loads(raw)
    except Exception:
        return {"required_knowledge": [], "skills_to_learn": []}


def generate_task_guidance(agent, task_description, logs):
    """
    Generates guidance, tips, or materials (online courses, books) for a task.
    """
    prompt = f"""
You are an expert career/goal planner.

Task description:
{task_description}

Provide practical guidance, learning resources, or tips to accomplish this task.  
Return JSON with a single key:
{{
    "guidance": string
}}

Return JSON ONLY.
"""
    raw = run_ai(agent, prompt, f"Task Guidance: {task_description}", logs)
    try:
        return json.loads(raw).get("guidance", "")
    except Exception:
        return ""

# ======================
# Sidebar
# ======================

with st.sidebar:
    start_date = st.date_input("Start Date", datetime.today())
    st.write("---")
    st.header("Availability per Day (hours)")

    weekday_hours = {
        "Monday": st.number_input("Monday", 0.0, 24.0, 4.0),
        "Tuesday": st.number_input("Tuesday", 0.0, 24.0, 4.0),
        "Wednesday": st.number_input("Wednesday", 0.0, 24.0, 4.0),
        "Thursday": st.number_input("Thursday", 0.0, 24.0, 4.0),
        "Friday": st.number_input("Friday", 0.0, 24.0, 4.0),
        "Saturday": st.number_input("Saturday", 0.0, 24.0, 0.0),
        "Sunday": st.number_input("Sunday", 0.0, 24.0, 0.0),
    }

weekly_hours = sum(weekday_hours.values())
daily_capacity_minutes = int(max(weekday_hours.values()) * 60)

# ======================
# Main UI
# ======================
st.set_page_config(
    page_title="Planimal – Logical Goal Planner",
    layout="wide"  # wide layout increases content width
)

st.title("Planimal – Logical Goal Planner")

user_input = st.text_area(
    "Enter your goal, experience, constraints, preferences, and any other context",
    "I am an experienced Business Analyst. I want to be AI PM in 6 days."
)

if st.button("Generate Plan"):
    with st.spinner("Planning logically..."):
        dev_logs = []

        # ---- Step 0: Generic Goal Overview
        overview = generate_goal_overview(agent, user_input, dev_logs)

        st.markdown("### 🧭 Overview")
        st.markdown("""
**Explanation:**  
- **Essential knowledge / capabilities:** Things you already need to know or understand deeply to succeed in this goal.  
- **Skills or areas to learn:** Knowledge gaps or practical skills you likely need to acquire to achieve the goal effectively.
""")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ Essential knowledge / capabilities")
            for item in overview["required_knowledge"]:
                st.markdown(f"- {item}")
        with col2:
            st.markdown("#### 🎯 Skills or areas to learn")
            for item in overview["skills_to_learn"]:
                st.markdown(f"- {item}")

        # ---- Step 1: Effort estimation
        clean_input = strip_time_constraints(user_input)
        effort_plans = []

        for i in range(5):
            raw = run_ai(
                agent,
                f"""
Estimate the TOTAL EFFORT required.

Ignore timelines and deadlines.

Schema:
{{
  "experience": string | null,
  "goal": string,
  "steps": [string, string, string],
  "total_effort_minutes": number
}}

USER CONTEXT:
{clean_input}
""",
                f"Effort Gen #{i+1}",
                dev_logs
            )

            try:
                effort_plans.append(EffortPlan(**json.loads(raw)))
            except Exception:
                continue

        if len(effort_plans) < 2:
            st.error("Could not reliably estimate effort. Try again.")
            st.stop()

        locked_effort = int(statistics.median(
            p.total_effort_minutes for p in effort_plans
        ))

        base_plan = effort_plans[0]
        base_plan.total_effort_minutes = locked_effort

        # ---- Step 2: Deterministic scheduling
        tasks = deterministic_schedule(
            base_plan.steps,
            base_plan.total_effort_minutes,
            start_date,
            daily_capacity_minutes
        )

        # ---- Step 2b: Add guidance/materials per task
        for task in tasks:
            task.guidance = generate_task_guidance(agent, task.task_description, dev_logs)

        scheduled_days = len(tasks)

        # ---- Step 3: Feasibility
        user_duration_minutes = parse_desired_duration_minutes(
            user_input,
            weekly_hours
        )

        if user_duration_minutes is None:
            feasibility = "No desired timeline provided"
        elif user_duration_minutes >= locked_effort:
            feasibility = "Feasible"
        else:
            deficit = locked_effort - user_duration_minutes
            feasibility = f"Infeasible – short by {deficit/60:.1f} hours"

        final_plan = FinalPlan(
            experience=base_plan.experience,
            goal=base_plan.goal,
            steps=tasks,
            total_effort_minutes=locked_effort,
            scheduled_days=scheduled_days,
            user_duration_minutes=user_duration_minutes,
            feasibility_comment=feasibility
        )

        # ---- Step 4: Plan Summary
        st.markdown(f"""
### ℹ️ Plan Summary

**Intrinsic effort required:** {locked_effort/60:.1f} hours  
**Scheduled duration (capacity-based):** {scheduled_days} days  
**Feasibility:** **{feasibility}**
""")

        st.dataframe(
            [
                {
                    "Date": t.date,
                    "Task": t.task_description,
                    "Minutes": t.allocated_minutes,
                    "Guidance / Material": t.guidance
                } for t in final_plan.steps
            ],
            use_container_width=True
        )

        # ---- Step 5: Dev logs
        if DEV_MODE:
            st.divider()
            st.subheader("🛠 Dev Trace")
            for log in dev_logs:
                with st.expander(log["label"]):
                    st.code(log["prompt"])
                    st.code(log["response"], language="json")
