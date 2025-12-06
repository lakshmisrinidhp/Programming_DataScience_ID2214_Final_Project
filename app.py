import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------
# 1. Model & tokenizer setup
# -------------------------

MODEL_ID = "lakshmisrinidh/merged_llama_model"  # merged CPU-friendly model on HF

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,      # CPU
    low_cpu_mem_usage=True,
)
model.to(device)
model.eval()


# -------------------------
# 2. Utility: clean meta text from model output
# -------------------------

def clean_response(full_text: str) -> str:
    """
    Removes system/user/assistant headers and fake date/knowledge cutoff lines
    so that only the useful assistant answer is shown.
    """
    text = full_text.strip()
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        low = stripped.lower()
        if low.startswith("system"):
            continue
        if low.startswith("user"):
            continue
        if low.startswith("assistant"):
            continue
        if "Cutting Knowledge Date:" in stripped:
            continue
        if "Today Date:" in stripped:
            continue

        cleaned_lines.append(stripped)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned if cleaned else text


# -------------------------
# 3. Prompt builder & generation for fitness plans
# -------------------------

SYSTEM_PROMPT = """
You are a careful, supportive fitness coach.
You create safe, realistic 7-day fitness and lifestyle plans.
Always:
- Adapt to the user's age, goal, schedule, equipment and injuries.
- Be conservative and safe.
- Remind them to consult a doctor before starting any new program.
- Use simple, clear language.
"""

def build_prompt(
    age,
    height,
    weight,
    goal,
    days_per_week,
    equipment,
    level,
    activity_level,
    constraints,
    notes,
):
    equipment_str = ", ".join(equipment) if isinstance(equipment, list) else str(equipment)

    return f"""{SYSTEM_PROMPT}
User profile:
- Age: {age}
- Height: {height}
- Weight: {weight}
- Main goal: {goal}
- Training days per week: {days_per_week}
- Daily activity level: {activity_level}
- Available equipment: {equipment_str}
- Fitness level: {level}
- Injuries / health issues: {constraints}
- Extra preferences: {notes}
Task:
Create a practical, realistic 7-day fitness and lifestyle plan.
Include:
1. A weekly overview (which days to train and which to rest).
2. For each training day: exercises with sets and reps or minutes, plus warm-up and cool-down ideas.
3. Simple daily movement tips (e.g. steps, breaks from sitting).
4. Basic nutrition guidance that matches the goal and is easy to follow.
5. Safety reminders and how to progress in future weeks.
"""


def generate_plan(
    age,
    height,
    weight,
    goal,
    days_per_week,
    equipment,
    level,
    activity_level,
    constraints,
    notes,
    temperature,
    max_new_tokens,
):
    prompt = build_prompt(
        age=age,
        height=height,
        weight=weight,
        goal=goal,
        days_per_week=int(days_per_week),
        equipment=equipment,
        level=level,
        activity_level=activity_level,
        constraints=constraints,
        notes=notes,
    )

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=0.9,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = clean_response(full_text)

    # If the model echoes the prompt, strip it off
    if cleaned.startswith(prompt):
        cleaned = cleaned[len(prompt):].strip()

    return cleaned


# --- helpers for loading state on the button ---

def set_button_loading():
    """Set button to loading label & disable it."""
    return gr.update(value="Generating… ⏳", interactive=False)

def reset_button():
    """Reset button label after generation."""
    return gr.update(value="Generate my 7-day plan", interactive=True)


# -------------------------
# 4. Gradio UI: Fitness plan generator
# -------------------------

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
)

with gr.Blocks(
    theme=theme,
    css="""
    /* Make all text dark so it's visible on white background */
    body, .prose *, .gr-markdown, .gr-markdown * {
        color: #111 !important;
    }
    #chat-container, #side-panel {
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        padding: 12px;
        background: #ffffff;
    }
    /* Hide any top tabs/navigation bar if Gradio creates them */
    div[data-testid="tab-nav"] {
        display: none !important;
    }
    """
) as demo:
    gr.Markdown(
        f"""
        <div style="text-align: center; margin-bottom: 0.5rem;">
            <h1 style="font-size: 2rem; margin-bottom: 0.25rem;">
                 Personalised Fitness Plan Generator
            </h1>
            <p style="font-size: 0.95rem;">
                Powered by your fine-tuned LLaMA model: <code>{MODEL_ID}</code> (CPU inference)
            </p>
            <p style="font-size: 0.85rem;">
                This app is for educational purposes only and does not provide medical advice.<br>
                Always consult a healthcare professional before starting a new exercise or diet program.
            </p>
        </div>
        """
    )

    with gr.Row():
        # Left panel: inputs
        with gr.Column(scale=3, elem_id="chat-container"):
            gr.Markdown("### Enter your details")

            with gr.Row():
                age = gr.Number(label="Age", value=25)
                level = gr.Dropdown(
                    ["Beginner", "Intermediate", "Advanced"],
                    label="Fitness level",
                    value="Beginner",
                )

            with gr.Row():
                height = gr.Textbox(label="Height (e.g. 175 cm)", value="")
                weight = gr.Textbox(label="Weight (e.g. 70 kg)", value="")

            goal = gr.Dropdown(
                ["Lose fat", "Build muscle", "Improve cardio fitness", "General health / maintain"],
                label="Main goal",
                value="General health / maintain",
            )

            with gr.Row():
                days_per_week = gr.Slider(
                    minimum=1, maximum=7, step=1,
                    label="Training days per week",
                    value=3,
                )
                activity_level = gr.Dropdown(
                    ["Sedentary", "Lightly active", "Moderately active", "Very active"],
                    label="Daily activity level",
                    value="Sedentary",
                )

            equipment = gr.CheckboxGroup(
                ["Bodyweight only", "Dumbbells", "Resistance bands", "Barbell / rack", "Full gym"],
                label="Available equipment (select all that apply)",
                value=["Bodyweight only"],
            )

            constraints = gr.Textbox(
                label="Injuries / health issues (optional)",
                placeholder="e.g. knee pain, back issues, asthma...",
                lines=2,
            )

            notes = gr.Textbox(
                label="Extra preferences (optional)",
                placeholder="e.g. I hate running, prefer short workouts, like training in the morning...",
                lines=2,
            )

            with gr.Accordion("Advanced settings", open=False):
                temperature = gr.Slider(
                    minimum=0.4,
                    maximum=1.4,
                    value=0.8,
                    step=0.1,
                    label="Temperature (creativity)",
                    info="Lower = more focused, Higher = more diverse",
                )
                max_new_tokens = gr.Slider(
                    minimum=128,
                    maximum=768,
                    value=384,
                    step=32,
                    label="Max new tokens",
                    info="Maximum length of the generated plan",
                )

            generate_btn = gr.Button("Generate my 7-day plan", variant="primary")
            output = gr.Markdown(label="Your personalised plan")

        # Right panel: instructions & explanation
        with gr.Column(scale=2, elem_id="side-panel"):
            gr.Markdown(
                """
                ### How to use this app
                1. Fill in your age, fitness level, goal and schedule.  
                2. Select the equipment you have access to.  
                3. Add any injuries or preferences.  
                4. Click **“Generate my 7-day plan”**.
                The model will create:
                - A 7-day training and rest schedule  
                - Exercises for each training day  
                - Simple movement tips for daily life  
                - Basic nutrition guidance  
                ### Important
                - This is **not** medical advice.  
                - If you have any health conditions, ask a doctor before following any plan.  
                - Start easier than you think you need, and progress slowly.
                ### Under the hood
                - The model is a fine-tuned LLaMA variant.  
                - It was trained with Parameter-Efficient Fine-Tuning (LoRA / QLoRA) on instructions.  
                - This Space runs inference on **CPU only** using the merged model weights.
                """
            )

    # --- Button behavior with loading state ---
    (
        generate_btn
        .click(                     # first: set button to "loading"
            fn=set_button_loading,
            inputs=None,
            outputs=generate_btn,
        )
        .then(                      # second: run the model
            fn=generate_plan,
            inputs=[
                age,
                height,
                weight,
                goal,
                days_per_week,
                equipment,
                level,
                activity_level,
                constraints,
                notes,
                temperature,
                max_new_tokens,
            ],
            outputs=[output],
        )
        .then(                      # finally: reset button text & enable it again
            fn=reset_button,
            inputs=None,
            outputs=generate_btn,
        )
    )


if __name__ == "__main__":
    demo.queue().launch()
