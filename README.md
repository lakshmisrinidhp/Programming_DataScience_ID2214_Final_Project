# Personalised Fitness Plan Generator (Fine-Tuned LLM)

URL for the UI for the fine-tuned LLM: https://huggingface.co/spaces/lakshmisrinidh/iris

This project builds a simple web application that generates a **7-day personalised fitness and lifestyle plan** based on a user’s profile.  
It is powered by **fine-tuned open-source Large Language Models (LLMs)** and runs on **CPU-only inference** inside a Hugging Face Space.

The goal of this project is to:
1. Fine-tune open-source LLMs  
2. Deploy them in a real UI  
3. Improve model performance and scalability  
4. Compare different models and discuss their results  

---

## Features

The app takes information from the user, including:

- Age  
- Height / Weight  
- Fitness goal  
- Training days per week  
- Daily activity level  
- Available equipment  
- Injuries or health issues  
- Extra preferences  

Then it generates a **simple plan**, including:

- Weekly overview  
- Exercises with sets/reps  
- Warm-ups and cool-downs  
- Movement suggestions  
- Basic nutrition ideas (optional)  
- Safety advice (optional)  

---

## Models Used

We fine-tuned **two different models**:

### 1. Llama-3.2-3B-Instruct (fine-tuned with PEFT)
- ~3 billion parameters  
- Larger and more capable  
- Slower on CPU  

### 2. Llama-3.2-1B-Instruct (fine-tuned with PEFT)
- ~1 billion parameters  
- Smaller and faster  
- Lower quality, but more responsive  

Both models were fine-tuned using **Unsloth**, with **LoRA / QLoRA** for efficient training.  
We then **merged** the weights to create a full 16-bit model suitable for CPU inference.

---

## Training Setup (Short Summary)

We used a **parameter-efficient fine-tuning approach** so we can train large models with limited hardware.

Key ideas:

- LoRA / QLoRA adapters  
- Mixed precision (4-bit, 8-bit)  
- Gradient checkpointing  
- Unsloth accelerated training  

We used an **instruction dataset** that helps the model follow tasks and output structured answers.

---

## Model-Centric Improvements

We explored ways to improve model performance by changing the **model configuration**:

- Tried **two model sizes** (1B vs 3B)  
- Adjusted **LoRA hyperparameters**  
- Trained the 1B model for **more steps**  
- Increased **max_new_tokens** during inference to allow longer answers  

These changes helped us explore differences in **quality, speed, and usability**.

---

## Data-Centric Improvements

We identified ways to improve results using better data:

- Combine more high-quality instruction datasets  
- Clean low-quality or duplicate samples  
- Build a small **custom dataset** with:
  - Fitness plans  
  - Safety notes  
  - Nutrition tips  
  - Back-friendly exercises  

Even **50–200 high-quality examples** could help the model become safer and more accurate.  
This was not fully implemented due to time but is a good future step.

---

## Evaluation of Models

We tested both models with the same user profile.

### 3B Model – Results

**Strengths**
- Clear weekly structure (training vs rest days)  
- Included warm-ups, cool-downs, and movement tips  
- Cardio-focused plans  

**Weaknesses**
- Ignored injury (back issues)  
- Recommended unsafe exercises (burpees, jumping jacks, high knees)  
- No nutrition or safety sections  
- Slower on CPU  

**Conclusion**  
Better structure and coaching tone, but **unsafe and slow**.

---

### 1B Model – Results

**Strengths**
- Faster, more responsive output  
- Simpler exercises (no jumping)  
- Beginner-friendly tone  

**Weaknesses**
- Repeated user prompt text  
- No clear weekly overview  
- Less cardio-focused  
- No nutrition, safety, or tracking sections  

**Conclusion**  
Much faster and usable, but **lower quality and less organized**.

---

## Speed on CPU

We deployed both models on **CPU-only hosting**.

Observations:

- The **1B model was significantly faster** than the 3B model  
- The app felt more responsive and interactive  
- The 3B model was slow when generating long plans  

This is expected because the 1B model has **3× fewer parameters**.  
CPU inference scales almost linearly with model size.

**Real-world takeaway:**  
For CPU-only deployment, **smaller models give a better user experience**, even if quality is lower.

---

## Summary Comparison Table

| Metric | 3B Model | 1B Model |
|---|---|---|
| Model size | Large | Small |
| Output quality | Higher | Lower |
| Structure | Better | Weak |
| Cardio focus | Strong | Weak |
| Safety for injuries | Poor | Slightly better |
| Nutrition section | Missing | Missing |
| Speed on CPU | Slow | Fast |
| User experience | Good ideas but slow | Fast but generic |

---

## Overall Conclusion

There is **no single best model**.

- The **3B model** gives better structure and more detailed plans, but ignores injuries and is slow.  
- The **1B model** is much faster and more responsive, but the plans are simpler and less organized.  

For a **real online app running on CPU**, the **1B model may be more practical**, even though it is less accurate.  

For **best plan quality**, a larger model or more domain-specific data would help.

---

## Future Improvements

Simple ideas for future work:

1. Create a small dataset of fitness plans and nutrition tips  
2. Use few-shot examples in the prompt  
3. Force the model to avoid unsafe movements  
4. Add real user feedback  
5. Quantize models to **4-bit** for faster inference  

These steps could make the system **safer, smarter, and more useful**.

---

## Tech Stack

- Python  
- Hugging Face Transformers  
- Unsloth  
- LoRA / QLoRA  
- Gradio (UI)  
- Torch  

---

## User Interface

We built a **simple web UI** using Gradio, where users:

1. Enter personal details  
2. Click a button  
3. Receive a personalised 7-day plan  

The UI is designed to be:

- Clean  
- Easy to use  
- CPU-friendly  


---

## Final Thoughts

This project shows how small open-source LLMs can be:

- Fine-tuned  
- Deployed  
- And adapted to a useful application  

It also shows the **trade-off between model size, quality, and speed**, especially when running on **CPU-only infrastructure**.

Even though the models are not perfect, this project demonstrates:

- Real LLM engineering  
- Real deployment  
- Real performance evaluation  

in a clear and accessible way.
