# 👨‍🍳 RecipeGPT: The AI Sous-Chef

> **"Hungry? Let the neural network decide dinner."**

**RecipeGPT** is a fine-tuned GPT-2 model that doesn't just write text—it *understands* the structure of cooking. Trained on the **3A2M Extended Dataset**, it creates cohesive, structured recipes complete with ingredient lists and step-by-step instructions based on your chosen title and genre.

---

## 🍽️ The Menu (Table of Contents)

* [🥘 What's Cooking?](#-whats-cooking-overview)
* [🧠 The Secret Sauce (Methodology)](#-the-secret-sauce-methodology)
* [🔪 Mise en Place (Installation)](#-mise-en-place-installation)
* [🔥 Let's Cook (Usage)](#-lets-cook-usage)
* [📈 Nutrition Facts (Training Stats)](#-nutrition-facts-training-stats)
* [🍰 Sample Serving](#-sample-serving)
* [🔮 Future Specials (Roadmap)](#-future-specials-roadmap)

---

## 🥘 What's Cooking? (Overview)

Most language models ramble. RecipeGPT follows a strict culinary protocol. By injecting custom control tokens into the vocabulary, we forced the model to learn the specific syntax of a recipe card.

**Capabilities:**

* **Genre Conditioning:** Can distinguish between `Bakery`, `Sides`, `Non-Veg`, and more.
* **Structured Output:** Generates ingredients first, then directions—never mixing them up.
* **Creativity:** Uses sampling to invent unique (and sometimes wild) dishes.

---

## 🧠 The Secret Sauce (Methodology)

To transform GPT-2 into a chef, we flattened recipes into a linear format using **Special Control Tokens**. This allows the model to "know" which part of the recipe it is currently generating.

### The Sequence Format

The model sees a recipe as one long string:

```text
<|startofrecipe|> <|genre|>DESSERT <|title|>CHOCOLATE CAKE <|ingredients|> Flour, Sugar... <|steps|> Mix... <|endofrecipe|>
```

### 🔑 Special Token Vocabulary

*Note: These tokens are added to the tokenizer to prevent them from being split.*

| Token ID | Symbol | Function |
| --- | --- | --- |
| **BOS** | `<|startofrecipe|>` | Start of recipe |
| **GENRE** | `<|genre|>` | Genre of the recipe |
| **TITLE** | `<|title|>` | Title of the recipe |
| **INGR** | `<|ingredients|>` | Ingredients section |
| **STEPS** | `<|steps|>` | Steps section |
| **EOS** | `<|endofrecipe|>` | End of recipe |

---

## 🔪 Mise en Place (Installation)

Get your environment ready in 3 steps.

**1. Clone the Kitchen**

```bash
git clone https://github.com/yourusername/recipegpt.git
cd recipegpt
```

**2. Gather Ingredients (Dependencies)**

```bash
pip install transformers datasets torch accelerate evaluate
```

**3. Download Weights**

* *Option A:* Train it yourself using `train.py`.
* *Option B:* Place your pre-trained `pytorch_model.bin` and `tokenizer.json` in the `./recipe-gpt2-model` folder.

---

## 🔥 Let's Cook (Usage)

Here is the Python recipe to run the generator. We use **Top-K** and **Top-P sampling** to ensure the recipes are creative but coherent.

```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# --- 1. Load the Sous-Chef ---
model_path = "./recipe-gpt2-model"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Chef on {device}...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)

# --- 2. The Cooking Function ---
def generate_recipe(title, genre):
    # Construct the strict prompt structure
    prompt = f"<|startofrecipe|><|genre|>{genre}<|title|>{title}<|ingredients|>"
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate (Temperature 1.0 = creative, 0.7 = safe)
    outputs = model.generate(
        input_ids, 
        max_length=512, 
        do_sample=True, 
        top_k=50, 
        top_p=0.95,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode and clean
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return generated

# --- 3. Order Up! ---
print(generate_recipe("Spicy Garlic Potatoes", "Sides"))
```

---

## 📈 Nutrition Facts (Training Stats)

This model was fine-tuned on a T4 GPU using Mixed Precision to save memory.

| Metric | Value | Meaning |
| --- | --- | --- |
| **Dataset Size** | 20,000 | Subset of 3A2M Extended Dataset |
| **Perplexity** | **8.90** | Low perplexity = High model confidence |
| **Precision** | FP16 | Mixed Precision for faster training |
| **Optimizer** | AdamW | With linear learning rate decay |

---

## 🍰 Sample Serving

Here is an actual raw output from the validation set:

> **📝 Recipe Card**  
> **Genre:** `Sides`  
> **Title:** `Spicy Garlic Potatoes`  
> **Ingredients:**
> * 3 large potatoes, cubed
> * 1/4 cup olive oil
> * Garlic powder, ground sage, paprika
> * Soy sauce, water, salt
> 
> **Directions:**
> 1. Heat oil in a Dutch oven over medium heat.
> 2. Add the potatoes and spices; mix well to coat.
> 3. Add soy sauce and water, then bring to a boil.
> 4. Cover and simmer until potatoes are tender.
> 5. Sprinkle with fresh sage and serve hot.

---

## 🔮 Future Specials (Roadmap)

* [ ] **Constraint Beam Search:** Force the model to use *only* ingredients currently in your fridge.
* [ ] **Calorie Estimator:** Add a regression head to predict nutritional value.
* [ ] **Full Dataset Training:** Scale up from 20k to the full 2M recipe dataset.

---

**Bon Appétit!** 👨‍🍳  
*Licensed under MIT.*
