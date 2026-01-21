import gradio as gr
from transformers import pipeline, GPT2Tokenizer
import os
import torch



MODEL_DIR = "./recipe-gpt2-model"

if not os.path.exists(MODEL_DIR):
    print(f"Error: Model not found at {MODEL_DIR}")
    print("Please make sure the 'recipe-gpt-model' folder is in the same directory as app.py")
    exit()

print("Loading tokenizer to get special tokens...")
try:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()


BOS_TOKEN = tokenizer.bos_token
EOS_TOKEN = tokenizer.eos_token
PAD_TOKEN = tokenizer.pad_token
GENRE_TOKEN = "<|genre|>"
TITLE_TOKEN = "<|title|>"
INGREDIENTS_TOKEN = "<|ingredients|>"
STEPS_TOKEN = "<|steps|>"


print("Loading text-generation pipeline... (This may take a moment on a CPU)")
generator = pipeline(
    'text-generation', 
    model=MODEL_DIR, 
    tokenizer=MODEL_DIR,
    device=-1 # -1 forces CPU
)

print("Model loaded successfully on CPU.")




def create_recipe(genre, title):
    """
    This function will be called by the Gradio interface.
    It takes a genre and title, and generates *both* ingredients and steps.
    """
    if not title or not genre:
        return "Please enter both a title and a genre."
        
    print(f"Generating recipe for: {title} ({genre})")
    

    prompt_text = (
        f"{BOS_TOKEN}"
        f"{GENRE_TOKEN}{genre}"
        f"{TITLE_TOKEN}{title}"
        f"{INGREDIENTS_TOKEN}"
    )

    output = generator(
        prompt_text,
        max_new_tokens=500,
        temperature=1.0, 
        top_k=50,
        top_p=0.95,
        do_sample=True, 
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
        
    )

    
    generated_text = output[0]['generated_text']
    clean_text = generated_text.replace(prompt_text, "").replace(EOS_TOKEN, "").strip()
    
    
    clean_text = clean_text.replace(INGREDIENTS_TOKEN, f"\n--- INGREDIENTS ---\n")
    clean_text = clean_text.replace(STEPS_TOKEN, f"\n\n--- STEPS ---\n")
    
    return clean_text




GENRE_CHOICES = [
    "bakery",
    "drinks",
    "non-veg",
    "vegetables",
    "fast food",
    "cereals",
    "meals",
    "sides",
    "fusion"
]

print("Building Gradio interface...")
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        """
        # GPT-2 Recipe Generator
        Select a genre and enter a title to generate a recipe.
        (Note: The model's quality depends on its training. This version was trained on a small dataset.)
        """
    )
    
    with gr.Row():
        genre_input = gr.Dropdown(
            label="Recipe Genre", 
            choices=GENRE_CHOICES,
            value="bakery" 
        )
        title_input = gr.Textbox(label="Recipe Title", placeholder="e.g., Spicy Chicken Wings")
    
    generate_btn = gr.Button("Generate Recipe", variant="primary")
    
    gr.Markdown("---")
    
    output_text = gr.Textbox(label="Generated Recipe", lines=15, interactive=False)
    
    generate_btn.click(
        fn=create_recipe, 
        inputs=[genre_input, title_input], 
        outputs=output_text
    )



print("Launching Gradio app... Open the local URL in your browser.")
demo.launch(share=True)