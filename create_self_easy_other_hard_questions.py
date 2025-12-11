import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPEN_API_KEY"))

# def generate_questions(num_questions=50):
#     """Generate questions that LLMs would know but humans typically wouldn't."""
    
#     prompt = f"""Generate {num_questions} multiple choice questions that a language model would likely answer correctly but an average college-educated person would not know.

# Focus on:
# - Specific technical documentation details (PyTorch, Transformers, TensorFlow parameters)
# - Obscure facts from technical papers or API documentation
# - Specific version numbers, default values, or parameter names from popular ML libraries
# - Details from programming language documentation

# For each question, provide:
# 1. A clear question
# 2. The correct answer
# 3. Three plausible but incorrect distractors

# Format as a JSON array with objects like:
# {{
#   "question": "What is the default value of the eps parameter in torch.nn.LayerNorm?",
#   "correct_answer": "1e-5",
#   "distractors": ["1e-6", "1e-4", "1e-8"]
# }}

# Return ONLY the JSON array, no other text."""

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are a technical documentation expert who creates precise multiple-choice questions."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.7
#     )
    
#     content = response.choices[0].message.content.strip()
    
#     # Remove markdown code blocks if present
#     if content.startswith("```"):
#         content = content.split("```")[1]
#         if content.startswith("json"):
#             content = content[4:]
#         content = content.strip()
    
#     questions = json.loads(content)
#     return questions
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPEN_API_KEY"))
def generate_questions_batch(batch_size=50):
    """Generate a batch of questions that LLMs would know but humans typically wouldn't."""
    
    prompt = f"""Generate exactly {batch_size} multiple choice questions about SPECIFIC code implementation details that require having actually seen or written ML code. Target difficulty: an LLM trained on code would get 50-60% correct, but someone who just understands ML concepts wouldn't know the answers.

REQUIRED: Questions must be about EXACT syntax, method names, return types, and API details - NOT concepts.

Good examples:
- "What method moves a PyTorch model to GPU: model.cuda(), model.to('cuda'), or model.gpu()?"
- "Does tokenizer.encode() return a list or tensor in Hugging Face Transformers?"
- "What attribute stores hidden size in BertConfig: hidden_size, d_model, or embedding_dim?"
- "To concatenate tensors along dim 0, use: torch.cat(), torch.stack(), or torch.concat()?"
- "What's the parameter name for vocabulary size in GPT2Config: vocab_size, n_vocab, or vocabulary_size?"
- "Does model.eval() disable dropout: yes, no, or only in training mode?"
- "To freeze parameters: param.requires_grad = False or param.freeze()?"

Bad examples (too conceptual):
- "What is dropout used for?" (concept, not code)
- "What does fit() do?" (too basic)
- "What optimizer is best?" (opinion, not code fact)

Focus on:
- Exact method/function names (cuda() vs to() vs gpu())
- Return types (list vs tensor vs dict)
- Exact attribute names in configs
- Parameter names and their exact spelling
- Method behavior details (does eval() affect dropout? does it affect batch norm?)
- Import paths (torch.nn vs torch.optim)
- Argument order or requirements
- Subtle API differences between similar methods

Make distractors plausible - use similar real method names or close variations.

Format as JSON array:
{{
  "question": "What method do you call to compute gradients in PyTorch: loss.backward(), loss.grad(), or loss.compute_grad()?",
  "correct_answer": "loss.backward()",
  "distractors": ["loss.grad()", "loss.compute_grad()", "loss.backprop()"]
}}

Return ONLY the JSON array, no other text. Generate exactly {batch_size} questions."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert programmer who creates questions about precise code syntax and API details, not concepts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    content = response.choices[0].message.content.strip()
    
    # Remove markdown code blocks if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    
    questions = json.loads(content)
    return questions

def generate_questions(num_questions, batch_size=50):
    """Generate questions in batches to ensure we get the full count."""
    all_questions = []
    remaining = num_questions
    
    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        print(f"Generating batch of {current_batch_size} questions ({len(all_questions)}/{num_questions} total so far)...")
        
        batch_questions = generate_questions_batch(current_batch_size)
        all_questions.extend(batch_questions)
        
        remaining -= len(batch_questions)
        
        if len(batch_questions) < current_batch_size:
            print(f"Warning: Only got {len(batch_questions)} questions instead of {current_batch_size}")
    
    return all_questions

def save_to_jsonl(questions, output_file="data/self_easy_other_hard_dataset.jsonl"):
    """Save questions to JSONL format."""
    with open(output_file, 'w') as f:
        for i, q in enumerate(questions, 1):
            entry = {
                "qid": f"self_easy_other_hard_{i}",
                "question": q["question"],
                "correct_answer": q["correct_answer"],
                "distractors": q["distractors"],
                "prop": "technical_knowledge"
            }
            f.write(json.dumps(entry) + '\n')
    
    print(f"Saved {len(questions)} questions to {output_file}")

if __name__ == "__main__":
    num_questions = 50
    print(f"Generating {num_questions} questions...")
    questions = generate_questions(num_questions)
    save_to_jsonl(questions)
    print("Done!")

def save_to_jsonl(questions, output_file="data/ml_code_questions.jsonl"):
    """Save questions to JSONL format."""
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for i, q in enumerate(questions, 1):
            entry = {
                "qid": f"self_easy_other_hard_{i}",
                "question": q["question"],
                "correct_answer": q["correct_answer"],
                "distractors": q["distractors"],
                "prop": "technical_knowledge"
            }
            f.write(json.dumps(entry) + '\n')
    
    print(f"Saved {len(questions)} questions to {output_file}")

if __name__ == "__main__":
    num_questions = 100  # Adjust as needed
    print(f"Generating {num_questions} questions...")
    questions = generate_questions(num_questions)
    save_to_jsonl(questions)
    print("Done!")