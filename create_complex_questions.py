# create_complex_questions.py
import json
import argparse
import os
from tqdm import tqdm
import time
from openai import OpenAI

def load_jsonl(filepath):
    """Load JSONL dataset"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, filepath):
    """Save data as JSONL"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def create_complex_version(client, question_data, temperature=0.7):
    """
    Use Claude Sonnet 4.5 to create a superficially complex version of the question.
    
    The goal: Same answer difficulty, but more complex wording/structure.
    """
    
    original_question = question_data['question']
    correct_answer = question_data['correct_answer']
    distractors = question_data['distractors']
    
    prompt = f"""You are rewriting multiple choice questions to be SUPERFICIALLY more complex while keeping the ACTUAL difficulty the same.

Original question: {original_question}
Correct answer: {correct_answer}
Wrong answers: {', '.join(distractors)}

Rewrite this question to be superficially complex by:
1. Using more formal/academic language
2. Adding unnecessary complexity to sentence structure
3. Using longer words where possible
4. Making it sound technical/sophisticated
5. BUT: Keep the SAME factual question (don't make it actually harder to answer)

The answer should still be the same! Just make the question LOOK harder without BEING harder.

Examples of superficial complexity:
- "What color is the sky?" → "What chromatic attribute characterizes the atmospheric expanse during standard daylight conditions?"
- "Who wrote Hamlet?" → "Which literary figure authored the theatrical composition commonly referenced as 'Hamlet'?"

Return ONLY the rewritten question text, nothing else."""

    try:
        response = client.chat.completions.create(
            model="anthropic/claude-sonnet-4.5:beta",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=200
        )
        
        complex_question = response.choices[0].message.content.strip()
        
        # Remove quotes if model added them
        if complex_question.startswith('"') and complex_question.endswith('"'):
            complex_question = complex_question[1:-1]
        if complex_question.startswith("'") and complex_question.endswith("'"):
            complex_question = complex_question[1:-1]
        
        return complex_question
    
    except Exception as e:
        print(f"Error processing question: {e}")
        return None

def create_simple_version(client, question_data, temperature=0.7):
    """
    Use Claude Sonnet 4.5 to create a superficially simple version of the question.
    
    The goal: Same answer difficulty, but simpler wording/structure.
    """
    
    original_question = question_data['question']
    correct_answer = question_data['correct_answer']
    distractors = question_data['distractors']
    
    prompt = f"""You are rewriting multiple choice questions to be SUPERFICIALLY simpler while keeping the ACTUAL difficulty the same.

Original question: {original_question}
Correct answer: {correct_answer}
Wrong answers: {', '.join(distractors)}

Rewrite this question to be superficially simple by:
1. Using casual/conversational language
2. Using short, simple sentences
3. Using common everyday words
4. Making it sound informal/friendly
5. BUT: Keep the SAME factual question (don't make it actually easier to answer)

The answer should still be the same! Just make the question LOOK easier without BEING easier.

Examples of superficial simplicity:
- "What chromatic attribute characterizes the atmospheric expanse?" → "What color is the sky?"
- "Which literary figure authored Hamlet?" → "Who wrote Hamlet?"

Return ONLY the rewritten question text, nothing else."""

    try:
        response = client.chat.completions.create(
            model="anthropic/claude-sonnet-4.5:beta",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=200
        )
        
        simple_question = response.choices[0].message.content.strip()
        
        # Remove quotes if model added them
        if simple_question.startswith('"') and simple_question.endswith('"'):
            simple_question = simple_question[1:-1]
        if simple_question.startswith("'") and simple_question.endswith("'"):
            simple_question = simple_question[1:-1]
        
        return simple_question
    
    except Exception as e:
        print(f"Error processing question: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Create superficially complex/simple versions of questions"
    )
    parser.add_argument("--input_file", type=str, required=True,
                       help="Input JSONL file with questions")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output JSONL file")
    parser.add_argument("--mode", type=str, choices=["complex", "simple", "both"],
                       default="complex",
                       help="Create complex, simple, or both versions")
    parser.add_argument("--num_questions", type=int, default=None,
                       help="Number of questions to process (default: all)")
    parser.add_argument("--openrouter_api_key", type=str, default=None,
                       help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation (default: 0.7)")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between API calls in seconds (default: 0.5)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key required. Either pass --openrouter_api_key "
            "or set OPENROUTER_API_KEY environment variable"
        )
    
    # Initialize OpenRouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    print(f"Loading questions from {args.input_file}...")
    data = load_jsonl(args.input_file)
    
    # Limit number of questions if specified
    if args.num_questions:
        data = data[:args.num_questions]
        print(f"Processing {args.num_questions} questions")
    else:
        print(f"Processing all {len(data)} questions")
    
    results = []
    
    for item in tqdm(data, desc="Processing questions"):
        result = {
            "qid": item["qid"],
            "original_question": item["question"],
            "correct_answer": item["correct_answer"],
            "distractors": item["distractors"],
            "prop": item.get("prop"),
            "s_pop": item.get("s_pop"),
            "o_pop": item.get("o_pop")
        }
        
        if args.mode in ["complex", "both"]:
            complex_q = create_complex_version(client, item, args.temperature)
            if complex_q:
                result["complex_question"] = complex_q
            time.sleep(args.delay)
        
        if args.mode in ["simple", "both"]:
            simple_q = create_simple_version(client, item, args.temperature)
            if simple_q:
                result["simple_question"] = simple_q
            time.sleep(args.delay)
        
        results.append(result)
    
    # Save results
    save_jsonl(results, args.output_file)
    print(f"\n✓ Saved {len(results)} questions to {args.output_file}")
    
    # Print some examples
    print("\n" + "="*80)
    print("EXAMPLES:")
    print("="*80)
    for i, item in enumerate(results[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Original:  {item['original_question']}")
        if 'complex_question' in item:
            print(f"Complex:   {item['complex_question']}")
        if 'simple_question' in item:
            print(f"Simple:    {item['simple_question']}")
        print(f"Answer:    {item['correct_answer']}")

if __name__ == "__main__":
    main()