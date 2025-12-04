# test_baseline_accuracy.py
from finetune_load_finetuned_model import load_base_model, load_finetuned_model
from finetune_data_handling import load_jsonl_dataset
import torch
from tqdm import tqdm
import argparse
from collections import Counter

def get_letter_token_ids(tokenizer, letter):
    token_ids = []
    ids = tokenizer.encode(" " + letter, add_special_tokens=False)
    if len(ids) == 1:
        token_ids.append(ids[0])
    ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) == 1 and ids[0] not in token_ids:
        token_ids.append(ids[0])
    return token_ids

def test_baseline(model, tokenizer, dataset, device="cuda"):
    correct = 0
    predictions = []
    correct_answers = []
    
    for row in tqdm(dataset):
        question = row["question"]
        options = row["options"]
        
        prompt = (
            f"{question}\n"
            f"A: {options['A']}\n"
            f"B: {options['B']}\n"
            f"C: {options['C']}\n"
            f"D: {options['D']}\n"
            "Answer:"
        )
        
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        enc = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        
        logits = out.logits[0, -1, :]
        
        # Get A, B, C, D token logits
        letter_logits = []
        for letter in "ABCD":
            ids = get_letter_token_ids(tokenizer, letter)
            if ids:
                letter_logits.append(max(logits[i].item() for i in ids))
            else:
                letter_logits.append(float('-inf'))
        
        pred = "ABCD"[letter_logits.index(max(letter_logits))]
        predictions.append(pred)
        correct_answers.append(row['correct_letter'])
        
        if pred == row['correct_letter']:
            correct += 1
    
    accuracy = correct / len(dataset)
    
    # Analyze distributions
    print("\n" + "="*60)
    print("BASELINE ACCURACY TEST RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {correct}/{len(dataset)} ({accuracy:.2%})")
    
    # Answer distribution
    pred_dist = Counter(predictions)
    correct_dist = Counter(correct_answers)
    
    print("\n--- ANSWER DISTRIBUTION ---")
    print("\nModel's predictions:")
    for letter in ['A', 'B', 'C', 'D']:
        count = pred_dist.get(letter, 0)
        pct = count/len(predictions)*100
        print(f"  {letter}: {count:4d} ({pct:5.1f}%)")
    
    print("\nCorrect answers distribution:")
    for letter in ['A', 'B', 'C', 'D']:
        count = correct_dist.get(letter, 0)
        pct = count/len(correct_answers)*100
        print(f"  {letter}: {count:4d} ({pct:5.1f}%)")
    
    # Chi-square test
    expected_per_option = len(predictions) / 4
    chi2_stat = sum((pred_dist.get(l, 0) - expected_per_option)**2 / expected_per_option 
                   for l in ['A', 'B', 'C', 'D'])
    print(f"\nChi-square test for uniformity: {chi2_stat:.2f}")
    print(f"  (Critical value at p=0.05: 7.81)")
    if chi2_stat < 7.81:
        print(f"  ✓ Answers look uniformly distributed")
    else:
        print(f"  ⚠️  Model shows significant answer bias")
    
    # Per-letter accuracy
    print("\n--- ACCURACY BY ANSWER CHOICE ---")
    for letter in ['A', 'B', 'C', 'D']:
        letter_total = correct_dist.get(letter, 0)
        if letter_total > 0:
            letter_correct = sum(1 for i, ca in enumerate(correct_answers) 
                               if ca == letter and predictions[i] == letter)
            letter_acc = letter_correct / letter_total
            print(f"  Questions with correct answer {letter}: {letter_correct}/{letter_total} ({letter_acc:.2%})")
    
    print("="*60)
    
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--lora_repo", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model...")
    if args.lora_repo:
        model, tokenizer = load_finetuned_model(args.base_model, args.lora_repo)
        print(f"Testing: {args.lora_repo}")
    else:
        model, tokenizer = load_base_model(args.base_model)
        print(f"Testing: {args.base_model}")
    
    model.to(device)
    
    print("Loading dataset...")
    dataset = load_jsonl_dataset(args.dataset_path)
    
    print(f"Testing on {len(dataset)} questions...")
    accuracy = test_baseline(model, tokenizer, dataset, device)

if __name__ == "__main__":
    main()