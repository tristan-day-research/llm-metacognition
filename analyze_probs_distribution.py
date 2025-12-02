import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter

# Load the JSON file
file_path = "explicit_confidence_task_logs/llama-3.1-8b-instruct_PopMC_0_difficulty_filtered_11412_2025-11-25-17-02-17_explicit_confidence_task_all_duplicate_questions_removed.json"

print("Loading JSON file...")
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract the highest probability answer for each question
answer_distribution = Counter()

print("Analyzing probs distribution...")
for question_id, result in data['results'].items():
    probs = result.get('probs', {})
    if probs:
        # Find the letter (A, B, C, or D) with the highest probability
        max_prob_letter = max(probs.items(), key=lambda x: x[1])[0]
        answer_distribution[max_prob_letter] += 1

# Print the distribution
print("\nDistribution of highest probability answers:")
total = sum(answer_distribution.values())
for letter in sorted(answer_distribution.keys()):
    count = answer_distribution[letter]
    percentage = (count / total) * 100
    print(f"{letter}: {count} ({percentage:.2f}%)")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
letters = sorted(answer_distribution.keys())
counts = [answer_distribution[letter] for letter in letters]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = ax1.bar(letters, counts, color=colors[:len(letters)], alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Answer Choice', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Highest Probability Answers\n(from probs field)', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add count labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{count}\n({count/total*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Pie chart
percentages = [(count / total) * 100 for count in counts]
wedges, texts, autotexts = ax2.pie(counts, labels=letters, autopct='%1.1f%%', 
                                    colors=colors[:len(letters)], startangle=90,
                                    textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Distribution of Highest Probability Answers\n(from probs field)', 
              fontsize=14, fontweight='bold')

# Make percentage text bold and larger
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

plt.tight_layout()
plt.savefig('probs_distribution.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved as 'probs_distribution.png'")

