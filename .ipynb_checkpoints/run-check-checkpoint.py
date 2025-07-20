import joblib
import re
import glob2
import os
from sklearn.metrics import classification_report, accuracy_score, f1_score

def simple_tokenizer(text):
    tokens = re.findall(r'\b[a-z]{2,}\b', text.lower())  # Words with ≥2 letters
    return tokens

# Sample labeled sentences
labeled_data = [
    # Negative (label: 0)
    ("While his innovative approach to business has earned him accolades in the media, the toxic work environment he cultivates has driven countless talented employees to resign in frustration.", 0),
    ("Though she is undeniably articulate and commands attention in every room she enters, the condescending manner in which she dismisses opposing viewpoints has alienated even her closest allies.", 0),
    ("Despite his generous philanthropic contributions, which have undeniably improved lives, the source of his wealth—rooted in environmental exploitation—continues to draw justified criticism.", 0),
    ("Her remarkable academic achievements are impressive, yet they often serve as a thin veil for the ruthless ambition that has left a trail of broken collaborations and betrayed mentors.", 0),
    ("Although he projects the image of a charismatic leader, those who’ve worked closely with him speak of manipulation, fear-based motivation, and a complete disregard for ethical standards.", 0),

    # Positive (label: 1)
    ("Even though her initial proposals were met with skepticism and her unorthodox methods raised eyebrows, her unwavering commitment and eventual success redefined the industry’s standard of innovation.", 1),
    ("While he occasionally comes across as blunt and unfiltered, his sincerity, refusal to sugarcoat the truth, and genuine concern for the people around him have earned him deep respect over time.", 1),
    ("Though burdened by personal trauma and often misunderstood for his withdrawn demeanor, his resilience and quiet acts of kindness speak volumes about the depth of his character.", 1),
    ("Despite being frequently underestimated due to her soft-spoken nature, she consistently outperforms expectations, solving complex problems with a blend of empathy and analytical precision.", 1),
    ("Even if his jokes sometimes border on inappropriate and his timing is far from perfect, his ability to bring people together and uplift morale when it’s most needed makes him truly indispensable.", 1),
]

# Added samples
labeled_data += [
    # More Negative (0)
    ("He speaks eloquently about justice and reform, yet his policies disproportionately harm the very communities he claims to defend.", 0),
    ("Although her speeches are full of hope and vision, her administration has consistently failed to deliver on even the most basic promises.", 0),
    ("While his calm demeanor gives the illusion of control, it often masks an alarming lack of urgency in moments of crisis.", 0),
    ("Despite earning praise for his diplomatic tone, his actions have escalated tensions rather than easing them.", 0),
    ("Though she carries herself with professionalism, her decisions often reflect favoritism and disregard for fairness.", 0),

    # More Positive (1)
    ("Even when faced with relentless opposition and misrepresentation, he maintained composure and responded with integrity that inspired many.", 1),
    ("Though her leadership style is unconventional and sometimes chaotic, it has produced undeniable results that others failed to achieve.", 1),
    ("Despite his lack of formal education, his innovative mind and tireless drive have propelled him into a position of real influence.", 1),
    ("Although her tone can come off as abrasive, her brutal honesty and unwavering dedication have earned lasting trust from her team.", 1),
    ("While he rarely takes credit and often stays in the background, his quiet efficiency has become the backbone of the entire operation.", 1),
]

# Detect models and vectorizers
model_files = sorted(glob2.glob("Benchmark-models/logistic_regression_model*.joblib"))
vectorizer_files = sorted(glob2.glob("Benchmark-models/tfidf_vectorizer*.joblib"))

# Pair them
paired = list(zip(model_files, vectorizer_files))

# Show detected pairs
print("[INFO] Detected Model-Vectorizer Pairs:")
for i, (model, vec) in enumerate(paired, 1):
    print(f"{i}. Model: {model}, Vectorizer: {vec}")

# User input to proceed
confirm = input("\nEnter '1' to continue: ")
if confirm.strip() != '1':
    print("[INFO] Operation cancelled by user.")
    exit()

# Optional: Competition between models
run_comp = input("Type 'yes' to run a competition between models on predefined labeled samples: ").strip().lower()

if run_comp == 'yes':
    print("\n[INFO] Running model competition on labeled samples...\n")
    for i, (model_path, vec_path) in enumerate(paired, 1):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)

        texts = [text for text, label in labeled_data]
        true_labels = [label for text, label in labeled_data]

        X_vec = vectorizer.transform(texts)
        pred_labels = model.predict(X_vec)

        print(f"\n[COMPETITION RESULT {i}] Model: {model_path}")
        print("Classification Report: \n", classification_report(true_labels, pred_labels, digits=3))
        print("F1 Score: ", f1_score(true_labels, pred_labels))
        print("Accuracy: ", accuracy_score(true_labels, pred_labels))
else:
    # Manual prediction on one input
    sample_text = input("Benchmark_talk#  ")
    for i, (model_path, vec_path) in enumerate(paired, 1):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)

        sample_vec = vectorizer.transform([sample_text])
        prediction = model.predict(sample_vec)

        print(f"\n[RESULT {i}] Model: {model_path}")
        print(f"[INFO] Prediction: {prediction[0]}")
