import pandas as pd
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# Load NLP Model & Sentence Embedding Model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Training Data with Stance Labels
train_data = [
    {"text": "Electric cars have the potential to significantly reduce pollution by cutting down emissions from traditional fuel-powered vehicles. Many experts believe this transition will improve air quality and contribute to a greener future.", 
     "target": "Electric cars", "stance": 1},  # Support
    
    {"text": "The rise of artificial intelligence is leading to widespread job automation, causing many workers to lose employment opportunities. Industries such as manufacturing and customer service are already heavily impacted, raising concerns about the future workforce.", 
     "target": "AI", "stance": -1},  # Oppose

    {"text": "Climate change is an undeniable reality, with rising global temperatures, melting ice caps, and increasing extreme weather events. Scientists warn that immediate action is required to mitigate the long-term damage caused by human activities.", 
     "target": "Climate change", "stance": 1},  # Support

    {"text": "Renewable energy sources like solar and wind power are gaining popularity as sustainable alternatives to fossil fuels. Governments and industries worldwide are investing heavily in green energy to combat climate change and reduce carbon footprints.", 
     "target": "Renewable energy", "stance": 1},  # Support

    {"text": "Despite technological advancements, self-driving cars remain a safety concern due to unpredictable failures and accidents. Many argue that autonomous vehicles lack the human instinct needed to make split-second decisions in life-threatening situations.", 
     "target": "Self-driving cars", "stance": -1},  # Oppose

    {"text": "Although nuclear energy provides a high power output, it is often criticized for its potential hazards, including radioactive waste and the risk of catastrophic accidents. Many environmentalists advocate for safer and more sustainable energy solutions.", 
     "target": "Nuclear energy", "stance": -1},  # Oppose

    {"text": "Electric vehicles are an environmentally friendly innovation, but their high initial cost remains a significant barrier to widespread adoption. Consumers often hesitate due to the expensive battery replacements and the lack of adequate charging infrastructure.", 
     "target": "Electric vehicles", "stance": 0},  # Neutral
]


# Extract Relations (Subject-Verb-Object)
def extract_relations(text):
    doc = nlp(text)
    relations = []
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):  # Find Subject
            subject = token.text
            verb = token.head.text
            for child in token.head.children:
                if child.dep_ in ("dobj", "attr", "acomp"):  # Find Object
                    object_ = child.text
                    relations.append((subject, verb, object_))
    return relations

# Construct Directed Acyclic Graph (DAG)
DAG = nx.DiGraph()
stance_labels = {}  # Store stance labels
relation_vectors = {}  # Store sentence embeddings

for data in train_data:
    text, target, stance = data["text"], data["target"], data["stance"]
    relations = extract_relations(text)
    for rel in relations:
        relation_text = " ".join(rel)
        relation_vectors[relation_text] = model.encode(relation_text)  # Encode relation
        DAG.add_edge(rel[0], rel[2], label=rel[1])
        stance_labels[(rel[0], rel[2])] = stance  # Store stance

# Visualize DAG
import matplotlib.pyplot as plt
import networkx as nx

def visualize_dag():
    plt.figure(figsize=(12, 8))  # Larger figure for better clarity

    # Define colors based on stance labels
    node_colors = []
    for node in DAG.nodes():
        stance_value = None
        for u, v in DAG.edges():
            if u == node or v == node:
                stance_value = stance_labels.get((u, v), -1)
                break  # Get stance for any one relation
        if stance_value == 1:
            node_colors.append('lightgreen')  # Support
        elif stance_value == -1:
            node_colors.append('lightcoral')  # Oppose
        else:
            node_colors.append('lightgray')  # Neutral

    # Use Kamada-Kawai layout for better spacing and readability
    pos = nx.kamada_kawai_layout(DAG)  

    # Draw nodes with labels
    nx.draw(DAG, pos, with_labels=True, node_size=3500, 
            node_color=node_colors, edge_color='black', 
            font_size=12, font_weight="bold", arrows=True)

    # Draw curved edges to avoid overlaps
    edge_labels = {(u, v): f"{d['label']} ({stance_labels.get((u, v), -1)})" for u, v, d in DAG.edges(data=True)}
    nx.draw_networkx_edge_labels(DAG, pos, edge_labels=edge_labels, font_size=10, font_color="blue", rotate=False, label_pos=0.4)
    nx.draw_networkx_edges(DAG, pos, edge_color='black', width=2, connectionstyle="arc3,rad=0.2")  # Curved edges

    plt.title("Directed Acyclic Graph (DAG) with Stance Labels", fontsize=14)
    plt.show()

# Call the visualization function
visualize_dag()



# Testing Data (Text + Target)
test_data = [
    {"text": "Electric vehicles help the environment.", "target": "Electric vehicles"},
    {"text": "AI is creating new opportunities.", "target": "AI"},
    {"text": "Climate change is a myth.", "target": "Climate change"},
    {"text": "Nuclear energy can provide clean power.", "target": "Nuclear energy"},
    {"text": "Self-driving cars improve road safety.", "target": "Self-driving cars"},
]

# Predict Stance Using DAG & Semantic Similarity
def predict_stance(test_text, test_target):
    test_relations = extract_relations(test_text)
    if not test_relations:
        return -1  # Unknown stance if no relation found

    test_vector = model.encode(" ".join(test_relations[0]))
    best_match, best_score, predicted_stance = None, 0, -1

    print(f"\nTest Text: {test_text} | Target: {test_target}")  # Debugging
    for stored_rel, stored_vector in relation_vectors.items():
        score = util.pytorch_cos_sim(test_vector, stored_vector).item()
        print(f"Comparing with: {stored_rel} | Similarity: {score:.3f}")  # Debugging
        if score > best_score:
            best_match, best_score = stored_rel, score
            predicted_stance = stance_labels.get((best_match.split()[0], best_match.split()[-1]), -1)

    print(f"Best Match: {best_match} | Score: {best_score:.3f} | Predicted Stance: {predicted_stance}\n")
    
    if best_score >= 0.55:  # Threshold for Support
        return 1
    elif best_score > 0.4:  # Threshold for Neutral
        return 0
    else:
        return -1  # Oppose / Unknown

# Run Predictions for Test Data
for test in test_data:
    text, target = test["text"], test["target"]
    stance = predict_stance(text, target)
    print(f"Text: {text} | Target: {target} | Predicted Stance: {stance}")

# Visualize DAG
visualize_dag()
