import pandas as pd
from modules.build_graph import build_graph, visualize_graph_pyvis

# Load dataset
df = pd.read_csv("data/dataset1.csv")

relations = []

for _, row in df.iterrows():
    symptom = str(row.get('Symptom', '')).strip()

    # Possible Diseases
    if pd.notna(row.get('Possible Diseases')):
        diseases = str(row['Possible Diseases']).split(',')
        for disease in diseases:
            relations.append((symptom, 'may_indicate', disease.strip()))

    # Severity
    if pd.notna(row.get('Severity')):
        severity = str(row['Severity']).strip()
        relations.append((symptom, 'has_severity', severity))

    # Average Duration
    if pd.notna(row.get('Average Duration (days)')):
        duration = str(row['Average Duration (days)']).strip()
        relations.append((symptom, 'lasts_for', f"{duration} days"))

    # Common in Region
    if pd.notna(row.get('Common in Region')):
        regions = str(row['Common in Region']).split(',')
        for region in regions:
            relations.append((symptom, 'common_in', region.strip()))

    # Language Availability
    if pd.notna(row.get('Language Availability')):
        langs = str(row['Language Availability']).split(',')
        for lang in langs:
            relations.append((symptom, 'available_in', lang.strip()))

# Build and visualize
if relations:
    G = build_graph(relations)
    visualize_graph_pyvis(G)
    print("✅ Knowledge graph generated successfully! Check 'ui/graph_viz.html'")
else:
    print("⚠️ No relations found. Please check your column names.")
