import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Dict, List


def load_log_files(log_dir: str) -> Dict[str, Dict]:
    experiments = {}
    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            with open(os.path.join(log_dir, filename), 'r') as f:
                content = f.read()
                config_start = content.find('{')
                config_end = content.find('}', config_start) + 1
                config = json.loads(content[config_start:config_end])
                
                results = {
                    'en': {},
                    'de': {},
                    'cross_lingual': {}
                }
                for lang in results.keys():
                    start = content.find(f"Evaluation Results for {lang.upper()} pairs:")
                    if start != -1:
                        end = content.find("\n\n", start)
                        section = content[start:end]
                        for line in section.split('\n')[1:]:
                            key, value = line.split(': ')
                            results[lang][key] = float(value)
                
                experiments[filename] = {
                    'config': config,
                    'results': results
                }
    return experiments


def load_csv_files(csv_dir: str, experiment: str) -> Dict[str, pd.DataFrame]:
    csv_data = {}
    experiment_dir = os.path.join(csv_dir, experiment)
    if os.path.exists(experiment_dir):
        for filename in os.listdir(experiment_dir):
            if filename.endswith(".csv"):
                csv_data[filename] = pd.read_csv(os.path.join(experiment_dir, filename), sep=';')
    return csv_data


def main():
    st.title("Sentence Pair Finder Experiment Dashboard")

    
    experiments = load_log_files("../../other_pairs/logs")

    
    selected_experiment = st.sidebar.selectbox(
        "Select Experiment",
        list(experiments.keys())
    )


    csv_data = load_csv_files("../../other_pairs/output", selected_experiment[:34])


    st.header("Experiment Configuration")
    st.json(experiments[selected_experiment]['config'])

    st.header("Evaluation Results")
    results = experiments[selected_experiment]['results']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['avg_bleu', 'avg_bert', 'avg_embedding_sim']
    x = list(range(len(metrics)))
    width = 0.25

    for i, lang in enumerate(['en', 'de', 'cross_lingual']):
        values = [results[lang].get(metric, 0) for metric in metrics]
        ax.bar([xi + i*width for xi in x], values, width, label=lang)

    ax.set_ylabel('Score')
    ax.set_title('Evaluation Metrics by Language')
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(metrics)
    ax.legend()

    st.pyplot(fig)


    if 'avg_lexical_div' in results['en'] and 'avg_lexical_div' in results['de']:
        st.subheader("Average Lexical Diversity")
        lex_div_data = {
            'Language': ['English', 'German'],
            'Lexical Diversity': [results['en']['avg_lexical_div'], results['de']['avg_lexical_div']]
        }
        lex_div_df = pd.DataFrame(lex_div_data)
        st.bar_chart(lex_div_df.set_index('Language'))

    st.header("Example Sentence Pairs")
    lang_option = st.selectbox("Select Language", ['en', 'de', 'cross_lingual'])
    csv_file = f"{lang_option}_diverse_pairs.csv"
    
    if csv_file in csv_data:
        df = csv_data[csv_file]
        st.dataframe(df)
        
        if 'Lexical Diversity' in df.columns and 'Similarity Score' in df.columns:
            if df['Lexical Diversity'].isnull().all():
                st.write("Lexical Diversity is empty. Cannot show scatter plot.")
            else:

                st.subheader("Similarity Score vs Lexical Diversity")
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x='Similarity Score', y='Lexical Diversity', ax=ax)
                st.pyplot(fig)
    else:
        st.write(f"No CSV file found for {lang_option}")

if __name__ == "__main__":
    main()