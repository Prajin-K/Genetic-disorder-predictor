import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Helper functions ----------------
def canonicalize_genotype(s):
    s = s.strip().replace(' ', '')
    if len(s) != 2:
        raise ValueError("Genotype must have exactly two letters (e.g., Cc, CC, cc)")
    a, b = s[0], s[1]
    if a.isupper() and b.isupper():
        return a.upper() + b.upper()
    elif a.islower() and b.islower():
        return a.lower() + b.lower()
    elif a.isupper() or b.isupper():
        return a.upper() + a.lower()
    else:
        return s

def get_gametes(genotype):
    return [genotype[0], genotype[1]]

def classify_phenotype(genotype):
    a, b = genotype[0], genotype[1]
    if a.isupper() and b.isupper():
        return 'Normal'
    elif (a.isupper() and b.islower()) or (a.islower() and b.isupper()):
        return 'Carrier'
    else:
        return 'Affected'

# ---------------- Core Predictor ----------------
def run_predictor(p1, p2):
    g1 = canonicalize_genotype(p1)
    g2 = canonicalize_genotype(p2)
    gam1, gam2 = get_gametes(g1), get_gametes(g2)

    punnett = []
    for i in range(2):
        row = []
        for j in range(2):
            child = canonicalize_genotype(gam1[i] + gam2[j])
            row.append(child)
        punnett.append(row)

    flat = [punnett[i][j] for i in range(2) for j in range(2)]
    data = pd.Series(flat).value_counts().reset_index()
    data.columns = ['Genotype', 'Count']
    data['Probability (%)'] = data['Count'] / 4 * 100
    data['Phenotype'] = data['Genotype'].apply(classify_phenotype)
    return data, punnett

# ---------------- GRADIO APP ----------------
def predict_single(p1, p2):
    try:
        df, punnett = run_predictor(p1, p2)

        punnett_df = pd.DataFrame(punnett, columns=["Gamete 1", "Gamete 2"])

        fig, ax = plt.subplots()
        ax.bar(df['Genotype'], df['Probability (%)'])
        ax.set_title('Genotype Distribution (%)')
        ax.set_ylim(0, 100)
        plt.tight_layout()

        fig2, ax2 = plt.subplots()
        phenotype_summary = df.groupby('Phenotype')['Probability (%)'].sum()
        ax2.pie(phenotype_summary, labels=phenotype_summary.index, autopct='%1.1f%%')
        ax2.set_title('Phenotype Distribution')
        plt.tight_layout()

        return punnett_df, df, fig, fig2
    except Exception as e:
        return f"Error: {e}", None, None, None


def batch_predict(csv_file):
    try:
        df_csv = pd.read_csv(csv_file.name)
        results = []
        for _, row in df_csv.iterrows():
            df_res, _ = run_predictor(row['Parent1'], row['Parent2'])
            df_res['Input'] = f"{row['Parent1']} x {row['Parent2']}"
            results.append(df_res)
        final = pd.concat(results)
        return final
    except Exception as e:
        return f"CSV Error: {e}"


with gr.Blocks(title="Genetic Disorder Predictor") as app:
    gr.Markdown("# 🧬 Genetic Disorder Predictor – Punnett Square Tool")

    with gr.Tab("Single Prediction"):
        p1 = gr.Textbox(label="Parent 1 Genotype (e.g., Cc, CC, cc)")
        p2 = gr.Textbox(label="Parent 2 Genotype (e.g., Cc, CC, cc)")
        btn = gr.Button("Predict")

        punnett_out = gr.Dataframe(label="Punnett Square")
        table_out = gr.Dataframe(label="Genotype Probability Table")
        plot1 = gr.Plot(label="Genotype Distribution (%)")
        plot2 = gr.Plot(label="Phenotype Distribution")

        btn.click(predict_single, inputs=[p1, p2], outputs=[punnett_out, table_out, plot1, plot2])

    with gr.Tab("Batch CSV Survey"):
        csv_input = gr.File(label="Upload CSV (Parent1, Parent2)")
        csv_btn = gr.Button("Run Batch Analysis")
        csv_output = gr.Dataframe(label="Survey Results")
        csv_btn.click(batch_predict, inputs=[csv_input], outputs=[csv_output])

app.launch()