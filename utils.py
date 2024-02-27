import json, os, re, pycm, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
from IPython.display import display, Markdown
from classifier import Classifier

def clean_output(llm_output):
    normalized_output = llm_output.lower()
    match = re.search(r'\b(positive|negative)\b', normalized_output)
    if match:
        return match.group(1)
    else:
        return "Error: Expected token not found"

def evaluate_model(model, datasets, filename):
    if os.path.exists(filename):
        results = json.load(open(filename, "r"))
    else:
        results = []
    for i, dataset in enumerate(datasets):
        if i < len(results) and 0 < len(results[i]):
            pass
        else:
            classifier = Classifier(
                model_name=model, 
                id=dataset["concept"]["id"], 
                term=dataset["concept"]["label"],
                definition=dataset["concept"]["definition"]
            )
            classifications = []
            for j, entity in tqdm(enumerate(dataset["data"]), desc=dataset["concept"]["label"], total=len(dataset["data"])):
                classification = classifier.classify(name=entity["label"], description=entity["description"])
                classification["predicted"] = clean_output(classification["answer"])
                classification["actual"] = entity["actual"]
                classifications.append(classification)
            results.append(classifications)
        json.dump(results, open(filename, "w+"))
    return results

def confusion_matrix(classifications):
    df = pd.DataFrame.from_records(classifications)
    return pycm.ConfusionMatrix(df["actual"].tolist(), df["predicted"].tolist(), digit=2, classes=[ 'positive', 'negative' ])

def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]

def performance_statistics(experiments):
    comparison_stats = []
    for experiment in experiments:
        cms = [ (classifications[0]["concept"], confusion_matrix(classifications)) for classifications in experiment[1] ]
        stats = [ 
            { 
                "concept": cm[0], 
                "ACC": cm[1].ACC['positive'], 
                "AUC": cm[1].AUC['positive'], 
                "F1 Macro": cm[1].F1_Macro,
                "Cohen's kappa": cm[1].Kappa
            } for cm in cms 
        ]
        stats_df = pd.DataFrame.from_records(stats)
        mean_stats_df = stats_df[["ACC", "AUC", "F1 Macro", "Cohen's kappa"]].agg(['mean'], axis=0)
        mean_stats = mean_stats_df.to_dict('records')[0]
        mean_stats["Model"] = experiment[0]
        comparison_stats.append(mean_stats)
    return comparison_stats

def display_summary_stats(experiments):
    comparison_stats = performance_statistics(experiments)
    comparison_results_df = pd.DataFrame.from_records(comparison_stats)
    df_for_table = comparison_results_df[["Model", "ACC", "AUC", "F1 Macro", "Cohen's kappa"]].sort_values(by="AUC", ascending=False)
    styled_df = df_for_table.style.apply(highlight_max, subset=['ACC', 'AUC', 'F1 Macro', "Cohen's kappa"])
    display(styled_df)

def display_detailed_stats(experiment):
    cms = [ (classifications[0]["concept"], confusion_matrix(classifications)) for classifications in experiment[1] ]
    stats = [ 
        { 
            "concept": cm[0], 
            "ACC": cm[1].ACC['positive'], 
            "AUC": cm[1].AUC['positive'], 
            "F1 Macro": cm[1].F1_Macro,
            "Cohen's kappa": cm[1].Kappa
        } for cm in cms 
    ]
    stats_df = pd.DataFrame.from_records(stats)
    aggregate_stats_df = stats_df[["ACC", "AUC", "F1 Macro", "Cohen's kappa"]].agg(['mean', 'max', 'min'], axis=0)
    mean_stats_df = stats_df[["ACC", "AUC", "F1 Macro", "Cohen's kappa"]].agg(['mean'], axis=0)
    mean_stats = mean_stats_df.to_dict('records')[0]
    mean_stats["Model"] = experiment[0]
    display(Markdown(f'## Model: {experiment[0]}'))
    display(Markdown('### Performance metrics'))
    display(Markdown('#### By concept'))
    display(stats_df)
    display(Markdown('#### Aggregate'))
    display(aggregate_stats_df)
    display(Markdown('### Confusion matrices'))
    display_cms(cms)

def display_cms(cms):
    fig = plt.figure(figsize=(20,14))
    gs = fig.add_gridspec(4, 5, hspace=0.5)
    axes = gs.subplots()
    for ax, (name, cm) in zip(axes.flat, cms):
        df = pd.DataFrame(cm.matrix).T.fillna(0)
        sns.heatmap(df, annot=True, fmt='d', cmap="YlGnBu", ax=ax)
        ax.set_title(name, wrap=True, fontsize=9)
        ax.set(xlabel='LLM', ylabel='KG')
    for ax in axes.flat[len(cms):]:
        ax.set_visible(False)
    plt.show()

def display_errors(results, text=False):
    for result in results:
        df = pd.DataFrame.from_records(result)
        df_short = df[["entity", "description", "actual", "predicted", "rationale"]]
        df_fp = df_short[((df["actual"] == "negative") & (df["predicted"] == "positive"))]
        n_fps = len(df_fp)
        df_fn = df_short[((df["actual"] == 'positive') & (df["predicted"] == 'negative'))]
        n_fns = len(df_fn)
        if len(df_fp) > 0 or len(df_fn) > 0:
            display(Markdown(f'### {result[0]["concept"]}: {n_fns} false negatives, {n_fps} false positives'))
            display(Markdown(f'#### Definition'))
            display(Markdown(f'{result[0]["definition"]}'))
        if len(df_fp) > 0:
            display(Markdown("#### False negatives (LLM positive, KG negative)"))
            if text:
                for e in df_fp.to_dict(orient='records'):
                    display(Markdown('---'))
                    display(Markdown(f'##### {e["entity"]}'))
                    display(Markdown(f'Description'))
                    display(Markdown(e["description"]))
                    display(Markdown(f'Rationale'))
                    display(Markdown(e["rationale"]))
                display(Markdown('---'))
            else:
                df_fp_styler = df_fp.style.set_properties(**{"text-align": "left", "vertical-align" : "top", "overflow-wrap": "break-word"})
                display(df_fp_styler)
        if len(df_fn) > 0:
            display(Markdown("#### False positives (LLM negative, KG positive)"))
            if text:
                for e in df_fn.to_dict(orient='records'):
                    display(Markdown('---'))
                    display(Markdown(f'##### {e["entity"]}'))
                    display(Markdown(f'Description'))
                    display(Markdown(e["description"]))
                    display(Markdown(f'Rationale'))
                    display(Markdown(e["rationale"]))
                display(Markdown('---'))
            else:
                df_fn_styler = df_fn.style.set_properties(**{"text-align": "left", "vertical-align" : "top", "overflow-wrap": "break-word"})
                display(df_fn_styler)