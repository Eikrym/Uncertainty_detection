# Reproducing the Experiments

This section describes how to reproduce the experiments and analyses contained in the `experiments` directory.

Most experiments can be executed through `Run_all_experiments.py`. PCA, accuracy evaluation, PCA statistics, and the subsequent headwise analysis are executed separately.

## Project Structure

The project structure looks as follows:


project-root/
├── requirements.txt
├── experiments/
│   ├── Run_all_experiments.py
│   ├── base.py
│   ├── accuracy.py
│   ├── changeByLayer.py
│   ├── entropyByLayer.py
│   ├── HeadwiseDivergence.py
│   ├── HeadwiseDivergenceAnalysis.py
│   ├── layerWiseDifference.py
│   ├── layerwiseResidualDifference.py
│   ├── LogitLensAnalysis.py
│   ├── principalComponentAnalysis.py
│   └── PCA_stat.py
└── results/



## Installation

Install all required Python packages from the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The experiments load models and the dataset from Hugging Face. For models that require authentication, provide a Hugging Face access token through the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN="your_huggingface_token"
```

The current implementation loads the models using CUDA and `float16`. Therefore, an NVIDIA GPU with sufficient VRAM and a working CUDA installation is required unless the model-loading code in `base.py` is adapted.

## Running the Experiments

The scripts use relative paths such as `../results`. They should therefore be executed from inside the `experiments` directory:

```bash
cd experiments
```

### Running the Main Experiments

Most experiments are controlled through:

```
Run_all_experiments.py
```

Before starting the run, configure the following three lists in that file:

```python
Models = [
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]
```

This list determines which models are evaluated.

The `experiments` list determines which analyses are executed:

```python
experiments = [
    ChangeByLayer,
    entropyByLayer,
    HeadwiseDivergence,
    LayerwiseDifference,
    LayerwiseResidualDifference,
    LogitLensAnalysis,
]
```

Experiments can be enabled or disabled by uncommenting or commenting out the corresponding entries.



The `datafeeds` list controls which dataset variants are evaluated:

```python
datafeeds = [
    "3",
    "5",
    "not_enough_info",
    "two_groups",
]
```

The available data feeds are:

* `"not_enough_info"` compares examples with a certain eligibility label against examples whose expert label is `not enough information`.
* `"3"` uses examples with exactly three expert-relevant sentences and creates variants in which one, two, or all three relevant sentences are removed.
* `"5"` creates variants in which one to five expert-relevant sentences are removed, where enough relevant sentences are available.
* `"two_groups"` compares the original examples with partially manipulated examples and examples from which all expert-relevant sentences have been removed.

Once the models, experiments, and data feeds have been configured, run:

```
python Run_all_experiments.py
```

The script iterates over all selected models, experiments, and data feeds. Model resources are released between experiments to reduce GPU memory usage.

If an individual model–experiment combination fails, the error is printed and the script continues with the remaining configurations.

## Experiments Included in the Main Runner

The following experiments can be executed through `Run_all_experiments.py`.

### Change by Layer

Implemented in:

```text
changeByLayer.py
```

This experiment measures how strongly the representation changes between consecutive transformer layers.

Results are written to:

```text
results/change_by_layer/<model>/<datafeed>/
```

### Entropy by Layer

Implemented in:

```text
entropyByLayer.py
```

This experiment measures the entropy of the model output across transformer layers.

Results are written to:

```text
results/entropy/<model>/<datafeed>/
```

### Logit Lens Analysis

Implemented in:

```text
LogitLensAnalysis.py
```

This experiment measures the probability mass assigned to uncertainty-related tokens at each transformer layer.

Results are written to:

```text
results/logit_lens/<model>/<datafeed>/
```

### Layerwise Distance to the Final Representation

Implemented in:

```text
layerWiseDifference.py
```

This experiment compares intermediate-layer representations with the representation from the final transformer layer.

Results are written to:

```text
results/layer_Wise_divergence_last_layer/<model>/<datafeed>/
```

### Layerwise Residual Difference

Implemented in:

```text
layerwiseResidualDifference.py
```

This experiment measures differences between the residual-stream representations of the original and manipulated data groups.

For paired data feeds, the experiment compares examples using their `annotation_id`. For the `not_enough_info` data feed, the two groups are compared as unpaired groups and bootstrap confidence intervals are calculated.

Results are written to:

```text
results/Res_Diff/<model>/<datafeed>/
```

### Headwise Divergence

Implemented in:

```text
HeadwiseDivergence.py
```

This experiment calculates representation differences separately for each attention head.

Unlike most other experiments, the output contains an additional directory for each investigated condition:

```text
results/Head_Div/<model>/<datafeed>/<condition>/
```


Each condition directory contains the corresponding heatmap, CSV values, and run metadata.

## Accuracy Evaluation

The output-label accuracy is evaluated separately using:

```bash
python accuracy.py
```

The model to be evaluated is configured in the `if __name__ == "__main__"` block at the bottom of `accuracy.py`.

For example:

```python
exp = accuracy("Qwen/Qwen2.5-7B-Instruct")
exp.run_analysis()
```

To evaluate several models, execute the analysis once for each required model.

The accuracy analysis evaluates the following dataset variants:

* `certain`
* `not_enough_info`
* `manipulated_1`
* `manipulated_2`
* `manipulated_3`
* `fully_manipulated`
* `partially_manipulated`

Results are written to:

```text
results/accuracy/<model>/
```

The directory contains:

```text
accuracies.png
result.csv
run_info.json
```

### Selecting the Expected Label for Manipulated Examples

The expected labels for manipulated examples are controlled in `base.py`.

By default, the manipulated examples retain their original `expert_eligibility` label. 

For the three-step manipulation, the relevant mappings are located after the formatting:

They are provided in commented form:

```python
# manipulated_1 = manipulated_1.map(
#     lambda row: {"expert_eligibility": "not enough information"}
# )
```

Equivalent mappings are present for `manipulated_2` and `manipulated_3`.

For the `two_groups` data feed, corresponding mappings are provided for:

```python
remove_all_info
remove_every_possibility_except_complete
```

To test whether manipulated examples should be classified as `not enough information`, uncomment the corresponding mappings before executing `accuracy.py` and set the results folder to accuracy-remaining. 

To test against the original expert label, leave these mappings commented out.

In the uploaded experiment directory, the relevant evaluation script is named `accuracy.py`. 

After changing the mapping, rerun:

```bash
python accuracy.py
```

## PCA Experiments

PCA experiments are executed separately using:

```bash
python principalComponentAnalysis.py
```

The predefined PCA runs are configured in the `pca_runs` dictionary at the bottom of `principalComponentAnalysis.py`.

Each entry has the following structure:

```python
(datafeed, layer, head)
```

For example:

```python
("not_enough_info", 24, None)
```

runs PCA on the residual-stream activation from layer 24.

An entry with a head index:

```python
("not_enough_info", 25, 25)
```

runs PCA on the activation of attention head 25 in layer 25.

The predefined configurations contain selected layers and heads for the evaluated models. To reproduce only a subset of the PCA experiments, remove or comment out unnecessary entries in `pca_runs`.

PCA results are written to:

```text
results/PCA/<model>/<datafeed>/
```

For residual-stream PCA, the head value is `None`, for example:

```text
24_None_PCA.png
pca_layer_24_head_None.csv
```

### PCA Statistical Analysis

After generating the PCA CSV files, run:

```bash
python PCA_stat.py
```

This script recursively reads all PCA CSV files below:

```text
results/PCA/
```

It performs a permutation-based PERMANOVA-style group-separation test in the two-dimensional PCA space and calculates:

* the PERMANOVA F statistic,
* the permutation p-value,
* the explained group effect size `eta_squared`,
* a Bonferroni-corrected p-value,
* and whether the result remains significant after correction.

The combined output is saved as:

```text
results/pca_stat_results.csv
```

Existing PCA CSV files must therefore be generated before `PCA_stat.py` is executed.

## Headwise Divergence Analysis

The headwise ranking analysis is implemented in:

```text
HeadwiseDivergenceAnalysis.py
```

This is a post-processing step and must be run after `HeadwiseDivergence.py`, because it reads the previously generated `head_div.csv` files.

First generate the required headwise divergence results through `Run_all_experiments.py` using:

```python
HeadwiseDivergence
```

The subsequent ranking analysis currently supports the following data feeds:

```text
3
two_groups
```

The model is configured in the main block at the bottom of `HeadwiseDivergenceAnalysis.py`:

```python
exp = HeadwiseDivergenceAnalysis("Qwen/Qwen2-0.5B-Instruct")
exp.run_analysis("3")
exp.run_analysis("two_groups")
```

After selecting the required model and data feeds, run:

```bash
python HeadwiseDivergenceAnalysis.py
```

The script compares the headwise divergence values across the manipulation levels and ranks layer–head combinations by their increase in divergence.

Results are written to:

```text
results/Head_Div_Analysis/<model>/<datafeed>/
```

The ranking is saved as:

```text
head_increase_layer_head_ranking.csv
```

A separate execution is required for each model.

## Results Directory

For most experiments, the results follow this general structure:

```text
results/
└── <experiment>/
    └── <model>/
        └── <datafeed>/
            ├── <plot>.png
            ├── result.csv
            └── run_info.json
```

Model names are converted into directory-safe names by replacing `/` with `_`.

For example:

```text
Qwen/Qwen2.5-7B-Instruct
```

becomes:

```text
Qwen_Qwen2.5-7B-Instruct
```

The results directory may therefore look as follows:

```text
results/
├── logit_lens/
│   └── Qwen_Qwen2.5-7B-Instruct/
│       ├── 3/
│       │   ├── Logit_Lens_Analysis.png
│       │   ├── result.csv
│       │   └── run_info.json
│       ├── 5/
│       ├── not_enough_info/
│       └── two_groups/
├── entropy/
├── change_by_layer/
├── layer_Wise_divergence_last_layer/
├── Res_Diff/
├── Head_Div/
├── Head_Div_Analysis/
├── PCA/
└── accuracy-remaining/
```

The exact plot filename depends on the experiment.

### Result CSV Files

The CSV files store the numerical values used to create the plots.

For layerwise experiments, `result.csv` generally contains:

* the layer index,
* one column for each data group,
* delta columns comparing manipulated groups with the `certain` group,
* and a final summary row containing the maximum difference from the `certain` group.


### Run Metadata

Most experiment directories contain a:

```text
run_info.json
```

This file should be used to identify the exact configuration associated with a result.

Depending on the experiment, it contains information such as:

```json
{
  "experiment": "logit lens",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "manipulation_type": "3",
  "dataset": "TrialGPT-Criterion-Annotations",
  "git_commit": "abc1234",
  "counts_by_name": [
    ["certain", 100],
    ["manipulated_1", 300],
    ["manipulated_2", 300],
    ["manipulated_3", 100]
  ]
}
```

The exact values depend on the executed experiment.


## Recommended Reproduction Order

A complete reproduction can be performed in the following order:

```bash
cd experiments

# Main layerwise and headwise experiments
python Run_all_experiments.py

# Output-label accuracy
python accuracy.py

# PCA plots and coordinates
python principalComponentAnalysis.py

# Statistical analysis of all PCA outputs
python PCA_stat.py

# Ranking based on the generated headwise divergence CSV files
python HeadwiseDivergenceAnalysis.py
```

Before each command, verify that the required models, data feeds, experiment classes, layers, heads, and label mappings are configured correctly in the corresponding Python file.

Running the same experiment again with the same model and data-feed configuration may overwrite existing plots, CSV files, or `run_info.json` files. Existing results should therefore be copied or versioned before repeating a configuration with modified parameters.

