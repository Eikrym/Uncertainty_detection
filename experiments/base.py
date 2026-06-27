import csv
import json
import subprocess

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformer_lens import HookedTransformer
from huggingface_hub import login
from datasets import load_dataset
from datasets import Dataset
from itertools import combinations
from collections import defaultdict
import torch
import re
import ast
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


class experiments_base:

    DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    DEFAULT_DATASET = "ncbi/TrialGPT-Criterion-Annotations"

    def __init__(self, model_name=DEFAULT_MODEL, dataset_name=DEFAULT_DATASET):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_name)

        #setup model and device
        self.model = HookedTransformer.from_pretrained_no_processing(self.model_name,torch_dtype=torch.float16, device="cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


        #get uncertainity tokens
        self.uncertainty_tokens = self.getUncertaintyTokens()

        #dataset to text
        self.usable_columns = self.dataset['train'].column_names[:6]
        self.dataset_subset = self.dataset['train'].select_columns(self.usable_columns)
        self.dataset_subset = self.dataset_subset.map(lambda row: self.format_input(self.usable_columns, row)) 

        #initialize prompts with correct fromat

    def getUncertaintyTokens(self):
        """Get uncertainty tokens for the model's tokenizer."""
        uncertainty_words = [
            "unknown", "unsure", "don", "'t", "know", "can't", "unable", "impossible",
            "doesn't", "exist", "fictional", "imaginary", "no", "such", "thing",
            "hypothetical", "made", "up", "never", "happened"
        ]
            
        token_ids = []
        for w in uncertainty_words:
            try:
                toks = self.model.to_tokens(" " + w, prepend_bos=False)[0]
                token_ids.extend(toks.tolist())
            except Exception:
                pass
        return sorted(list(set(token_ids)))    
    def build_chat_prompt(self, prompt, add_generation_prompt=True):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful medical trial eligibility reviewer.\n"
                    "Use only the given input.\n"
                    "Do not assume facts that are not stated.\n"
                    "When information is missing or unclear, explicitly mention that."
                )
            },
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "Review the patient's eligibility for this trial in 2-4 sentences."
                )
            }
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )

    def prompt_to_tokens(self, prompt, add_generation_prompt=True):
        text = self.build_chat_prompt(prompt, add_generation_prompt=add_generation_prompt)
        tokens = self.model.to_tokens(text)
        return tokens.to(self.model.cfg.device)

    def get_model_response(self, prompt, max_new_tokens=50):


        tokens = self.model.to_tokens(prompt).to(self.model.cfg.device)

        with torch.no_grad():
            generated = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                verbose=False
            )

        return self.model.to_string(
            generated[0, tokens.shape[1]:]
        ).strip()
        

    #turn data into text, so model can read it
    def format_input(self, allowed_columns, row):

        def renumber_note(note):
            if not isinstance(note, str):
                return note

            # Ganze nummerierte Blöcke extrahieren
            blocks = re.findall(
                r'^\s*\d+\..*(?:\n(?!\s*\d+\.).*)*',
                note,
                flags=re.MULTILINE
            )

            # Falls keine nummerierten Blöcke gefunden werden, Note unverändert lassen
            if not blocks:
                return note

            renumbered_blocks = []

            for new_number, block in enumerate(blocks, start=1):
                # Nur die Zahl am Anfang ersetzen, Inhalt bleibt gleich
                new_block = re.sub(
                    r'^\s*\d+\.',
                    f'{new_number}.',
                    block,
                    count=1
                )
                renumbered_blocks.append(new_block)

            return "\n".join(renumbered_blocks)

        text = ""

        for col in allowed_columns:
            display_col = "note" if col == "_note" else col
            value = row[col]

            # Nummerierung nur bei patient note / manipuliertem _note korrigieren
            if col in {"note", "_note"}:
                value = renumber_note(value)

            text += f"{display_col}: {value}\n"

        row["input_text"] = text
        return row
    def _get_investigating_datasets(self, manipulation_type):
        manipulation_type = str(manipulation_type)

        if manipulation_type == "3":
            certain, manipulated_1, manipulated_2, manipulated_3 = self.get_three_sentence_removal_variants()
            return [
                ("certain", certain),
                ("manipulated_1", manipulated_1),
                ("manipulated_2", manipulated_2),
                ("manipulated_3", manipulated_3),
            ]

        elif manipulation_type == "5":
            certain, manipulated_1, manipulated_2, manipulated_3, manipulated_4, manipulated_5 = self.get_relevant_sentence_removal_variants()
            return [
                ("certain", certain),
                ("manipulated_1", manipulated_1),
                ("manipulated_2", manipulated_2),
                ("manipulated_3", manipulated_3),
                ("manipulated_4", manipulated_4),
                ("manipulated_5", manipulated_5),
            ]

        elif manipulation_type == "two_groups":
            certain, fully_manipulated, partially_manipulated = self.get_full_and_partial_removal_variants()
            return [
                ("certain", certain),
                ("fully_manipulated", fully_manipulated),
                ("partially_manipulated", partially_manipulated),
            ]

        elif manipulation_type == "not_enough_info":
            certain, not_enough_info = self.get_certain_and_uncertain_examples()
            return [
                ("certain", certain),
                ("not_enough_info", not_enough_info),
            ]

        else:
            raise ValueError(
                f"Unknown manipulation_type: {manipulation_type}. "
                "Use one of: '3', '5', 'two_groups', 'not_enough_info'."
            )
        
        
    def get_certain_and_uncertain_examples(self):

        certain = self.dataset['train'].filter(lambda x: x['expert_eligibility'] in {"included", "not included", "excluded"})
        not_enough_info = self.dataset['train'].filter(lambda x: x['expert_eligibility'] in {"not enough information"})

        certain = certain.select_columns(['criterion_type','note', 'criterion_text', 'annotation_id'])
        not_enough_info = not_enough_info.select_columns(['criterion_type','note', 'criterion_text', 'annotation_id'])

        certain = certain.map(lambda row: self.format_input({'criterion_type','note', 'criterion_text'}, row))
        not_enough_info = not_enough_info.map(lambda row: self.format_input({'criterion_type','note', 'criterion_text'}, row))
        return certain, not_enough_info  

    #remove first relevant sentence from note, to create manipulated version of dataset
    def get_three_sentence_removal_variants(self):

        # Keep only certain eligibility labels
        certain = self.dataset['train'].filter(
            lambda x: x['expert_eligibility'] in {
                "included",
                "excluded",
                "not included"
            }
        )

        # Keep only relevant columns
        certain = certain.select_columns([
            'note',
            'criterion_text',
            'criterion_type',
            'expert_sentences',
            'annotation_id'
        ])

        # Function applied to every row
        def remove_single_relevant_sentence(example, removable_sentence):

            note = example['note']


            # Split note into sentences
            # Ganze nummerierte Zeilen extrahieren
            blocks = re.findall(
                r'^\d+\..*(?:\n(?!\d+\.).*)*',
                note,
                flags=re.MULTILINE
            )

        

            # Safety check
            for sentence in sorted(removable_sentence, reverse=True):
                if 0 <= sentence < len(blocks):
                    del blocks[sentence]

            # Mit ursprünglichen Zeilenumbrüchen zusammensetzen
            example['_note'] = "\n".join(blocks)
            example['removed_sentences'] = removable_sentence

            return example
        
        def remove_each_relevant_sentence(example):
            important_sentences = example["expert_sentences"]
            results_1, results_2, results_3 = [], [], []
            if not important_sentences:
                    new_example = example.copy()
                    new_example["_note"] = example["note"]
                    new_example["sentence_removed"] = None
                    return [new_example]
            
            if isinstance(important_sentences, str):
                    try:
                        important_sentences = ast.literal_eval(important_sentences)
                    except Exception:
                        important_sentences = []
            
            if len(important_sentences) == 3:
                
                results_1 = []
                
                for sentence_idx in range(len(important_sentences)):
                    new_example = remove_single_relevant_sentence(
                        example.copy(),
                        [important_sentences[sentence_idx]]
                    )
                    results_1.append(new_example)

            if len(important_sentences) == 3:
                results_2 = []
                for sentence_pair in combinations(important_sentences, 2):
                    new_example = remove_single_relevant_sentence(
                        example.copy(),
                        list(sentence_pair)
                    )
                    results_2.append(new_example)
            
            if len(important_sentences) == 3:
                results_3 = []
                for sentence_triplet in combinations(important_sentences, 3):
                    new_example = remove_single_relevant_sentence(
                        example.copy(),
                        list(sentence_triplet)
                    )
                    results_3.append(new_example)
            

            
            return results_1, results_2, results_3

        # Apply manipulation
        new_rows_1 = []
        new_rows_2 = []
        new_rows_3 = []

        for example in certain:
            results_1, results_2, results_3= remove_each_relevant_sentence(example)

            new_rows_1.extend(results_1)
            new_rows_2.extend(results_2)
            new_rows_3.extend(results_3)

        manipulated_1 = Dataset.from_list(new_rows_1)
        manipulated_2 = Dataset.from_list(new_rows_2)
        manipulated_3 = Dataset.from_list(new_rows_3)
        certain = certain.map(lambda row: self.format_input(("criterion_type","criterion_text","note"), row))
        manipulated_1 = manipulated_1.map(lambda row: self.format_input(("criterion_type","criterion_text","_note"), row))
        manipulated_2 = manipulated_2.map(lambda row: self.format_input(("criterion_type","criterion_text","_note"), row))
        manipulated_3 = manipulated_3.map(lambda row: self.format_input(("criterion_type","criterion_text","_note"), row))


        #print('printing inseide manipulation')
        #print(certain[1])
        #print(manipulated[1])
        return certain, manipulated_1, manipulated_2, manipulated_3

    def get_relevant_sentence_removal_variants(self):

        # Keep only certain eligibility labels
        certain = self.dataset['train'].filter(
            lambda x: x['expert_eligibility'] in {
                "included",
                "excluded",
                "not included"
            }
        )

        # Keep only relevant columns
        certain = certain.select_columns([
            'note',
            'criterion_text',
            'criterion_type',
            'expert_sentences',
            'annotation_id'
        ])

        # Function applied to every row
        def remove_single_relevant_sentence(example, removable_sentence):

            note = example['note']


            # Split note into sentences
            # Ganze nummerierte Zeilen extrahieren
            blocks = re.findall(
                r'^\d+\..*(?:\n(?!\d+\.).*)*',
                note,
                flags=re.MULTILINE
            )

        

            # Safety check
            for sentence in sorted(removable_sentence, reverse=True):
                if 0 <= sentence < len(blocks):
                    del blocks[sentence]

            # Mit ursprünglichen Zeilenumbrüchen zusammensetzen
            example['_note'] = "\n".join(blocks)
            example['removed_sentences'] = removable_sentence

            return example
        
        def remove_each_relevant_sentence(example):
            important_sentences = example["expert_sentences"]
            results_1, results_2, results_3, results_4, results_5 = [], [], [], [], []
            if not important_sentences:
                    new_example = example.copy()
                    new_example["_note"] = example["note"]
                    new_example["sentence_removed"] = None
                    return [new_example]
            
            if isinstance(important_sentences, str):
                    try:
                        important_sentences = ast.literal_eval(important_sentences)
                    except Exception:
                        important_sentences = []
            
            if len(important_sentences) >= 1:
                
                results_1 = []
                
                for sentence_idx in range(len(important_sentences)):
                    new_example = remove_single_relevant_sentence(
                        example.copy(),
                        [important_sentences[sentence_idx]]
                    )
                    results_1.append(new_example)

            if len(important_sentences) >= 2:
                results_2 = []
                for sentence_pair in combinations(important_sentences, 2):
                    new_example = remove_single_relevant_sentence(
                        example.copy(),
                        list(sentence_pair)
                    )
                    results_2.append(new_example)
            
            if len(important_sentences) >= 3:
                results_3 = []
                for sentence_triplet in combinations(important_sentences, 3):
                    new_example = remove_single_relevant_sentence(
                        example.copy(),
                        list(sentence_triplet)
                    )
                    results_3.append(new_example)
            
            if len(important_sentences) >= 4:
                results_4 = []
                for sentence_quadruplet in combinations(important_sentences, 4):
                    new_example = remove_single_relevant_sentence(
                        example.copy(),
                        list(sentence_quadruplet)
                    )
                    results_4.append(new_example)
            
            if len(important_sentences) >= 5:
                results_5 = []
                for sentence_quintuplet in combinations(important_sentences, 5):
                    new_example = remove_single_relevant_sentence(
                        example.copy(),
                        list(sentence_quintuplet)
                    )
                    results_5.append(new_example)

            
            return results_1, results_2, results_3, results_4, results_5

        # Apply manipulation
        new_rows_1 = []
        new_rows_2 = []
        new_rows_3 = []
        new_rows_4 = []
        new_rows_5 = []

        for example in certain:
            results_1, results_2, results_3, results_4, results_5 = remove_each_relevant_sentence(example)

            new_rows_1.extend(results_1)
            new_rows_2.extend(results_2)
            new_rows_3.extend(results_3)
            new_rows_4.extend(results_4)
            new_rows_5.extend(results_5)

        manipulated_1 = Dataset.from_list(new_rows_1)
        manipulated_2 = Dataset.from_list(new_rows_2)
        manipulated_3 = Dataset.from_list(new_rows_3)
        manipulated_4 = Dataset.from_list(new_rows_4)
        manipulated_5 = Dataset.from_list(new_rows_5)

        certain = certain.map(lambda row: self.format_input(("criterion_type","criterion_text","note"), row))
        manipulated_1 = manipulated_1.map(lambda row: self.format_input(("criterion_type","criterion_text","_note"), row))
        manipulated_2 = manipulated_2.map(lambda row: self.format_input(("criterion_type","criterion_text","_note"), row))
        manipulated_3 = manipulated_3.map(lambda row: self.format_input(("criterion_type","criterion_text","_note"), row))
        manipulated_4 = manipulated_4.map(lambda row: self.format_input(("criterion_type","criterion_text","_note"), row))
        manipulated_5 = manipulated_5.map(lambda row: self.format_input(("criterion_type","criterion_text","_note"), row))


        #print('printing inseide manipulation')
        #print(certain[1])
        #print(manipulated[1])
        return certain, manipulated_1, manipulated_2, manipulated_3, manipulated_4, manipulated_5

    def get_full_and_partial_removal_variants(self):

        # Keep only certain eligibility labels
        certain = self.dataset['train'].filter(
            lambda x: x['expert_eligibility'] in {
                "included",
                "excluded",
                "not included"
            }
        )

        # Keep only relevant columns
        certain = certain.select_columns([
            'note',
            'criterion_text',
            'criterion_type',
            'expert_sentences',
            'annotation_id'
        ])

        def parse_expert_sentences(important_sentences):
            """
            Convert expert_sentences into a clean list of integers.
            Handles both lists and stringified lists.
            """
            if not important_sentences:
                return []

            important_sentences = ast.literal_eval(important_sentences)

            cleaned = []
            for x in important_sentences:
                cleaned.append(int(x))
            return cleaned

        def remove_sentences(example, removable_sentences):
            """
            Removes the given sentence indices from the numbered note blocks.
            """
            note = example['note']

            blocks = re.findall(
                r'^\d+\..*(?:\n(?!\d+\.).*)*',
                note,
                flags=re.MULTILINE
            )

            for sentence_idx in sorted(removable_sentences, reverse=True):
                if 0 <= sentence_idx < len(blocks):
                    del blocks[sentence_idx]

            new_example = example.copy()
            new_example['_note'] = "\n".join(blocks)
            new_example['removed_sentences'] = list(removable_sentences)

            return new_example

        def create_groups_for_example(example):
            important_sentences = parse_expert_sentences(
                example["expert_sentences"]
            )

            # If there are no important sentences, keep unchanged version.
            if not important_sentences:
                return [], []

            # --------------------------------------------------
            # Group 1:
            # Remove all important sentences.
            # --------------------------------------------------
            remove_all_example = remove_sentences(
                example.copy(),
                important_sentences
            )

            remove_all_group = [remove_all_example]

            # --------------------------------------------------
            # Group 2:
            # Remove every possible non-empty combination
            # except the complete set.
            #
            # Example: [1, 2, 3]
            # keep:
            #   [1], [2], [3],
            #   [1, 2], [1, 3], [2, 3]
            # exclude:
            #   [1, 2, 3]
            # --------------------------------------------------
            partial_removal_group = []

            n = len(important_sentences)

            for r in range(1, n):
                for removable_combo in combinations(important_sentences, r):
                    new_example = remove_sentences(
                        example.copy(),
                        list(removable_combo)
                    )
                    partial_removal_group.append(new_example)

            return remove_all_group, partial_removal_group

        # Apply manipulation
        remove_all_rows = []
        partial_removal_rows = []

        for example in certain:
            remove_all_group, partial_removal_group = create_groups_for_example(
                example
            )

            remove_all_rows.extend(remove_all_group)
            partial_removal_rows.extend(partial_removal_group)

        remove_all_info = Dataset.from_list(remove_all_rows)
        remove_every_possibility_except_complete = Dataset.from_list(
            partial_removal_rows
        )

        # Format original certain data
        certain = certain.map(
            lambda row: self.format_input(
                ("criterion_type", "criterion_text", "note"),
                row
            )
        )

        # Format group 1
        remove_all_info = remove_all_info.map(
            lambda row: self.format_input(
                ("criterion_type", "criterion_text", "_note"),
                row
            )
        )

        # Format group 2
        remove_every_possibility_except_complete = (
            remove_every_possibility_except_complete.map(
                lambda row: self.format_input(
                    ("criterion_type", "criterion_text", "_note"),
                    row
                )
            )
        )

        return (
            certain,
            remove_all_info,
            remove_every_possibility_except_complete
        )
    
    def plot(self, layers, masses_by_name, manipulation_type, out_dir, xLabel, yLabel, title):
        plt.figure(figsize=(10, 6))
        for name, mass in masses_by_name.items():
            if mass is None:
                continue

            plt.plot(
                layers,
                mass,
                label=name,
                marker="o",
            )
        
        if manipulation_type == "not_enough_info":
            delta = [f - r for f, r in zip(masses_by_name["not_enough_info"], masses_by_name["certain"])]
            plt.plot(layers, delta, label="Delta (not enough info - certain)", linestyle="--", color="darkgreen")
            # Annotate max delta layer
            if len(delta) > 0:
                idx_max = max(range(len(delta)), key=lambda i: delta[i])
                plt.scatter([layers[idx_max]], [delta[idx_max]], color="darkgreen", zorder=3)
                plt.annotate(
                f"max Δ @ L{layers[idx_max]}\n{delta[idx_max]:.2e}",
                    (layers[idx_max], delta[idx_max]),
                textcoords="offset points", xytext=(10, -10), ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#2e7d32", alpha=0.8)
                )
            plt.suptitle("Delta highlights where uncertainty begins to rise", fontsize=10, y=0.97)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(
            f"{title}\n{self.model_name}",
        )
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()
    
        out_path = f"{out_dir}/logit_lens_Mass.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
        plt.close()

    def _save_csv(self, layers, results_by_name, out_dir):
        out_path = f"{out_dir}/logit_lens_Mass.csv"
        with open(out_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            header = ["Layer"] + list(results_by_name.keys())
            writer.writerow(header)
            for i, layer in enumerate(layers):
                row = [layer] + [results_by_name[name][i] if results_by_name[name] is not None else None for name in results_by_name.keys()]
                writer.writerow(row)
        print(f"Saved CSV to {out_path}")

    def _save_run_info(self, out_dir, manipulation_type, counts_by_name, experiment, model = None, dataset = "TrialGPT-Criterion-Annotations"):
        if model == None: 
            model = self.model_name
        run_info = { "experiment": experiment, "model": model, "manipulation_type": manipulation_type, "dataset": dataset, "git_commit": self._get_git_commit(), "counts_by_name": counts_by_name}
        out_path = f"{out_dir}/run_info.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)

        print(f"Saved run info to {out_path}")

    def _get_git_commit(self):
        """
        Returns current git commit hash, or 'unknown' if unavailable.
        """

        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            return commit.decode("utf-8").strip()
        except Exception:
            return "unknown"


    def _run_model_with_cache(self, prompt):
        """Run model and return logits + cache."""
        logits, cache = self.model.run_with_cache(prompt)
        return logits, cache    
            