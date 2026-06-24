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

    def __init__(self, model_name, dataset, normal_prompt, uncertainty_prompt):
        self.model_name = model_name
        self.dataset = dataset
        self.normal_prompt = normal_prompt
        self.uncertainty_prompt = uncertainty_prompt 

        #setup model and device
        #self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")
        #self.model = HookedTransformer.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")
        self.model = HookedTransformer.from_pretrained_no_processing(model_name,torch_dtype=torch.float16, device="cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


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
    def getPreparedData(self):

        certain = self.dataset['train'].filter(lambda x: x['expert_eligibility'] in {"included", "not included", "excluded"})
        not_enough_info = self.dataset['train'].filter(lambda x: x['expert_eligibility'] in {"not enough information"})

        certain = certain.select_columns(['criterion_type','note', 'criterion_text'])
        not_enough_info = not_enough_info.select_columns(['criterion_type','note', 'criterion_text'])

        certain = certain.map(lambda row: self.format_input({'criterion_type','note', 'criterion_text'}, row))
        not_enough_info = not_enough_info.map(lambda row: self.format_input({'criterion_type','note', 'criterion_text'}, row))
        return certain, not_enough_info  

    #remove first relevant sentence from note, to create manipulated version of dataset
    def getManipulatedData3(self):

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
            'expert_sentences'
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

    def getManipulatedData5(self):

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
            'expert_sentences'
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

    def getManipulatedDataTwoGroups(self):

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
            'expert_sentences'
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



    def _run_model_with_cache(self, prompt):
        """Run model and return logits + cache."""
        logits, cache = self.model.run_with_cache(prompt)
        return logits, cache    
            

        


class LayerwiseResidualDifference(experiments_base):
    """Measures layerwise residual stream divergence between real and fake prompts."""

    def __init__(self, model_name, dataset, normal_prompt, uncertainty_prompt):
        super().__init__(model_name, dataset, normal_prompt, uncertainty_prompt)



    def compute_group_residual_diff(self, cache_group_a,cache_group_b):
        diffs = []
        
        for layer in range(self.model.cfg.n_layers):
            resid_a_sum = None
            resid_b_sum = None
            n = 0
            for i in range(len(cache_group_a)):
                cache_a = cache_group_a[i]
                cache_b = cache_group_b[i]
                resid_a = cache_a[f"blocks.{layer}.hook_resid_post"][0, -1] #nimmt für aktuellen layer den ersten batch und letzten token -> [d_model]
                resid_b = cache_b[f"blocks.{layer}.hook_resid_post"][0, -1]

                
                if resid_a_sum is None:
                    resid_a_sum = resid_a.clone()
                    resid_b_sum = resid_b.clone()
                else:
                    resid_a_sum += resid_a
                    resid_b_sum += resid_b
                n += 1
                
           

            resid_a_mean = resid_a_sum/n
            resid_b_mean = resid_b_sum/n

            # Unterschied der Gruppenmittelwerte
            diff = torch.norm(resid_a_mean - resid_b_mean).item()

            diffs.append(diff)
        return diffs

    def _plot_results(self, diffs):
        """Plot layerwise differences."""
        plt.figure(figsize=(10, 6))

        for name, diff in diffs.items():
            plt.plot(
                range(len(diff)),
                diff,
                marker="o",
                linewidth=2,
                markersize=8,
                label=name
            )

        plt.title(
            f"Layerwise Residual Difference: {self.model_name}",
            fontsize=14,
            fontweight="bold"
        )
        plt.xlabel("Layer", fontsize=12)
        plt.ylabel("Residual Norm Difference", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = f"layerwise_residual_diff_{self.model_name.replace('/', '_')}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
        plt.close()

    def run(self):
        results = []

        for i, row in enumerate(self.dataset_subset):
            if i >= 20:
                break

            normal_prompt = self.normal_prompt + row["input_text"]
            uncertainty_prompt = self.uncertainty_prompt + row["input_text"]
            """Execute the experiment."""
            print(f"\n{'='*60}")
            print(f"{i + 1}. Experiment: Layerwise Residual Difference")
            print(f"Model: {self.model_name}")
            print(f"{'='*60}")

            print("Running model on real prompt...")
            logits_real, cache_real = self._run_model_with_cache(self.normal_prompt)

            print("Running model on fake prompt...")
            logits_fake, cache_fake = self._run_model_with_cache(self.uncertainty_prompt)

            print("Computing layerwise residual differences...")
            layer_diffs = self._compute_residual_diff(cache_real, cache_fake)

            # Plot
            #output_path = os.path.join(self.model_dir, "layerwise_residual_diff.png")
            #self._plot_results(layer_diffs, output_path)

            #print(f"Results saved to {self.model_dir}")
            print(layer_diffs)
            results.append(layer_diffs)
        return results

    def run_over_dataset(self):

        """Executes the full logit lens analysis workflow for the initialized model."""
        print(f"\n{'='*60}")
        print(f"Processing model: {self.model_name}")
        print(f"{'='*60}")
        try:
            certain, not_enough_info = self.getPreparedData()
            Investigating_datasets = [("certain", certain), ("not enough information", not_enough_info)]
            all_layer_diffs = {}
            batch_certain = None
            for name, dataset in Investigating_datasets:
                """Execute the experiment."""
                print(f"\n{'='*60}")
                print(f"{name}. Experiment: Layerwise Residual Difference")
                print(f"Model: {self.model_name}")
                print(f"{'='*60}")
                batch_cache = []
                for i in range(len(dataset)):
                    prompt = self.normal_prompt + dataset[i]['input_text']
                    prompt = self.build_chat_prompt(prompt)
                    #logits, cache = self._run_model_with_cache(prompt)
                    
                    with torch.no_grad():
                        _, cache = self.model.run_with_cache(prompt, names_filter=lambda name: "hook_resid_post" in name, return_type=None)
                    batch_cache.append(cache)
                if batch_certain is None:
                    batch_certain = list(batch_cache)
                    
                print("Computing layerwise residual differences...")
                current_layer_diffs  = self.compute_group_residual_diff(batch_certain, batch_cache)
                all_layer_diffs[name]= current_layer_diffs
            self._plot_results(all_layer_diffs) #funktion _plot_results anpassen, damit sie mehrere Linien plotten kann, eine für certain und eine für not enough information
            return current_layer_diffs


        except Exception as e:
            print(f"Error processing {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Free memory per model
            if self.model is not None:
                del self.model
                self.model = None # Clear reference
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


class HeadwiseDivergence(experiments_base):
    """Analyzes which attention heads contribute most to divergence between real and fake prompts."""

    def __init__(self, model_name, dataset, normal_prompt, uncertainty_prompt):
        super().__init__(model_name, dataset, normal_prompt, uncertainty_prompt)

    def _compute_headwise_divergence_global(self, cache_group_a, cache_group_b):
        """Compute divergence per attention head."""
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        head_diffs = torch.zeros(n_layers, n_heads)

        for layer in range(n_layers):

            activation_sum_a = None
            activation_sum_b = None
            for i in range(len(cache_group_a)):

                cache_a = cache_group_a[i]
                cache_b = cache_group_b[i]
                a_heads = cache_a[f"blocks.{layer}.attn.hook_z"][0]
                b_heads = cache_b[f"blocks.{layer}.attn.hook_z"][0]

                min_len = min(a_heads.shape[0], b_heads.shape[0])
                
                if activation_sum_a is not None:
                    min_len = min(a_heads.shape[0], b_heads.shape[0], activation_sum_a.shape[0], activation_sum_b.shape[0])
                    activation_sum_a = activation_sum_a[:min_len, :, :]
                    activation_sum_b = activation_sum_b[:min_len, :, :]

                a_heads = a_heads[:min_len, :, :]
                b_heads = b_heads[:min_len, :, :]
        

                if activation_sum_a is None:
                    activation_sum_a = a_heads.clone()
                    activation_sum_b = b_heads.clone()
                else:
                    activation_sum_a += a_heads
                    activation_sum_b += b_heads
            
            activation_a = activation_sum_a/len(cache_group_a)
            activation_b = activation_sum_b/len(cache_group_b)

            for head in range(n_heads):
                diff = torch.norm(
                    activation_a[:, head, :] - activation_b[:, head, :]
                ).item()
                head_diffs[layer, head] = diff

        return head_diffs


    def _compute_headwise_divergence_mean(self, cache_group_a, cache_group_b):
        """Compute divergence per attention head."""
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        head_diffs = torch.zeros(n_layers, n_heads)

        for layer in range(n_layers):

            activation_sum_a = None
            activation_sum_b = None
            for i in range(len(cache_group_a)):

                cache_a = cache_group_a[i]
                cache_b = cache_group_b[i]
                a_heads = cache_a[f"blocks.{layer}.attn.hook_z"][0]
                b_heads = cache_b[f"blocks.{layer}.attn.hook_z"][0]

                min_len = min(a_heads.shape[0], b_heads.shape[0])
                
                if activation_sum_a is not None:
                    min_len = min(a_heads.shape[0], b_heads.shape[0], activation_sum_a.shape[0], activation_sum_b.shape[0])
                    activation_sum_a = activation_sum_a[:min_len, :, :]
                    activation_sum_b = activation_sum_b[:min_len, :, :]

                a_heads = a_heads[:min_len, :, :]
                b_heads = b_heads[:min_len, :, :]
        

                if activation_sum_a is None:
                    activation_sum_a = a_heads.clone()
                    activation_sum_b = b_heads.clone()
                else:
                    activation_sum_a += a_heads
                    activation_sum_b += b_heads
            
            activation_a = activation_sum_a/len(cache_group_a)
            activation_b = activation_sum_b/len(cache_group_b)

            for head in range(n_heads):
                diff = activation_a[:, head, :] - activation_b[:, head, :]

                # norm per token
                token_diffs = torch.norm(diff, dim=-1)

                # mean over tokens
                mean_diff = token_diffs.mean()

                head_diffs[layer, head] = mean_diff.item()

        return head_diffs

    def _compute_headwise_divergence_last_Token(self, cache_group_a, cache_group_b):
        """Compute divergence per attention head."""
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        head_diffs = torch.zeros(n_layers, n_heads)

        for layer in range(n_layers):

            activation_sum_a = None
            activation_sum_b = None
            for i in range(len(cache_group_a)):

                cache_a = cache_group_a[i]
                cache_b = cache_group_b[i]
                a_heads = cache_a[f"blocks.{layer}.attn.hook_z"][0][-1]
                b_heads = cache_b[f"blocks.{layer}.attn.hook_z"][0][-1]

                if activation_sum_a is None:
                    activation_sum_a = a_heads.clone()
                    activation_sum_b = b_heads.clone()
                else:
                    activation_sum_a += a_heads
                    activation_sum_b += b_heads
            
            activation_a = activation_sum_a/len(cache_group_a)
            activation_b = activation_sum_b/len(cache_group_b)

            for head in range(n_heads):
                last_diff = torch.norm(activation_a[head, :] - activation_b[head, :])
                head_diffs[layer, head] = last_diff.item()

        return head_diffs

    

    
    def run(self):
        """Execute the experiment."""
        print(f"\n{'='*60}")
        print(f"Experiment: Headwise Divergence")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}")

        certain, not_enough_info = self.getPreparedData()
        batch_cache_certain = []
        batch_cache_not_enough_info = []
        for i in range(10):
            print("Running model on certain prompt...")
            certain_prompt = self.normal_prompt + certain[i]['input_text']
            certain_prompt = self.build_chat_prompt(certain_prompt)
            logits_certain, cache_certain = self._run_model_with_cache(certain_prompt)
            batch_cache_certain.append(cache_certain)

            print("Running model on not enough information prompt...")
            not_enough_info_prompt = self.normal_prompt + not_enough_info[i]['input_text']
            not_enough_info_prompt = self.build_chat_prompt(not_enough_info_prompt)

            logits_not_enough_info, cache_not_enough_info = self._run_model_with_cache(not_enough_info_prompt)
            batch_cache_not_enough_info.append(cache_not_enough_info)
                

        print("Computing headwise divergence...")
        head_diffs = self._compute_headwise_divergence_global(batch_cache_certain, batch_cache_not_enough_info)
            

        # Plot heatmap
        self._plot_heatmap(head_diffs)

        # Print top heads
        self._print_top_heads(head_diffs, top_k=15)
        return head_diffs

    def _plot_heatmap(self, head_diffs):
        """Plot headwise divergence as heatmap."""
        plt.figure(figsize=(14, 8))
        im = plt.imshow(head_diffs.numpy(), aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(im, label="Head Divergence")
        plt.xlabel("Head Index", fontsize=12)
        plt.ylabel("Layer", fontsize=12)
        plt.title(f"Headwise Divergence Heatmap: {self.model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        plt.close()
    
    def _print_top_heads(self, head_diffs, top_k=10):
        """Print top-k most divergent heads."""
        flat_heads = sorted(
            [((l, h), head_diffs[l, h].item()) for l in range(self.model.cfg.n_layers)
             for h in range(self.model.cfg.n_heads)],
            key=lambda x: x[1], reverse=True
        )

        print(f"\nTop {top_k} Most Divergent Heads:")
        print(f"{'Rank':<6} {'Head':<10} {'Divergence':<15}")
        print(f"{'-'*35}")
        for i, ((layer, head), div) in enumerate(flat_heads[:top_k]):
            print(f"{i+1:<6} L{layer}H{head:<8} {div:<15.4f}")

        return flat_heads

class LayerwiseDifference(experiments_base):
    """Analyzes the difference in each layer compared to the last layer"""

    def __init__(self, model_name, dataset, normal_prompt, uncertainty_prompt):
        super().__init__(model_name, dataset, normal_prompt, uncertainty_prompt)

    def run(self):
        """Execute the experiment."""
        print(f"\n{'='*60}")
        print(f"Experiment: Difference to last layer")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}")

        certain, not_enough_info = self.getPreparedData()
        Investigating_datasets = [("certain", certain), ("not enough information", not_enough_info)]
        for name, dataset in Investigating_datasets:
            print(f"\n{'='*60}")
            print(f"Computing layerwise differences for {name}")
            print(f"{'='*60}")
            batch_cache = []
            for i in range(len(dataset)):
                prompt = self.normal_prompt + dataset[i]['input_text']
                prompt = self.build_chat_prompt(prompt)
                logits, cache = self._run_model_with_cache(prompt)
                batch_cache.append(cache)
                
            print("Computing headwise divergence...")
            head_diffs  = self._compute_layerwise_divergence_to_last_layer(batch_cache)
            head_diffs[name]= head_diffs
        self._plot_layerDiff(head_diffs, True)
        return head_diffs
    
    def _compute_layerwise_divergence_to_last_layer(self, cache_group_a):
        """Compute divergence per attention head."""
        n_layers = self.model.cfg.n_layers
        layer_activations=[]

        for layer in range(n_layers):

            activation_sum_a = None
            #pro prompt hat cache_group_a eine Liste von caches, da wir pro prompt einen cache haben. 
            # Wir müssen über alle caches iterieren und die Aktivierungen aufsummieren, um dann den Mittelwert zu bilden.
            for i in range(len(cache_group_a)):

                cache_a = cache_group_a[i]
                a_heads = cache_a[f"blocks.{layer}.attn.hook_z"][0][-1] #nimmt batch 0 und letzten token damit hat man die struktur [n_heads, d_head]. n_heads ist die Anzahl der Köpfe und d_head die Dimension jedes Kopfes.
                a_mean = a_heads.mean(dim=0) #Mittelt über die Köpfe. Jetzt hat man die durchschnittliche Aktivierung über alle Köpfe für diesen Layer (für diesen prompt). Die Struktur ist jetzt [d_head], also die durchschnittliche Aktivierung pro Dimension des Kopfes.

                if activation_sum_a is None:
                    activation_sum_a = a_mean.clone()
                else:
                    activation_sum_a += a_mean
            
            layer_activations.append(activation_sum_a/len(cache_group_a)) #jetzt haben wir die durchschnittliche Aktivierung über alle Prompts für diesen Layer. Und höngen es an die Liste die später alle Layeraktivierungen enthält. 

        # Vergleich mit letztem Layer
        layer_diffs = torch.zeros(n_layers)
        last_activation = layer_activations[-1]

        for layer in range(n_layers):

            diff = torch.norm(
                layer_activations[layer] - last_activation
            )

            layer_diffs[layer] = diff

        return layer_diffs
    
    def _plot_layerDiff(self, curves):
        plt.figure(figsize=(10, 6))

        for name, curve in curves.items():
            plt.plot(
                range(len(curve)),
                curve.numpy(),
                marker="o",
                label=name
            )

        plt.xlabel("Layer")
        plt.ylabel("Cosine distance to final layer")
        plt.title("Layerwise distance to final representation")
        plt.legend()
        plt.grid(True, alpha=0.4)
        out_path = f"logit_lens_Mass_{self.model_name.replace('/', '_')}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
        plt.close()




class LogitLensAnalysis(experiments_base):
    def __init__(self, model_name, dataset, normal_prompt, uncertainty_prompt):
        super().__init__(model_name, dataset, normal_prompt, uncertainty_prompt)

    def _mass_by_layer(self, prompt, token_ids):
        """Compute uncertainty mass per layer using logit lens on the last position."""
        # Cache only resid_post to reduce memory
        print(f"entering _mass_by_layer for prompt: '{prompt[:50]}...' ")
        print('this is the prompt the model receives')
        print(prompt)
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(
                prompt,
                names_filter=lambda n: ("hook_resid_post" in n),
            )
        print(f"exiting _mass_by_layer")
        masses = []
        for layer in range(self.model.cfg.n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]  # [d_model]
            # Project through ln_final + unembed to get pseudo-logits
            try:
                pseudo_logits = self.model.unembed(self.model.ln_final(resid))  # [vocab]
            except Exception:
                # Fallback: some models may require different ordering; try unembed then ln_final
                pseudo_logits = self.model.ln_final(self.model.unembed(resid))
            probs = torch.softmax(pseudo_logits, dim=-1)
            masses.append(probs[token_ids].sum().item())
        return masses


    def entropy_by_layer(self, prompt):
        """Compute entropy for each layer"""
        # Cache only resid_post to reduce memory
        print(f"entering _entropy_by_layer for prompt: '{prompt[:50]}...' ")
        logits, cache = self.model.run_with_cache(
            prompt,
            names_filter=lambda n: ("hook_resid_post" in n),
        )
        masses = []
        for layer in range(self.model.cfg.n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]  # [d_model]
            # Project through ln_final + unembed to get pseudo-logits
            try:
                pseudo_logits = self.model.unembed(self.model.ln_final(resid))  # [vocab]
            except Exception:
                # Fallback: some models may require different ordering; try unembed then ln_final
                pseudo_logits = self.model.ln_final(self.model.unembed(resid))
            log_probs = torch.log_softmax(pseudo_logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum()
            masses.append(entropy)

        print(f"exiting entropy_by_layer")

        return masses

    def change_by_layer(self, prompt):
        logits, cache = self.model.run_with_cache(
            prompt,
            names_filter=lambda n: "hook_resid_post" in n,
        )

        prev_resid = None
        instabilities = []

        for layer in range(self.model.cfg.n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]

            if prev_resid is not None:
                cos_sim = torch.nn.functional.cosine_similarity(
                    resid, prev_resid, dim=0
                )
                instabilities.append((1 - cos_sim).item())
            else:
                instabilities.append(0)

            prev_resid = resid

        return instabilities
    


    #just to replicate old results
    def run(self):
        """Executes the full logit lens analysis workflow for the initialized model."""
        print(f"\n{'='*60}")
        print(f"Processing model: {self.model_name}")
        print(f"{'='*60}")
        try:

            u_ids = self.getUncertaintyTokens()
            print(f"Uncertainty token ids: {len(u_ids)} found")
            if not u_ids:
                print("Warning: No uncertainty tokens found; results may be uninformative.")

            print("Computing masses for real prompt...")
            real_mass = self._mass_by_layer(self.normal_prompt, u_ids)

            print("Computing masses for fictional prompt...")
            fake_mass = self._mass_by_layer(self.uncertainty_prompt, u_ids)

            layers = list(range(self.model.cfg.n_layers))

            print("Plotting and saving...")
            self._plot_logit_lens(layers, real_mass, fake_mass)
            self._print_layerwise_table(layers, real_mass, fake_mass)

        except Exception as e:
            print(f"Error processing {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Free memory per model
            if self.model is not None:
                del self.model
                self.model = None # Clear reference
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def run_analysis(self):
        """Executes the full logit lens analysis workflow for the initialized model."""
        print(f"\n{'='*60}")
        print(f"Processing model: {self.model_name}")
        print(f"{'='*60}")
        try:
            
            certain_mass = None
            not_enough_info_mass = None
            n = 0
            u_ids = self.getUncertaintyTokens()

            #certain, manipulated_1, manipulated_2, manipulated_3, manipulated_4, manipulated_5 = self.getManipulatedData5()
            #certain, fully_manipulated, partially_manipulated = self.getManipulatedDataTwoGroups()
            certain, manipulated_1, manipulated_2, manipulated_3 = self.getManipulatedData3()
            #certain, not_enough_info = self.getPreparedData()
            Investigating_datasets = [
                ("certain", certain),
                #("not_enough_info", not_enough_info),
                #("fully_manipulated", fully_manipulated),
                #("partially_manipulated", partially_manipulated),
                ("manipulated_1", manipulated_1),
                ("manipulated_2", manipulated_2),
                ("manipulated_3", manipulated_3),
                #("manipulated_4", manipulated_4),
                #("manipulated_5", manipulated_5),
            ]
            manipulated_masses = {}
            counting = []

            for name, dataset in Investigating_datasets:
                print(f"\n{'='*60}")
                print(f"Computing masses for {name}")
                print(f"{'='*60}")

                total_mass = None
                n = 0

                for i in range(len(dataset)):
                    print(f"Computing masses for {name} prompt {i+1}/{len(dataset)}...")

                    prompt = self.normal_prompt + dataset[i]["input_text"]
                    prompt = self.build_chat_prompt(prompt)

                    #new_mass = self._mass_by_layer(prompt, u_ids)
                    #new_certain_mass = self.entropy_by_layer(certain_prompt)
                    new_mass = self.change_by_layer(prompt)
                    #response = self.get_model_response(certain_prompt)
                    #self._print_response(response, i, certain_prompt)

                    if total_mass is None:
                        total_mass = new_mass
                    else:
                        total_mass = [
                            a + b for a, b in zip(total_mass, new_mass)
                        ]

                    n += 1
                counting.append((name, n))
                if n > 0:
                    average_mass = [x / n for x in total_mass]
                else:
                    average_mass = None

                manipulated_masses[name] = average_mass
            layers = list(range(self.model.cfg.n_layers))
            print("Plotting and saving...")
            #self._plot_logit_lens(layers, manipulated_masses["certain"], manipulated_masses["not_enough_info"])
            #self._plot_logit_lens(layers, manipulated_masses["certain"], manipulated_masses["manipulated_1"],manipulated_masses["manipulated_2"],manipulated_masses["manipulated_3"],manipulated_masses["manipulated_4"],manipulated_masses["manipulated_5"])
            self._plot_logit_lens(layers, manipulated_masses["certain"], manipulated_masses["manipulated_1"],manipulated_masses["manipulated_2"],manipulated_masses["manipulated_3"])
            #self._plot_logit_lens(layers, manipulated_masses["certain"], manipulated_masses["fully_manipulated"],manipulated_masses["partially_manipulated"])
            print(counting)
            #self._print_layerwise_table(layers, certain_mass, not_enough_info_mass)

        except Exception as e:
            print(f"Error processing {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Free memory per model
            if self.model is not None:
                del self.model
                self.model = None # Clear reference
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


    def _print_response(self, response, i, prompt):
        print("\n" + "="*80)
        print(f"PROMPT {i}")
        print(prompt)
        print("\nMODEL RESPONSE:")
        print(response)
        print("="*80)
        
    def _print_layerwise_table(self, layers, certain_mass, not_enough_info_mass):
      """Print (and optionally write) a descriptive table of layer-wise uncertainty masses."""
      delta = [f - r for f, r in zip(not_enough_info_mass, certain_mass)]

      header = f"\nAverage Layer-wise logit lens uncertainty mass (over prompts) ({self.model_name})\n"
      header += "Layer  |   certain Mass    |  Not enough Info Mass    |   Delta\n"
      header += "------ | -------------- | ------------- | --------------"

      lines = [header]

      for l, r, f, d in zip(layers, certain_mass, not_enough_info_mass, delta):
          lines.append(
              f"{l:>5}  | {r:>14.3e} | {f:>13.3e} | {d:>14.3e}"
          )

      table = "\n".join(lines)

      # Print to console
      print(table)
    #def _plot_logit_lens(self, layers, certain_mass, manipulated_1_mass, manipulated_2_mass):
    #def _plot_logit_lens(self, layers, certain_mass, manipulated_1_mass, manipulated_2_mass, manipulated_3_mass, manipulated_4_mass, manipulated_5_mass):
    #def _plot_logit_lens(self, layers, certain_mass, not_enough_info_mass):
    #def _plot_logit_lens(self, layers, certain_mass, fully_manipulated_mass, partially_manipulated_mass):
    
    
    def _plot_logit_lens(self, layers, certain_mass, manipulated_1_mass, manipulated_2_mass, manipulated_3_mass):
        plt.figure(figsize=(10, 6))
        plt.plot(layers, certain_mass, label="certain prompt", marker="o", color="steelblue")
        #plt.plot(layers, not_enough_info_mass, label="not enough info", marker="o", color="coral")
        #plt.plot(layers, fully_manipulated_mass, label="fully manipulated", marker="o", color="seagreen")
        #plt.plot(layers, partially_manipulated_mass, label="partially manipulated", marker="o", color="darkgreen")
        plt.plot(layers, manipulated_1_mass, label="manipulated_1", marker="o", color="seagreen")
        plt.plot(layers, manipulated_2_mass, label="manipulated_2", marker="o", color="darkgreen")
        plt.plot(layers, manipulated_3_mass, label="manipulated_3", marker="o", color="purple")
        #plt.plot(layers, manipulated_4_mass, label="manipulated_4", marker="o", color="orange")
        #plt.plot(layers, manipulated_5_mass, label="manipulated_5", marker="o", color="brown")
        #delta = [f - r for f, r in zip(not_enough_info_mass, certain_mass)]
        #plt.plot(layers, delta, label="Delta (not enough info - certain)", linestyle="--", color="darkgreen")
        # Annotate max delta layer
        #if len(delta) > 0:
        #    idx_max = max(range(len(delta)), key=lambda i: delta[i])
        #    plt.scatter([layers[idx_max]], [delta[idx_max]], color="darkgreen", zorder=3)
        #    plt.annotate(
        #       f"max Δ @ L{layers[idx_max]}\n{delta[idx_max]:.2e}",
        #        (layers[idx_max], delta[idx_max]),
        #       textcoords="offset points", xytext=(10, -10), ha="left",
        #        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#2e7d32", alpha=0.8)
        #    )
        plt.xlabel("Layer")
        plt.ylabel("Average (over prompts) Change by Layer")
        plt.title(
            f"Change per Layer\n{self.model_name}",
        )
        #plt.suptitle("Delta highlights where uncertainty begins to rise", fontsize=10, y=0.97)
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()
        out_path = f"logit_lens_Mass_{self.model_name.replace('/', '_')}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
        plt.close()

class VocabularyAnalysis(experiments_base):
    def __init__(self, model_name, dataset, normal_prompt, uncertainty_prompt):
        super().__init__(model_name, dataset, normal_prompt, uncertainty_prompt)

    def compute_pair_token_delta(self,real_logits, fictional_logits):
        """
        Berechnet Δ(t) = p(t | fictional) - p(t | real)
        für ein einzelnes Real/Fictional-Prompt-Paar.

        real_logits: Tensor[vocab_size]
        fictional_logits: Tensor[vocab_size]
        tokenizer: HuggingFace tokenizer
        top_k: Anzahl der Tokens mit größtem positiven Delta

        Returns:
            List[dict]
        """

        p_real = torch.softmax(real_logits.float(), dim=-1)
        p_fictional = torch.softmax(fictional_logits.float(), dim=-1)

        delta = p_fictional - p_real

        return p_real, p_fictional, delta

    
    
    def run(self):
        """Executes the full vocabulary analysis workflow for the initialized model."""
        print(f"\n{'='*60}")
        print(f"Processing model: {self.model_name}")
        print(f"{'='*60}")
        try:
            print(f"Loading {self.model_name}...")

            certain, not_enough_info,_,_,_,_ = self.getManipulatedData()
            sum_p_real = None
            sum_p_fictional = None
            sum_delta = None

            for i in range(len(certain)):
                print("Running model on certain prompt...")
                certain_prompt = self.normal_prompt + certain[i]['input_text']
                certain_prompt = self.build_chat_prompt(certain_prompt)
                with torch.no_grad():
                    logits_certain = self.model(certain_prompt)

                print("Running model on not enough information prompt...")
                not_enough_info_prompt = self.normal_prompt + not_enough_info[i]['input_text']
                not_enough_info_prompt = self.build_chat_prompt(not_enough_info_prompt)
                with torch.no_grad():
                    logits_not_enough_info = self.model(not_enough_info_prompt)

                print("Computing token deltas...")
                p_real, p_fictional, delta = self.compute_pair_token_delta(logits_certain[0, -1, :], logits_not_enough_info[0, -1, :])
                sum_p_real = p_real if sum_p_real is None else sum_p_real + p_real
                sum_p_fictional = p_fictional if sum_p_fictional is None else sum_p_fictional + p_fictional
                sum_delta = delta if sum_delta is None else sum_delta + delta
                # nach compute_pair_token_delta(...)
                del logits_certain
                del logits_not_enough_info
                torch.cuda.empty_cache()
            
            top_ids = torch.argsort(sum_delta, descending=True)[:50]

            print("\nTop 50 Δ-Tokens")
            print("=" * 80)

            for rank, token_id in enumerate(top_ids, start=1):
                token_id = token_id.item()

                token = self.tokenizer.decode([token_id])

                print(
                    f"{rank:2d}. "
                    f"{repr(token):20s} "
                    f"Δ={sum_delta[token_id]:.6f} "
                    f"P_fic={sum_p_fictional[token_id]:.6f} "
                    f"P_real={sum_p_real[token_id]:.6f}"
                )

            

        except Exception as e:
            print(f"Error processing {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Free memory per model
            if self.model is not None:
                del self.model
                self.model = None # Clear reference
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


class PrincipalComponentAnalysis(experiments_base):
    def __init__(self, model_name, dataset, normal_prompt, uncertainty_prompt):
        super().__init__(model_name, dataset, normal_prompt, uncertainty_prompt)

    def _get_last_token_residual(self, prompt, layer=-1):
        """
        Extract residual stream activation at the last token.
        layer=-1 means last transformer layer.
        """
        logits, cache = self.model.run_with_cache(
            prompt,
            names_filter=lambda name: "hook_resid_post" in name
        )

        if layer == -1:
            layer = self.model.cfg.n_layers - 1

        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]

        return resid.detach().cpu().float().numpy()

    def run(self, layer=19, max_examples_per_group=None):
        certain, not_enough_info = self.getPreparedData()

        activations = []
        labels = []

        # certain examples
        n_certain = len(certain)
        if max_examples_per_group is not None:
            n_certain = min(n_certain, max_examples_per_group)

        for i in range(n_certain):
            print(f"Processing certain example {i+1}/{n_certain}")

            prompt = self.normal_prompt + certain[i]["input_text"]
            prompt = self.build_chat_prompt(prompt)

            resid = self._get_last_token_residual(prompt, layer=layer)

            activations.append(resid)
            labels.append("certain")

        # not enough information examples
        n_uncertain = len(not_enough_info)
        if max_examples_per_group is not None:
            n_uncertain = min(n_uncertain, max_examples_per_group)

        for i in range(n_uncertain):
            print(f"Processing not_enough_info example {i+1}/{n_uncertain}")

            prompt = self.normal_prompt + not_enough_info[i]["input_text"]
            prompt = self.build_chat_prompt(prompt)

            resid = self._get_last_token_residual(prompt, layer=layer)

            activations.append(resid)
            labels.append("not_enough_info")

        # Convert to matrix: [num_examples, d_model]
        X = np.stack(activations)

        # PCA to 2 dimensions
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        explained_variance = pca.explained_variance_ratio_

        # Plot
        plt.figure(figsize=(9, 7))

        labels_array = np.array(labels)

        certain_mask = labels_array == "certain"
        uncertain_mask = labels_array == "not_enough_info"

        plt.scatter(
            X_pca[certain_mask, 0],
            X_pca[certain_mask, 1],
            label="certain",
            alpha=0.7,
            color="steelblue"
        )

        plt.scatter(
            X_pca[uncertain_mask, 0],
            X_pca[uncertain_mask, 1],
            label="not enough information",
            alpha=0.7,
            color="coral"
        )

        if layer == -1:
            layer_title = self.model.cfg.n_layers - 1
        else:
            layer_title = layer

        plt.xlabel(f"PC1 ({explained_variance[0] * 100:.2f}% variance)")
        plt.ylabel(f"PC2 ({explained_variance[1] * 100:.2f}% variance)")
        plt.title(
            f"PCA of last-token residual stream\n"
            f"{self.model_name}, layer {layer_title}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = f"logit_lens_Mass_{self.model_name.replace('/', '_')}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
        plt.close()
        

        return X_pca, labels, pca
    
    import torch


class UncertaintyVocabularyAnalysis(experiments_base):
    def __init__(
        self,
        model_name,
        dataset,
        normal_prompt,
        uncertainty_prompt,
        seed_words=None,
        top_k=100,
        max_positions=None,
        eps=1e-9,
    ):
        super().__init__(model_name, dataset, normal_prompt, uncertainty_prompt)

        self.top_k = top_k
        self.max_positions = max_positions
        self.eps = eps

        self.seed_words = seed_words or [
            "unknown", "unsure", "don", "'t", "know", "can't", "unable", "impossible",
            "doesn't", "exist", "fictional", "imaginary", "no", "such", "thing",
            "hypothetical", "made", "up", "never", "happened",
            "insufficient", "unclear", "not", "enough", "information",
            "cannot", "determine", "missing", "unavailable", "uncertain"
        ]

    def extract_logits(self, model_output):
        """
        Makes the class robust to both:
        - raw logits tensors
        - HuggingFace outputs with .logits
        """
        if hasattr(model_output, "logits"):
            return model_output.logits
        return model_output

    def pooled_token_distribution(self, logits):
        """
        Pools token probabilities over all token positions.

        logits: Tensor[batch, seq_len, vocab_size] or Tensor[seq_len, vocab_size]

        Returns:
            Tensor[vocab_size]
            Average probability distribution over all selected sequence positions.
        """

        logits = self.extract_logits(logits)

        if logits.dim() == 3:
            logits = logits[0] #nur noch [seq_len, vocab_size]

        if self.max_positions is not None:
            logits = logits[-self.max_positions:]

        probs = torch.softmax(logits.float(), dim=-1) #berechnet die Wkeit über das gesamte Vokabular für jedes Token in der Sequenz

        pooled_probs = probs.mean(dim=0) #mittel über die Token-Positionen, um eine durchschnittliche Wahrscheinlichkeitsverteilung über das Vokabular zu erhalten

        return pooled_probs

    def compute_pair_token_statistics(self, certain_logits, uncertain_logits):
        """
        Computes pooled token probabilities for one certain / uncertain pair.

        Returns:
            p_certain: Tensor[vocab_size]
            p_uncertain: Tensor[vocab_size]
            delta: Tensor[vocab_size]
        """

        p_certain = self.pooled_token_distribution(certain_logits) #für jeden token im Vokabular die durchschnittliche Wahrscheinlichkeit über die Sequenzpositionen für den "certain" Prompt
        p_uncertain = self.pooled_token_distribution(uncertain_logits) #für jeden token im Vokabular die durchschnittliche Wahrscheinlichkeit über die Sequenzpositionen für den "uncertain" Prompt

        delta = p_uncertain - p_certain

        return p_certain, p_uncertain, delta

    def get_seed_token_ids(self):
        """
        Converts seed uncertainty words into token ids.

        This includes variants with and without leading whitespace because
        tokenizers often represent word-initial tokens differently.
        """

        token_ids = set()

        variants = []

        for word in self.seed_words:
            variants.append(word)
            variants.append(" " + word)
            variants.append(word.capitalize())
            variants.append(" " + word.capitalize())

        for text in variants:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            for token_id in ids:
                token_ids.add(token_id)

        return token_ids

    def is_candidate_token(self, token_id):
        """
        Filters out tokens that are usually not useful as uncertainty indicators.
        Keeps subword tokens, but removes pure whitespace, punctuation-only tokens,
        and special tokens.
        """

        if token_id in set(self.tokenizer.all_special_ids):
            return False

        token = self.tokenizer.decode([token_id])
        stripped = token.strip()

        if stripped == "":
            return False

        if len(stripped) == 1 and not stripped.isalnum():
            return False

        return True

    def compute_global_scores(
        self,
        sum_p_certain,
        sum_p_uncertain,
        sumsq_p_certain,
        sumsq_p_uncertain,
        n,
    ):
        """
        Computes global vocabulary scores after processing all examples.

        Returns:
            mean_p_certain
            mean_p_uncertain
            delta
            log_odds
            effect_size
            score
        """

        mean_p_certain = sum_p_certain / n #durchschnittliche Wahrscheinlichkeit jedes Tokens über alle "certain" Prompts
        mean_p_uncertain = sum_p_uncertain / n

        var_p_certain = (sumsq_p_certain / n) - (mean_p_certain ** 2) #Varianz jedes Tokens über alle "certain" Prompts
        var_p_uncertain = (sumsq_p_uncertain / n) - (mean_p_uncertain ** 2) #Varianz jedes Tokens über alle "uncertain" Prompts

        var_p_certain = torch.clamp(var_p_certain, min=0.0)
        var_p_uncertain = torch.clamp(var_p_uncertain, min=0.0)

        pooled_std = torch.sqrt((var_p_certain + var_p_uncertain) / 2.0 + self.eps)#Pooled Standardabweichung jedes Tokens über beide Gruppen

        delta = mean_p_uncertain - mean_p_certain

        log_odds = torch.log(mean_p_uncertain + self.eps) - torch.log(mean_p_certain + self.eps) #log odds > 0 bedeutet, dass das Token in "uncertain" Prompts wahrscheinlicher ist als in "certain" Prompts

        effect_size = delta / pooled_std #How large is this difference compared to the variation across examples?

        score = log_odds * effect_size #kombiniert log odds und effect size zu einem Gesamtscore, der sowohl die Richtung als auch die Stärke der Unterscheidungskraft eines Tokens berücksichtigt

        return (
            mean_p_certain,
            mean_p_uncertain,
            delta,
            log_odds,
            effect_size,
            score,
        )

    def print_ranked_tokens(
        self,
        mean_p_certain,
        mean_p_uncertain,
        delta,
        log_odds,
        effect_size,
        score,
        title,
        token_ids,
    ):
        print(f"\n{title}")
        print("=" * 100)

        rank = 1

        for token_id in token_ids:
            token_id = token_id.item() if torch.is_tensor(token_id) else token_id

            if not self.is_candidate_token(token_id):
                continue

            token = self.tokenizer.decode([token_id])

            print(
                f"{rank:3d}. "
                f"{repr(token):25s} "
                f"score={score[token_id]: .6f} "
                f"Δ={delta[token_id]: .8f} "
                f"log_odds={log_odds[token_id]: .6f} "
                f"effect={effect_size[token_id]: .6f} "
                f"P_uncertain={mean_p_uncertain[token_id]: .8f} "
                f"P_certain={mean_p_certain[token_id]: .8f}"
            )

            rank += 1

            if rank > self.top_k:
                break

    def print_seed_word_scores(
        self,
        mean_p_certain,
        mean_p_uncertain,
        delta,
        log_odds,
        effect_size,
        score,
    ):
        print("\nSeed uncertainty token scores")
        print("=" * 100)

        seed_token_ids = sorted(list(self.get_seed_token_ids()))

        rows = []

        for token_id in seed_token_ids:
            if not self.is_candidate_token(token_id):
                continue

            token = self.tokenizer.decode([token_id])

            rows.append({
                "token_id": token_id,
                "token": token,
                "score": score[token_id].item(),
                "delta": delta[token_id].item(),
                "log_odds": log_odds[token_id].item(),
                "effect_size": effect_size[token_id].item(),
                "p_uncertain": mean_p_uncertain[token_id].item(),
                "p_certain": mean_p_certain[token_id].item(),
            })

        rows = sorted(rows, key=lambda x: x["score"], reverse=True)

        for rank, row in enumerate(rows, start=1):
            print(
                f"{rank:3d}. "
                f"{repr(row['token']):25s} "
                f"score={row['score']: .6f} "
                f"Δ={row['delta']: .8f} "
                f"log_odds={row['log_odds']: .6f} "
                f"effect={row['effect_size']: .6f} "
                f"P_uncertain={row['p_uncertain']: .8f} "
                f"P_certain={row['p_certain']: .8f}"
            )

    def run(self):
        """Executes pooled uncertainty vocabulary analysis."""

        print(f"\n{'=' * 60}")
        print(f"Processing model: {self.model_name}")
        print(f"{'=' * 60}")

        print(f"Loading {self.model_name}...")

        certain, _, _, not_enough_info, _, _ = self.getManipulatedData()

        sum_p_certain = None
        sum_p_uncertain = None

        sumsq_p_certain = None
        sumsq_p_uncertain = None

        n = 0

        for i in range(min(len(certain), len(not_enough_info))):
            print(f"\nExample {i + 1}/{len(certain)}")

            print("Running model on certain prompt...")
            certain_prompt = self.normal_prompt + certain[i]["input_text"]
            certain_prompt = self.build_chat_prompt(certain_prompt)

            with torch.no_grad():
                logits_certain = self.model(certain_prompt)

            print("Running model on not enough information prompt...")
            uncertain_prompt = self.normal_prompt + not_enough_info[i]["input_text"]
            uncertain_prompt = self.build_chat_prompt(uncertain_prompt)

            with torch.no_grad():
                logits_uncertain = self.model(uncertain_prompt)

            print("Pooling token probabilities over all positions...")
            p_certain, p_uncertain, _ = self.compute_pair_token_statistics(
                logits_certain,
                logits_uncertain,
            )

            if sum_p_certain is None:
                sum_p_certain = torch.zeros_like(p_certain)
                sum_p_uncertain = torch.zeros_like(p_uncertain)
                sumsq_p_certain = torch.zeros_like(p_certain)
                sumsq_p_uncertain = torch.zeros_like(p_uncertain)

            sum_p_certain += p_certain
            sum_p_uncertain += p_uncertain

            sumsq_p_certain += p_certain ** 2
            sumsq_p_uncertain += p_uncertain ** 2

            n += 1

            del logits_certain
            del logits_uncertain
            del p_certain
            del p_uncertain

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        (
            mean_p_certain,
            mean_p_uncertain,
            delta,
            log_odds,
            effect_size,
            score,
        ) = self.compute_global_scores(
            sum_p_certain,
            sum_p_uncertain,
            sumsq_p_certain,
            sumsq_p_uncertain,
            n,
        )

        positive_mask = delta > 0

        masked_score = score.clone()
        masked_score[~positive_mask] = -float("inf")

        top_score_ids = torch.argsort(masked_score, descending=True)

        self.print_ranked_tokens(
            mean_p_certain,
            mean_p_uncertain,
            delta,
            log_odds,
            effect_size,
            score,
            title="Top uncertainty-associated tokens by score",
            token_ids=top_score_ids,
        )

        top_delta_ids = torch.argsort(delta, descending=True)

        self.print_ranked_tokens(
            mean_p_certain,
            mean_p_uncertain,
            delta,
            log_odds,
            effect_size,
            score,
            title="Top uncertainty-associated tokens by raw probability delta",
            token_ids=top_delta_ids,
        )

        self.print_seed_word_scores(
            mean_p_certain,
            mean_p_uncertain,
            delta,
            log_odds,
            effect_size,
            score,
        )

        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class PromptTokenComparison(experiments_base):
    def __init__(self, model_name, dataset, normal_prompt, uncertainty_prompt, top_k=20):
        super().__init__(model_name, dataset, normal_prompt, uncertainty_prompt)
        self.top_k = top_k

    def get_probs(self, prompt):
        with torch.no_grad():
            logits = self.model(prompt)

        if hasattr(logits, "logits"):
            logits = logits.logits

        if logits.dim() == 3:
            logits = logits[0,-1,:]

        probs = torch.softmax(logits.float(), dim=-1)
        return probs

    def print_top_delta(self, p_certain, p_uncertain):
        delta = p_uncertain - p_certain
        top_ids = torch.argsort(delta, descending=True)[:self.top_k]

        print(f"{'Rank':<5} {'Token':<20} {'P_certain':<14} {'P_uncertain':<14} {'Delta':<14}")
        print("-" * 75)

        for rank, token_id in enumerate(top_ids, start=1):
            token_id = token_id.item()
            token = self.tokenizer.decode([token_id]).replace("\n", "\\n")

            print(
                f"{rank:<5} "
                f"{repr(token):<20} "
                f"{p_certain[token_id].item():<14.6e} "
                f"{p_uncertain[token_id].item():<14.6e} "
                f"{delta[token_id].item():<14.6e}"
            )

    def run(self):
        data = self.getManipulatedData()

        certain = data[0]
        uncertain = data[3] if len(data) > 3 else data[1]

        for i in range(min(len(certain), len(uncertain))):
            print("\n" + "=" * 80)
            print(f"Prompt pair {i + 1}")
            print("=" * 80)

            certain_prompt = self.normal_prompt + certain[i]["input_text"]
            uncertain_prompt = self.normal_prompt + uncertain[i]["input_text"]

            if hasattr(self, "build_chat_prompt"):
                certain_prompt = self.build_chat_prompt(certain_prompt)
                uncertain_prompt = self.build_chat_prompt(uncertain_prompt)

            p_certain = self.get_probs(certain_prompt)
            p_uncertain = self.get_probs(uncertain_prompt)

            self.print_top_delta(p_certain, p_uncertain)

            del p_certain, p_uncertain
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    


#model_name ="gpt2"
#model_name ="Qwen/Qwen2.5-0.5B-Instruct"
#model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
#model_name = "google/gemma-2-9b-it"
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)
#model_name = "meta-llama/Llama-3.2-1B-Instruct"
dataset = load_dataset("ncbi/TrialGPT-Criterion-Annotations")


#normal_prompt = "Youre Job is to classify the following data into different categories. The information about the type of criterea comes after criterion_type. If it is a Inclusion criteria, you should classify it as included, not included, not enough information or not applicable. If it is a exclustion criteria you should classify it as Excluded, not excluded, not enough information or not applicable" #+ dataset_subset[1] ['input_text'] 
#uncertainty_prompt = "Youre Job is to classify the following data into different categories. The information about the type of criterea comes after criterion_type. If it is a Inclusion criteria, you should classify it as included, not included, not enough information or not applicable. If it is a exclustion criteria you should classify it as Excluded, not excluded, not enough information or not applicable. if you are uncertain, please tell me"
#uncertainty_prompt = "tell me if youre uncertain!"




#prompt="In the follwoing you will receive a Patient file and a study criteria. Please classify the patient as a match or missmatch for the criteria. "
#prompt = (
#    "You will receive a criterion_text and a patient note. "
#    "Decide if the patient matches the criterion. "
#    "Answer only with one of: match, no match, not enough information. \n\n"
#)
#prompt = "Does the patient match the criterion? Answer with: yes, no, unknown. \n\n"
prompt=""
uncertainty_prompt=""

#2.0
expResDiff = LayerwiseResidualDifference(model_name, dataset, prompt, uncertainty_prompt)
expResDiff.run_over_dataset()


#2.1 - Uncertainty mass measuring
#exp2 = LogitLensAnalysis(model_name, dataset, prompt, uncertainty_prompt)
#exp2.run_analysis()

#2.2
#expHeadDiv = HeadwiseDivergence(model_name, dataset, prompt, uncertainty_prompt)
#expHeadDiv.run()

#3.0
# Prompts (changeable)
#prompt_real = "What is the freezing point of water at standard atmospheric pressure?"
#prompt_fake = "What is the freezing point of water on Planet Xylon, where gravity is twice that of Earth?"
#analyzer = LogitLensAnalysis(
#    model_name,
#    dataset,
#   prompt_real,
#    prompt_fake,
#)
#analyzer.run()


#3.2.3 - difference to last layer 
#exp323 = LayerwiseDifference(model_name, dataset, prompt, uncertainty_prompt)
#exp323.run()

#5.3.1 - Vocabulary analysis
#exp531 = VocabularyAnalysis(model_name, dataset, prompt, uncertainty_prompt)
#exp531.run()

#5.4.1 - PCA
#exp541 = PrincipalComponentAnalysis(model_name, dataset,prompt,uncertainty_prompt)
#exp541.run()

#6.1.1 - Uncertainty Vocabulary Analysis
#exp661 = UncertaintyVocabularyAnalysis(model_name, dataset, prompt, uncertainty_prompt)
#exp661.run()

#7.1.1 - Prompt Token Comparison
#exp711 = PromptTokenComparison(model_name, dataset, prompt, uncertainty_prompt)
#exp711.run()
