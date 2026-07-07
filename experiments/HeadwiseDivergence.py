from base import *

class HeadwiseDivergence(experiments_base):
    """Analyzes which attention heads contribute most to divergence between certain and uncertain inputs."""

    def __init__(self, Model_name=None):
        if(Model_name is None):
            super().__init__()
        else:
            super().__init__(model_name=Model_name)

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
    
    def _compute_headwise_divergence_last_Token_paired(self, cache_group_a, cache_group_b):
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        head_diffs = torch.zeros(n_layers, n_heads)

        cache_a_by_idx = {
            original_idx: cache
            for cache, original_idx in cache_group_a
        }

        for layer in range(n_layers):
            diff_sum = None

            for cache_b, id in cache_group_b:
                if id not in cache_a_by_idx:
                    raise ValueError(f"Cache with id {id} was not found in certain cache")

                cache_a = cache_a_by_idx[id]

                a_heads = cache_a[f"blocks.{layer}.attn.hook_z"][0][-1]
                b_heads = cache_b[f"blocks.{layer}.attn.hook_z"][0][-1]

                diff = torch.norm(a_heads - b_heads, dim=-1)

                if diff_sum is None:
                    diff_sum = diff.clone()
                else:
                    diff_sum += diff

            diff_avg = diff_sum / len(cache_group_b)
            head_diffs[layer, :] = diff_avg

        return head_diffs

    def get_out_dir(self, manipulation_type, name):
        safe_model_name = self.model_name.replace("/", "_")
        out_dir = f"../results/Head_Div/{safe_model_name}/{manipulation_type}/{name}"
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def plot_head_div(self, head_diffs, out_dir, xLabel, yLabel, title):
        plt.figure(figsize=(14, 8))
        im = plt.imshow(head_diffs.numpy(), aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(im, label="Head Divergence")  
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(
            f"{title}\n{self.model_name}",
        )
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()
    
        out_path = f"{out_dir}/head_div.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
        plt.close()

    def _save_csv_head_div(self, head_diffs, out_dir, top_k=10):
        out_path = f"{out_dir}/head_div.csv"

        # Falls Tensor auf GPU liegt oder Gradienten hat
        head_diffs_np = head_diffs.detach().cpu()

        n_layers, n_heads = head_diffs_np.shape

        # Top heads berechnen
        flat_heads = sorted(
            [((l, h), head_diffs_np[l, h].item()) for l in range(n_layers)
            for h in range(n_heads)],
            key=lambda x: x[1],
            reverse=True
        )

        with open(out_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)

            # Haupttabelle: Layer x Heads
            header = ["Layer"] + [f"Head_{head}" for head in range(n_heads)]
            writer.writerow(header)

            for layer in range(n_layers):
                row = [layer] + [head_diffs_np[layer, head].item() for head in range(n_heads)]
                writer.writerow(row)

            # Leerzeile
            writer.writerow([])

            # Top-k Heads am Ende
            writer.writerow([f"Top {top_k} Most Divergent Heads"])
            writer.writerow(["Rank", "Layer", "Head", "Divergence"])

            for i, ((layer, head), div) in enumerate(flat_heads[:top_k]):
                writer.writerow([i + 1, layer, head, div])

        print(f"Saved CSV to {out_path}")

    def _save_run_info_head_div(self, out_dir, exact_manipulation, datapoints, experiment, model = None, dataset = "TrialGPT-Criterion-Annotations"):
        if model == None: 
            model = self.model_name
        run_info = { "experiment": experiment, "model": model, "exact_manipulation": exact_manipulation, "dataset": dataset, "git_commit": self._get_git_commit(), "datapoints": datapoints}
        out_path = f"{out_dir}/run_info.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)

        print(f"Saved run info to {out_path}")
    
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
    
    def run_analysis(self,  manipulation_type="3", max_examples = None):

        """Execute the experiment."""
        print(f"\n{'='*60}")
        print(f"Experiment: Headwise Divergence")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}")
        try:
            Investigating_datasets = self._get_investigating_datasets(manipulation_type)
            all_layer_diffs = {}
            batch_certain = None
            for name, dataset in Investigating_datasets:
                """Execute the experiment."""
                print(f"\n{'='*60}")
                print(f"{name}. Experiment: Headwise Divergence")
                print(f"Model: {self.model_name}")
                print(f"{'='*60}")
                batch_cache = []
                n=0
                if name == "certain":
                    limit = len(dataset)
                else:
                    limit = len(dataset) if max_examples is None else min(len(dataset), max_examples)
                for i in range(limit):
                    print(f"Computing divergence for {name} prompt {i+1}/{len(dataset)}...") 
                    prompt = self.build_chat_prompt(dataset[i]["input_text"])
                    
                    with torch.no_grad():
                        _, cache = self.model.run_with_cache(prompt, names_filter=lambda name: "attn.hook_z" in name, return_type=None)
                    batch_cache.append((cache, dataset[i]['annotation_id']))
                    n +=1
                if batch_certain is None:
                    batch_certain = list(batch_cache)
                print("Computing headwise divergence...")
                if(manipulation_type == "not_enough_info"):
                    batch_cache = [cache for cache,_ in batch_cache]
                    if isinstance(batch_certain[0], tuple):
                        batch_certain = [cache for cache,_ in batch_certain]
                    current_layer_diffs = self._compute_headwise_divergence_last_Token(batch_certain, batch_cache)
                else:
                    current_layer_diffs  = self._compute_headwise_divergence_last_Token_paired(batch_certain, batch_cache)
                out_dir = self.get_out_dir(manipulation_type, name)
                print("Plotting and saving...")
                self.plot_head_div(current_layer_diffs, out_dir, "Head Index", "Layer", f"Headwise Divergence Heatmap: certain vs {name}")

                print("Saving CSV...")
                self._save_csv_head_div(current_layer_diffs, out_dir)

                print("Saving run_info.json...")
                self._save_run_info_head_div(out_dir, name, n, "Headwise Divergence")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            return current_layer_diffs


        except Exception as e:
            print(f"Error processing {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Free memory per model
            #if self.model is not None:
            #    del self.model
            #    self.model = None # Clear reference
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
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

if __name__ == "__main__":
    expHeadDiv = HeadwiseDivergence()
    expHeadDiv.run_analysis("two_groups", max_examples=160)