from base import * 

class LayerwiseResidualDifference(experiments_base):
    """Measures layerwise residual stream divergence between real and fake prompts."""

    def __init__(self, Model_name=None):
        if(Model_name is None):
            super().__init__()
        else:
            super().__init__(model_name=Model_name)



    def compute_group_residual_diff(self, cache_group_a, cache_group_b, n_boot=2000):
        """
        Computes layerwise residual-stream distance between two unpaired groups.

        Main value:
            || mean(group_a) - mean(group_b) ||

        Bootstrap CI:
            Resample group_a and group_b independently with replacement,
            recompute the group-mean distance, and take 2.5/97.5 percentiles.
        """

        n_layers = self.model.cfg.n_layers

        # --------------------------------------------------
        # 1. Extract residuals into arrays:
        #    group_a_resids shape: [n_a, n_layers, d_model]
        #    group_b_resids shape: [n_b, n_layers, d_model]
        # --------------------------------------------------
        group_a_resids = []
        group_b_resids = []

        for cache in cache_group_a:
            prompt_resids = []
            for layer in range(n_layers):
                resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]
                prompt_resids.append(resid.detach().cpu().float().numpy())
            group_a_resids.append(prompt_resids)

        for cache in cache_group_b:
            prompt_resids = []
            for layer in range(n_layers):
                resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]
                prompt_resids.append(resid.detach().cpu().float().numpy())
            group_b_resids.append(prompt_resids)

        group_a_resids = np.array(group_a_resids, dtype=np.float32)
        group_b_resids = np.array(group_b_resids, dtype=np.float32)

        n_a = group_a_resids.shape[0] #Anzahl der Prompts in Gruppe A
        n_b = group_b_resids.shape[0] #Anzahl der Prompts in Gruppe B

        # --------------------------------------------------
        # 2. Compute original group distance per layer
        # --------------------------------------------------
        mean_a = group_a_resids.mean(axis=0)  # [n_layers, d_model], da über alle Prompts gemittelt
        mean_b = group_b_resids.mean(axis=0)  # [n_layers, d_model]

        diffs = np.linalg.norm(mean_a - mean_b, axis=1).tolist()

        # --------------------------------------------------
        # 3. Bootstrap confidence interval
        # --------------------------------------------------
        boot_diffs = np.empty((n_boot, n_layers), dtype=np.float32)

        for b in range(n_boot):
            sample_a_idx = np.random.choice(n_a, size=n_a, replace=True)
            sample_b_idx = np.random.choice(n_b, size=n_b, replace=True)

            sample_a = group_a_resids[sample_a_idx]
            sample_b = group_b_resids[sample_b_idx]

            sample_mean_a = sample_a.mean(axis=0)
            sample_mean_b = sample_b.mean(axis=0)

            boot_diffs[b, :] = np.linalg.norm(
                sample_mean_a - sample_mean_b,
                axis=1
            )

        lower = np.percentile(boot_diffs, 2.5, axis=0)
        upper = np.percentile(boot_diffs, 97.5, axis=0)

        return diffs, lower.tolist(), upper.tolist()
    
    def compute_group_residual_diff_paired(self, cache_group_a, cache_group_b):
        """
        Computes mean pairwise residual-stream difference per layer.

        Returns:
            prompt_layer_diffs:
                list with shape [n_prompts, n_layers]
                Each row is one prompt pair.
            
            diffs:
                list with shape [n_layers]
                Mean residual difference per layer.
        """

        cache_a_by_idx = {
            original_idx: cache
            for cache, original_idx in cache_group_a
        }

        prompt_layer_diffs = []

        for cache_b, original_idx in cache_group_b:
            if original_idx not in cache_a_by_idx:
                raise ValueError(
                    f"original_idx {original_idx} not found in cache_group_a"
                )

            cache_a = cache_a_by_idx[original_idx]

            diffs_for_this_prompt = []

            for layer in range(self.model.cfg.n_layers):
                resid_a = cache_a[f"blocks.{layer}.hook_resid_post"][0, -1]
                resid_b = cache_b[f"blocks.{layer}.hook_resid_post"][0, -1]

                pairwise_diff = torch.norm(resid_a - resid_b).item()

                diffs_for_this_prompt.append(pairwise_diff)

            prompt_layer_diffs.append(diffs_for_this_prompt)

        prompt_layer_diffs_np = np.array(prompt_layer_diffs, dtype=np.float32)

        diffs = prompt_layer_diffs_np.mean(axis=0).tolist()

        return prompt_layer_diffs, diffs
        
    def get_out_dir(self, manipulation_type):
        safe_model_name = self.model_name.replace("/", "_")
        out_dir = f"../results/Res_Diff/{safe_model_name}/{manipulation_type}"
        os.makedirs(out_dir, exist_ok=True)

        return out_dir



    

    def run_analysis(self,  manipulation_type="3", max_examples = None):

        """Computes the layerwise Residual Difference between different datagroups. """
        print(f"\n{'='*60}")
        print(f"Processing model: {self.model_name}")
        print(f"{'='*60}")
        try:
            Investigating_datasets = self._get_investigating_datasets(manipulation_type)
            all_layer_diffs = {}
            all_prompt_layer_diffs = {}
            ci_by_name = {}
            batch_certain = None
            counting = []
            for name, dataset in Investigating_datasets:
                """Execute the experiment."""
                print(f"\n{'='*60}")
                print(f"{name}. Experiment: Layerwise Residual Difference")
                print(f"Model: {self.model_name}")
                print(f"{'='*60}")
                batch_cache = []
                n=0
                limit = len(dataset) if max_examples is None else min(len(dataset), max_examples)
                for i in range(limit):
                    print(f"Computing masses for {name} prompt {i+1}/{len(dataset)}...") 
                    prompt = self.build_chat_prompt(dataset[i]["input_text"])
                    
                    with torch.no_grad():
                        _, cache = self.model.run_with_cache(prompt, names_filter=lambda name: "hook_resid_post" in name, return_type=None)
                    batch_cache.append((cache, dataset[i]['annotation_id']))
                    n +=1
                if batch_certain is None:
                    batch_certain = list(batch_cache)
                counting.append((name, n))
                print("Computing layerwise residual differences...")
                if(manipulation_type == "not_enough_info"):
                    batch_cache = [cache for cache,_ in batch_cache]
                    if isinstance(batch_certain[0], tuple):
                        batch_certain = [cache for cache,_ in batch_certain]
                    current_layer_diffs, lower, upper = self.compute_group_residual_diff(batch_certain, batch_cache, n_boot=2000)
                    prompt_layer_diffs = None
                    ci_by_name[name] = (lower, upper)
                else:
                    prompt_layer_diffs, current_layer_diffs  = self.compute_group_residual_diff_paired(batch_certain, batch_cache)
                    ci_by_name[name] = None
                all_layer_diffs[name]= current_layer_diffs
                all_prompt_layer_diffs[name] = prompt_layer_diffs
                
            layers = list(range(self.model.cfg.n_layers))
            out_dir = self.get_out_dir(manipulation_type) # Get output directory based on model name

             
            print("Plotting and saving...")

            self.plot(layers, all_layer_diffs, manipulation_type,out_dir, "Layer", "Residual Norm Difference to certain", "Layerwise Residual Difference", all_prompt_layer_diffs, 2000, ci_by_name=ci_by_name)

            print("Saving CSV...")
            self._save_csv(layers, all_layer_diffs, out_dir)

            print("Saving run_info.json...")
            self._save_run_info(out_dir,manipulation_type, counting, "logit lens")
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

if __name__ == "__main__":
    expResDiff = LayerwiseResidualDifference()
    expResDiff.run_analysis("not_enough_info")
    #expResDiff.run_over_dataset("3")
    #expResDiff.run_over_dataset("5")
    #expResDiff.run_over_dataset("two_groups")