from base import * 

class LayerwiseResidualDifference(experiments_base):
    """Measures layerwise residual stream divergence between real and fake prompts."""

    def __init__(self):
        super().__init__()



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
    
    def compute_group_residual_diff_paired(self, cache_group_a, cache_group_b):
        """
        Computes mean pairwise residual-stream difference per layer.

        For every manipulated prompt in group_b, compare it only with its
        corresponding original prompt from group_a via original_idx.

        cache_group_a and cache_group_b contain tuples:
            (cache, original_idx)
        """

        diffs = []

        # creates dictionary with id and cahce following. 
        cache_a_by_idx = {
            original_idx: cache
            for cache, original_idx in cache_group_a
        }

        for layer in range(self.model.cfg.n_layers):
            pairwise_diffs = []

            for cache_b, original_idx in cache_group_b:
                if original_idx not in cache_a_by_idx:
                    raise ValueError(
                        f"original_idx {original_idx} not found in cache_group_a"
                    )

                cache_a = cache_a_by_idx[original_idx]

                resid_a = cache_a[f"blocks.{layer}.hook_resid_post"][0, -1]
                resid_b = cache_b[f"blocks.{layer}.hook_resid_post"][0, -1]

                pairwise_diff = torch.norm(resid_a - resid_b)

                pairwise_diffs.append(pairwise_diff)

            layer_diff = torch.stack(pairwise_diffs).mean().item()
            diffs.append(layer_diff)

        return diffs
    
    def get_out_dir(self, manipulation_type):
        safe_model_name = self.model_name.replace("/", "_")
        out_dir = f"../results/Res_Diff/{safe_model_name}/{manipulation_type}"
        os.makedirs(out_dir, exist_ok=True)

        return out_dir



    

    def run_over_dataset(self,  manipulation_type="3"):

        """Computes the layerwise Residual Difference between different datagroups. """
        print(f"\n{'='*60}")
        print(f"Processing model: {self.model_name}")
        print(f"{'='*60}")
        try:
            Investigating_datasets = self._get_investigating_datasets(manipulation_type)
            all_layer_diffs = {}
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
                for i in range(min(len(dataset), 183)):
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
                    current_layer_diffs = self.compute_group_residual_diff(batch_certain, batch_cache)
                else:
                    current_layer_diffs  = self.compute_group_residual_diff_paired(batch_certain, batch_cache)
                all_layer_diffs[name]= current_layer_diffs
                
            layers = list(range(self.model.cfg.n_layers))
            out_dir = self.get_out_dir(manipulation_type) # Get output directory based on model name

             
            print("Plotting and saving...")
            self.plot(layers,all_layer_diffs, manipulation_type,out_dir, "Layer", "Residual Norm Difference to certain", "Layerwise Residual Difference")

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

expResDiff = LayerwiseResidualDifference()
#expResDiff.run_over_dataset("not_enough_info")
#expResDiff.run_over_dataset("3")
expResDiff.run_over_dataset("5")
#expResDiff.run_over_dataset("two_groups")