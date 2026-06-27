from base import *

class LayerwiseDifference(experiments_base):
    """Analyzes the difference in each layer compared to the last layer"""

    def __init__(self):
        super().__init__()


    
    def run_analysis(self, manipulation_type="3"):
        """Executes the full logit lens analysis workflow for the initialized model."""
        print(f"\n{'='*60}")
        print(f"Processing model: {self.model_name}")
        print(f"{'='*60}")
        try:
            n = 0
            Investigating_datasets = self._get_investigating_datasets(manipulation_type)
            div_to_last_layer = {}
            counting = []

            for name, dataset in Investigating_datasets:
                print(f"\n{'='*60}")
                print(f"Computing difference in activation to final layer for {name}")
                print(f"{'='*60}")

                batch_cache = []
                n = 0

                for i in range(len(dataset)):
                    print(f"Computing masses for {name} prompt {i+1}/{len(dataset)}...") 
                    prompt = self.build_chat_prompt(dataset[i]["input_text"])
                    with torch.no_grad():
                        _, cache = self.model.run_with_cache(prompt, names_filter=lambda name: "attn.hook_z" in name, return_type=None)
                    batch_cache.append(cache)
                    n += 1

                current_div_to_last_layer  = self._compute_layerwise_divergence_to_last_layer_direct(batch_cache)
                div_to_last_layer[name]= current_div_to_last_layer
                counting.append((name, n))

            layers = list(range(self.model.cfg.n_layers))
            out_dir = self.get_out_dir() # Get output directory based on model name

            print("Plotting and saving...")
            self.plot(layers, div_to_last_layer, manipulation_type, out_dir, "Layer", "Mean L2 distance to final-layer head activations", "Layerwise distance to final representation")

            print("Saving CSV...")
            self._save_csv(layers, div_to_last_layer, out_dir)

            print("Saving run_info.json...")
            self._save_run_info(out_dir,manipulation_type, counting, "layerwise divergence")

            print(counting)

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
        
    def get_out_dir(self):
        safe_model_name = self.model_name.replace("/", "_")
        out_dir = f"../results/layer_Wise_divergence_last_layer/{safe_model_name}" 
        os.makedirs(out_dir, exist_ok=True)

        return out_dir



    def _compute_layerwise_divergence_to_last_layer(self, cache_group_a):
        """Compute divergence per attention head."""
        n_layers = self.model.cfg.n_layers
        layer_activations=[]

        for layer in range(n_layers):

            activation_sum_a = None
            #pro layer, pro prompt-gruppe hat cache_group_a eine Liste von caches, da wir pro prompt einen cache haben. 
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
    
    def _compute_layerwise_divergence_to_last_layer_direct(self, cache_group_a):
        """Compute average per-head divergence to last layer across prompts."""
        n_layers = self.model.cfg.n_layers
        layer_diffs = torch.zeros(n_layers)

        for layer in range(n_layers):
            diffs = []

            for cache in cache_group_a:
                heads_layer = cache[f"blocks.{layer}.attn.hook_z"][0, -1]
                heads_last = cache[f"blocks.{n_layers-1}.attn.hook_z"][0, -1]

                # [n_heads]
                diff_per_head = torch.norm(heads_layer - heads_last, dim=-1)

                # scalar for this prompt/layer
                diffs.append(diff_per_head.mean())

            layer_diffs[layer] = torch.stack(diffs).mean()

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

#3.2.3 - difference to last layer 
exp323 = LayerwiseDifference()
exp323.run_analysis()