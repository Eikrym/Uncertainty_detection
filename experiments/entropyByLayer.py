from base import *

class entropyByLayer(experiments_base):
    def __init__(self, Model_name=None):
        if(Model_name is None):
            super().__init__()
        else: 
            super().__init__(model_name=Model_name)




    def entropy_by_layer(self, prompt):
        """Compute entropy for each layer"""
        # Cache only resid_post to reduce memory
        print(f"entering _entropy_by_layer for prompt: '{prompt[:50]}...' ")
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                prompt,
                names_filter=lambda n: ("hook_resid_post" in n),
            )
        masses = []
        for layer in range(self.model.cfg.n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1]  # [d_model]
            # Project through ln_final + unembed to get pseudo-logits
            pseudo_logits = resid @ self.model.W_U
            pseudo_logits = pseudo_logits + self.model.b_U

            log_probs = torch.log_softmax(pseudo_logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum()
            masses.append(entropy.item())

        print(f"exiting entropy_by_layer")

        return masses

    
    
    def run_analysis(self, manipulation_type="3", max_examples = None):
        """Executes the full logit lens analysis workflow for the initialized model."""
        print(f"\n{'='*60}")
        print(f"Processing model: {self.model_name}")
        print(f"{'='*60}")
        try:
            n = 0

            Investigating_datasets = self._get_investigating_datasets(manipulation_type)
            manipulated_masses = {}
            prompt_entropies_by_name = {}
            counting = []

            for name, dataset in Investigating_datasets:
                print(f"\n{'='*60}")
                print(f"Computing masses for {name}")
                print(f"{'='*60}")

                total_mass = None
                n = 0
                limit = len(dataset) if max_examples is None else min(len(dataset), max_examples)
                prompt_entropies = []
                for i in range(limit):
                    print(f"Computing masses for {name} prompt {i+1}/{len(dataset)}...") 
                    prompt = self.build_chat_prompt(dataset[i]["input_text"])

                    new_mass = self.entropy_by_layer(prompt)
                    prompt_entropies.append(new_mass)

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
                prompt_entropies_by_name[name] = prompt_entropies
            layers = list(range(self.model.cfg.n_layers))

            out_dir = self.get_out_dir(manipulation_type) # Get output directory based on model name

            print("Plotting and saving...")
            self.plot(layers, manipulated_masses, manipulation_type, out_dir, "Layer", "Average (over prompts) entropy by Layer", "entropy per Layer", prompt_entropies_by_name, 2000)

            print("Saving CSV...")
            self._save_csv(layers, manipulated_masses, out_dir)

            print("Saving run_info.json...")
            self._save_run_info(out_dir,manipulation_type, counting, "entropy by Layer")

            print(counting)

        except Exception as e:
            print(f"Error processing {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def get_out_dir(self, manipulation_type):
        safe_model_name = self.model_name.replace("/", "_")
        out_dir = f"../results/entropy/{safe_model_name}/{manipulation_type}"
        os.makedirs(out_dir, exist_ok=True)

        return out_dir

#2.1 - entropy by Layer
if __name__ == "__main__":
    exp2 = entropyByLayer()
    exp2.run_analysis("two_groups")