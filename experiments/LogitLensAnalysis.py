from base import *

class LogitLensAnalysis(experiments_base):
    def __init__(self):
        super().__init__()

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
    
    def run_analysis(self, manipulation_type="3"):
        """Executes the full logit lens analysis workflow for the initialized model."""
        print(f"\n{'='*60}")
        print(f"Processing model: {self.model_name}")
        print(f"{'='*60}")
        try:
            n = 0
            u_ids = self.getUncertaintyTokens()

            Investigating_datasets = self._get_investigating_datasets(manipulation_type)
            manipulated_masses = {}
            counting = []

            for name, dataset in Investigating_datasets:
                print(f"\n{'='*60}")
                print(f"Computing masses for {name}")
                print(f"{'='*60}")

                total_mass = None
                n = 0

                for i in range(1):
                    print(f"Computing masses for {name} prompt {i+1}/{len(dataset)}...") 
                    prompt = self.build_chat_prompt(dataset[i]["input_text"])

                    #new_mass = self._mass_by_layer(prompt, u_ids)
                    #new_certain_mass = self.entropy_by_layer(certain_prompt)
                    new_mass = self.change_by_layer(prompt)

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

            out_dir = self.get_out_dir() # Get output directory based on model name

            print("Plotting and saving...")
            self._plot_logit_lens(layers, manipulated_masses, manipulation_type, out_dir, "Layer", "Average (over prompts) Change by Layer", "Change per Layer")

            print("Saving CSV...")
            self._save_uncertainty_mass_csv(layers, manipulated_masses, out_dir)

            print("Saving run_info.json...")
            self._save_run_info(out_dir,manipulation_type, counting, "logit lens")

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
        out_dir = f"../results/logit_lens/{safe_model_name}"
        os.makedirs(out_dir, exist_ok=True)

        return out_dir

#2.1 - Uncertainty mass measuring
exp2 = LogitLensAnalysis()
exp2.run_analysis()