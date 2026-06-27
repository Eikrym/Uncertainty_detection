from base import *

class ChangeByLayer(experiments_base):
    def __init__(self):
        super().__init__()


    def change_by_layer(self, prompt):
        _, cache = self.model.run_with_cache(
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

            Investigating_datasets = self._get_investigating_datasets(manipulation_type)
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
                    prompt = self.build_chat_prompt(dataset[i]["input_text"])
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

            out_dir = self.get_out_dir(manipulation_type) # Get output directory based on model name

            print("Plotting and saving...")
            self.plot(layers, manipulated_masses, manipulation_type, out_dir, "Layer", "Average (over prompts) Change by Layer", "Change per Layer")

            print("Saving CSV...")
            self._save_csv(layers, manipulated_masses, out_dir)

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

    def get_out_dir(self, manipulation_type):
        safe_model_name = self.model_name.replace("/", "_")
        out_dir = f"../results/change_by_layer/{safe_model_name}/{manipulation_type}"
        os.makedirs(out_dir, exist_ok=True)

        return out_dir

#2.1 - Uncertainty mass measuring
exp2 = ChangeByLayer()
#exp2.run_analysis("3")
exp2.run_analysis("5")