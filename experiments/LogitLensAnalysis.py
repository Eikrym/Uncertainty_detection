from base import experiments_base

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


#2.1 - Uncertainty mass measuring
exp2 = LogitLensAnalysis()
exp2.run_analysis()