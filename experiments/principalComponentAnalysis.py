from base import *
class PrincipalComponentAnalysis(experiments_base):
    def __init__(self):
        super().__init__()

    def _get_last_token_residual(self, prompt, layer=-1):
        """
        Extract residual stream activation at the last token.
        layer=-1 means last transformer layer.
        """
        _, cache = self.model.run_with_cache(
            prompt,
            names_filter=lambda name: "hook_resid_post" in name
        )

        if layer == -1:
            layer = self.model.cfg.n_layers - 1

        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :] #batch: 0, Token-position: last, d-model (full residual stream)

        return resid.detach().cpu().float().numpy()
    
    def get_out_dir(self, manipulation_type):
        safe_model_name = self.model_name.replace("/", "_")
        out_dir = f"../results/PCA/{safe_model_name}/{manipulation_type}"
        os.makedirs(out_dir, exist_ok=True)

        return out_dir
    
    def plot_PCA(self,layer,names, out_dir, labels,X_pca,explained_variance, manipulation_type): 
        # Plot
        plt.figure(figsize=(9, 7))
        masks = {}
        labels_array = np.array(labels)
        cmap = plt.cm.get_cmap("tab10", len(names))
        i = 0
        for name in names: 
            masks[name] = labels_array == name

            plt.scatter(
            X_pca[masks[name], 0],
            X_pca[masks[name], 1],
            label=name,
            alpha=0.7,
            color=cmap(i)
            )
            i += 1

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
        out_path = f"{out_dir}/PCA.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
        plt.close()

    def _save_pca_csv(self, X_pca, labels, out_dir, layer):
        out_path = f"{out_dir}/pca_layer_{layer}.csv"

        with open(out_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Example", "Label", "PC1", "PC2"])

            for i, (coords, label) in enumerate(zip(X_pca, labels)):
                writer.writerow([i, label, coords[0], coords[1]])

        print(f"Saved PCA CSV to {out_path}")

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
    
    def run_analysis(self, manipulation_type="3", layer = 15):
        """Executes the full logit lens analysis workflow for the initialized model."""
        print(f"\n{'='*60}")
        print(f"Processing model: {self.model_name}")
        print(f"{'='*60}")
        try:
            n = 0
            u_ids = self.getUncertaintyTokens()

            Investigating_datasets = self._get_investigating_datasets(manipulation_type)
            activations = []
            labels = []
            counting = []
            names = []

            for name, dataset in Investigating_datasets:
                print(f"\n{'='*60}")
                print(f"Computing PCA for {name}")
                print(f"{'='*60}")

                total_mass = None
                n = 0

                for i in range(len(dataset)):
                    print(f"Computing residuals for {name} prompt {i+1}/{len(dataset)}...") 
                    prompt = self.build_chat_prompt(dataset[i]["input_text"])

                    resid = self._get_last_token_residual(prompt, layer=layer)
                    activations.append(resid)
                    labels.append(name)
                    n += 1
                counting.append((name, n))
                # Convert to matrix: [num_examples, d_model]
                X = np.stack(activations)

                # PCA to 2 dimensions
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)

                explained_variance = pca.explained_variance_ratio_

                names.append(name)

            out_dir = self.get_out_dir(manipulation_type) # Get output directory based on model name

            print("Plotting and saving...")
            self.plot_PCA(layer, names, out_dir, labels, X_pca, explained_variance, manipulation_type)

            print("Saving CSV...")
            self._save_pca_csv(X_pca, labels, out_dir, layer)

            print("Saving run_info.json...")
            self._save_run_info(out_dir,manipulation_type, counting, "PCA")

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

#5.4.1 - PCA
exp541 = PrincipalComponentAnalysis()
exp541.run_analysis("3",-1)