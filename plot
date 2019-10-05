    def make_cat_acts_2d_walk_fig(last_pc):
        """
        Returns fig showing evolution of average category activations in 2D space using SVD.
        """
        start = time.time()

        palette = np.array(sns.color_palette("hls", model.hub.probe_store.num_cats))
        num_saved_mb_names = len(model.ckpt_mb_names)
        num_walk_timepoints = min(num_saved_mb_names, FigsConfigs.DEFAULT_NUM_WALK_TIMEPOINTS)
        walk_mb_names = extract_n_elements(model.ckpt_mb_names, num_walk_timepoints)
        # fit pca model on last data_step
        pca_model = sklearnPCA(n_components=last_pc)
        model.acts_df = reload_acts_df(model.model_name, model.ckpt_mb_names[-1], model.hub.mode)
        cat_prototype_acts_df = model.get_multi_cat_prototype_acts_df()
        pca_model.fit(cat_prototype_acts_df.values)
        # transform acts from remaining data_steps with pca model
        cat_acts_2d_mats = []
        for mb_name in walk_mb_names:
            model.acts_df = reload_acts_df(model.model_name, mb_name, model.hub.mode)
            cat_prototype_acts_df = model.get_multi_cat_prototype_acts_df()
            cat_act_2d_pca = pca_model.transform(cat_prototype_acts_df.values)
            cat_acts_2d_mats.append(cat_act_2d_pca[:, last_pc - 2:])
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 7), dpi=FigsConfigs.DPI)
        for cat_id, cat in enumerate(model.hub.probe_store.cats):
            x, y = zip(*[acts_2d_mat[cat_id] for acts_2d_mat in cat_acts_2d_mats])
            ax.plot(x, y, c=palette[cat_id], lw=FigsConfigs.LINEWIDTH)
            xtext, ytext = cat_acts_2d_mats[-1][cat_id, :]
            txt = ax.text(xtext, ytext, str(cat), fontsize=8,
                          color=palette[cat_id])
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=FigsConfigs.LINEWIDTH, foreground="w"), PathEffects.Normal()])
        ax.axis('off')
        x_maxval = np.max(np.dstack(cat_acts_2d_mats)[:, 0, :]) * 1.2
        y_maxval = np.max(np.dstack(cat_acts_2d_mats)[:, 1, :]) * 1.2
        ax.set_xlim([-x_maxval, x_maxval])
        ax.set_ylim([-y_maxval, y_maxval])
        ax.axhline(y=0, linestyle='--', c='grey', linewidth=1.0)
        ax.axvline(x=0, linestyle='--', c='grey', linewidth=1.0)
        ax.set_title('PCA (Components {} and {}) Walk across Timepoints'.format(last_pc - 1, last_pc))
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig
