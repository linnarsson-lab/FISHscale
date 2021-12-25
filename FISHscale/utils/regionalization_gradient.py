import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view


class Regionalization_Gradient:
    
    def regionalization_gradient_make(self, df_hex, labels, colors=None, cm=plt.cm.nipy_spectral_r,
                                      max_depth=14, random_state=0, plot=True, n_jobs=-1, **kwargs):
        """Convert discreet regionalization to mixtures of labels.
        
        Aim for around 90% accuracy by tuning the max_depth, but be carefull
        with over fitting.
        
        Args:
            df_hex (pd.DataFrame): Pandas Dataframe with hexagonally binned 
                data. 
            labels ([list, np.ndarray]): Array with region labels for the 
                hexagonal bins in df_hex. For plotting labels should be between
                zero and one.
            colors (list, optional): List of RGB(A) color tuples for each 
                hexagonal bin. If not given will use the default or given
                matplotlib colormap. Defaults to None.
            cm (matplotlib colormap, optional): If "colors" is not given will
                use this colormap to give colors to the labels. Can be any 
                matplotlib colormap.
                Defaults to plt.cm.nipy_spectral_r.
            max_depth (int, optional): Maximum depth of the Random Forest 
                Classifier. Tweak this to prevent under or over fitting. See
                Scikit Learn documentation for more details. Defaults to 14.
            random_state (int, optional): Random state for the Random Forest
                Classifier. Defaults to 0.
            plot (bool, optional): If True, will plot the original lables,
                classifier calls and mixed colors. Defaults to True.
            n_jobs (int, optional): Number of jobs to run. Defaults to -1.
            **kwargs (optional): Kwargs will be passed to the initiation of the
                RandomForestClassifier. Refer to the Scikit Learn documentation
                for options and explanation.
        Returns:
            [list, np.ndarray, np.ndarray, dict]: mixed_colors, predicted_prob, 
                predicted_labels, color_dict
            mixed_colors: List with mixed RGB colors.
            predicted_prob: Array with probabilities for all labels for each
                hexagonal tiles.
            predicted_labels: Array with predicted labels of classification.
            color_dict: Dictionary with label colors.
        """
        
        if n_jobs == -1:
            n_jobs = self.cpu_count
            
        #Initiate classifier 
        clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state, n_jobs=n_jobs, **kwargs)
        
        #Make dictionary of original colors
        unique_labels = np.unique(labels)
        color_dict = {}
        if type(colors) == type(None):
            for i, l in enumerate(unique_labels):
                color_dict[l] = cm(i/(unique_labels.shape[0] - 1))
        else:
            for l in unique_labels:
                index = np.where(labels==l)[0][0]
                color_dict[l] = colors[index]
        
        #Fit       
        clf.fit(df_hex.T, labels)
        
        #Predict labels
        predicted_labels = clf.predict(df_hex.T)
        total = labels.shape[0]
        matches = (labels == predicted_labels).sum()
        percentage = round((matches/total)*100)
        print(f'Perfect matches: {matches} out of {total} which is {percentage}% accuracy')
        if percentage > 98:
            print('Warning, percentage of identical matches is high which might indicate over fitting of the classifier.')
            print('Consider initiating the classifier again with more stringent settings.')
    
        #Get probability of class
        predicted_prob = clf.predict_proba(df_hex.T)
        #Get colors in same order as clasifier class
        region_colors = [color_dict[i] for i in clf.classes_]
        
        #Mix colors based on class probablitiy
        mix_color = []
        for p in predicted_prob:
            weighted_color = p[:,np.newaxis] * region_colors
            new_color = weighted_color.sum(axis=0)
            mix_color.append(new_color[:-1])
            
            
        if plot:
            
            fig, axes = plt.subplots(figsize=(20,20), ncols=3)
            
            #Plot original labels
            ax0 = axes[0]
            if type(colors) == type(None):
                self.hexbin_plot(labels, cm=cm, ax=ax0)
            else:
                self.hexbin_plot(colors, ax=ax0)
            ax0.set_title('Original Labels', fontsize=14)
            ax0.set_axis_off()
            
            #Plot predicted labels
            ax1=axes[1]
            if type(colors) == type(None):
                self.hexbin_plot(predicted_labels, cm=cm, ax=ax1)
            else:
                #Reuse same colors as original
                colors_new = [color_dict[i] for i in predicted_labels]
                self.hexbin_plot(colors_new, ax=ax1)
            ax1.set_title(f'Predicted Labels, {percentage}% accuracy', fontsize=14)
            ax1.set_axis_off()
            
            ax2 = axes[2]
            self.hexbin_plot(mix_color, ax=ax2)
            ax2.set_title('Mixed Labels', fontsize=14)
            ax2.set_axis_off()
            
        return mix_color, predicted_prob, predicted_labels, color_dict
    
    
class Regionalization_Gradient_Multi:
    
    def regionalization_gradient_make_multi(self, regionalization_result, colors=None, cm=plt.cm.nipy_spectral_r,
                                      max_depth=15, random_state=0, n_jobs=-1, **kwargs):
        """Convert discreet regionalization to mixtures of labels.
        
        Runs self.regionalization_gradient_make() for each individual dataset.
        Results are stored in a dictionary per dataset and can be accessed
        easily using self.get_dict_item().
        
        Aim for around 90% accuracy by tuning the max_depth, but be carefull
        with over fitting.
        
        Args:
            regionalization_result (dictionary): Results form the 
                self.regionalize() function. 
            colors (list, optional): List of RGB(A) color tuples for each 
                hexagonal bin. If not given will use the default or given
                matplotlib colormap. Defaults to None.
            cm (matplotlib colormap, optional): If "colors" is not given will
                use this colormap to give colors to the labels. Can be any 
                matplotlib colormap.
                Defaults to plt.cm.nipy_spectral_r.
            max_depth (int, optional): Maximum depth of the Random Forest 
                Classifier. Tweak this to prevent under or over fitting. See
                Scikit Learn documentation for more details. Defaults to 15.
            random_state (int, optional): Random state for the Random Forest
                Classifier. Defaults to 0.
            n_jobs (int, optional): Number of jobs to run. Defaults to -1.
            **kwargs (optional): Kwargs will be passed to the initiation of the
                RandomForestClassifier. Refer to the Scikit Learn documentation
                for options and explanation.
        Returns:
            Dictionary containing:
                - mixed_colors: List with mixed RGB colors.
                - predicted_prob: Array with probabilities for all labels for each
                hexagonal tiles.
                - predicted_labels: Array with predicted labels of classification.
                - color_dict: Dictionary with label colors.
            
        """
        #Instantiate class with function
        RG = Regionalization_Gradient()
        
        #Merge data
        results_df = self.get_dict_item(regionalization_result, 'df_hex')
        merged_data, samples = self.merge_norm(results_df, mode=None)
        merged_labels = np.concatenate(self.get_dict_item(regionalization_result, 'labels_merged'))

        #Mix colors
        if n_jobs == -1:
            n_jobs = self.cpu_count
        mixed_colors = RG.regionalization_gradient_make(merged_data, merged_labels, colors=colors, cm=cm,
                                                        max_depth=max_depth, random_state=random_state, plot=False,
                                                        n_jobs=n_jobs, **kwargs)

        #Get sample indexes
        shapes = [i.shape[1] for i in results_df]
        sample_breaks = np.cumsum(shapes)
        sample_breaks = np.insert(sample_breaks, 0, 0)
        
        #Put results in dictionary
        results = {}

        for name, i in zip(self.datasets_names, sliding_window_view(sample_breaks, window_shape=2)):
            results[name] = {}
            
            for rn, r in zip(['mixed_colors', 'predicted_prob', 'predicted_labels'], mixed_colors[:3]):
                results[name][rn] = r[i[0]:i[1]]
                
            results[name]['color_dict'] = mixed_colors[3]
        
        #Get input
        #dfs =  self.get_dict_item(regionalization_result, 'df_hex')
        #labels = self.get_dict_item(regionalization_result, 'labels')
        
        #Use pre-determined colors to get the mixing right
        #if type(colors) == type(None):
        #    colors = [cm(l) for l in labels]
        
        #results = {}
        #Loop over datasets and run the function
        #for d, df, l, c in tqdm(zip(self.datasets, dfs, labels, colors)):
        #    results[d.dataset_name] = {}
        #    mixed_colors, predicted_prob, predicted_labels, color_dict = d.regionalization_gradient_make(df, 
        #                                                                                                 l, 
        #                                                                                                 colors=c, 
        #                                                                                                 cm=cm,
        #                                                                                                 max_depth=max_depth,
        #                                                                                                 random_state=random_state,
        #                                                                                                 plot=plot,
        #                                                                                                 n_jobs=n_jobs,
        #                                                                                                 **kwargs)
        #    results[d.dataset_name]['mixed_colors'] = mixed_colors
        #    results[d.dataset_name]['predicted_prob'] = predicted_prob
        #    results[d.dataset_name]['predicted_labels'] = predicted_labels
        #    results[d.dataset_name]['color_dict'] = color_dict
        
        return results
        
        
        
        
        
        
        