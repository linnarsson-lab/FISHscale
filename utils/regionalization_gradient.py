import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class Regionalization_Gradient:
    
    def regionalization_gradient_make(self, df_hex, labels, colors=None, cm=plt.cm.nipy_spectral_r,
                                      max_depth=5, random_state=0, plot=True, n_jobs=-1, **kwargs):
        """Convert discreet regionalization to mixtures of labels.
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
                Scikit Learn documentation for more details. Defaults to 5.
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