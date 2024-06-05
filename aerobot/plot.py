import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
import pandas as pd
from matplotlib.colors import ListedColormap
from aerobot.io import FEATURE_TYPES, FEATURE_SUBTYPES, RESULTS_PATH, load_results_dict
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Dict, NoReturn, List
import os
import scipy.optimize
import colour

# https://journals.asm.org/writing-your-paper#figures-tables



def plot_color_palette():
    '''Defines a nice color palette. Taken from viz.py.'''
    palette = dict()
    palette['green'] = '#7AA974'
    palette['palegreen'] = '#DCECCB'
    palette['lightgreen'] = '#BFD598'
    palette['darkgreen'] = '#7E9D90'
    palette['yellow'] = '#EAC264'
    palette['lightyellow'] = '#F3DAA9'
    palette['paleyellow'] = '#FFEDCE'
    palette['blue'] = '#738FC1'
    palette['lightblue'] = '#A9BFE3'
    palette['darkblue'] = '#535D87'
    palette['paleblue'] = '#C9D7EE'
    palette['red'] = '#D56C55'
    palette['lightred'] = '#E8B19D'
    palette['palered'] = '#F1D4C9'
    palette['purple'] = '#AB85AC' 
    palette['lightpurple'] = '#D4C2D9'
    palette['darkbrown'] = '#905426'
    return palette


def plot_color_map():
    '''Creates a custom color map to match the other color scheme.'''
    start_color = colour.Color(plot_color_palette()['paleblue'])
    end_color = colour.Color(plot_color_palette()['darkblue'])
    gradient = list(start_color.range_to(end_color, 20))
    gradient = [c.hex for c in gradient]
    return ListedColormap(gradient)


def plot_configure_mpl(title_font_size:int=6.5, label_font_size:float=6.5,
                       legend_font_size:int=6, figure_width:float=6.875,
                       figure_height:float=2.5, line_width:float=0.75):
    '''Configure the matplotlib RC file, many of these settings are taken from the viz.py file.'''
    # Some specs to make plots look nice. 
    plt.rc('font', **{'family':'sans-serif', 'sans-serif':['Arial'], 'size':label_font_size})
    plt.rc('legend', **{'fontsize':legend_font_size})
    plt.rc('xtick', **{'labelsize':label_font_size, 'color':'k'})
    plt.rc('ytick', **{'labelsize':label_font_size, 'color':'k'})
    plt.rc('axes',  **{'titlesize':title_font_size, 'labelsize':label_font_size})

    plt.rcParams['mathtext.fontset'] = 'stixsans'
    plt.rcParams['mathtext.sf'] = 'sans'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (figure_width, figure_height)

    mpl.rcParams['errorbar.capsize'] = 3
    mpl.rcParams['lines.linewidth'] = line_width

    palette = plot_color_palette() # Get the color palette and set a default cycler. 
    palette_order = ['purple', 'green', 'blue', 'red', 'yellow']
    color_cycle = [palette[c] for c in palette_order] + [palette[c] for c in palette.keys() if c not in palette_order]
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', color_cycle)

    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')




# Pretty names for each feature type, for plotting. 
PRETTY_NAMES = {'KO':'Gene families', 'embedding.geneset.oxygen':'Oxygen gene set', 'chemical':'Chemical features'}
PRETTY_NAMES['aa_1mer'] = 'Amino acid counts'
PRETTY_NAMES['aa_2mer'] = 'Amino acid dimers'
PRETTY_NAMES['aa_3mer'] = 'Amino acid trimers'
PRETTY_NAMES['nt_1mer'] = 'Nucleotide counts'
PRETTY_NAMES['nt_2mer'] = 'Nucleotide dimers'
PRETTY_NAMES['nt_3mer'] = 'Nucleotide trimers'
PRETTY_NAMES['cds_1mer'] = 'CDS nucleotide counts'
PRETTY_NAMES['cds_2mer'] = 'CDS nucleotide dimers'
PRETTY_NAMES['cds_3mer'] = 'CDS nucleotide trimers'
PRETTY_NAMES.update({'embedding.genome':'Genome embedding'})
PRETTY_NAMES.update({'metadata':'Metadata', 'metadata.oxygen_genes':'Oxygen gene set', 'metadata.pct_oxygen_genes':'Percentage oxygen genes', 'metadata.number_of_genes':'Number of genes'})
PRETTY_NAMES.update({f'nt_{i}mer':f'Nucleotide {i}-mer' for i in range(4, 6)})
PRETTY_NAMES.update({f'cds_{i}mer':f'CDS nucleotide {i}-mer' for i in range(4, 6)})


ANNOTATION_BASED_FEATURE_TYPES = ['metadata.oxygen_genes', 'metadata.pct_oxygen_genes', 'KO', 'embedding.geneset.oxygen']
# All remaining feature types are "annotation-free."
ANNOTATION_FREE_FEATURE_TYPES = [f for f in FEATURE_SUBTYPES + FEATURE_TYPES if f not in ANNOTATION_BASED_FEATURE_TYPES]


def plot_order_feature_types(feature_types, order_by:Dict[str, float]=dict()) -> List[str]:
    '''Order the input list of feature types such that annotation-based feature types and annotation-free
    feature types are grouped together. This function also takes a dictionary of values as input, which can be used to 
    order features within annotation-free and annotation-based categories (e.g. to sort by increasing order of validation accuracy).'''
    ordered_feature_types = []
    # First split the input features into annotation-free and annotation-based categories. 
    for group in [np.array(ANNOTATION_BASED_FEATURE_TYPES), np.array(ANNOTATION_FREE_FEATURE_TYPES)]:
        group = np.array([f for f in feature_types if f in group])
        # Sort in ascending order according to the values in the order_by dictionary. 
        sort_idxs = np.argsort([order_by.get(f, 0) for f in group])
        # Use the indices to sort the feature type lists within each category.
        ordered_feature_types += group[sort_idxs].tolist()


    return ordered_feature_types


def plot_training_curve(results:Dict, ax:plt.Axes=None) -> NoReturn:
    '''Plot the Nonlinear classifier training curve. Save the in the current working directory.'''
    assert results['model_class'] == 'nonlinear', 'plot_training_curve: Model class must be Nonlinear.'
    assert 'training_losses' in results, 'plot_training_curve: Results dictionary must contain training losses.'

    # Extract some information from the results dictionary. 
    train_losses = results.get('training_losses', [])
    train_accs = results.get('training_accs', [])
    val_losses = results.get('validation_losses', [])
    val_accs = results.get('validation_accs', [])
    feature_type = results['feature_type']

    acc_ax = loss_ax.twinx() # Create another axis for displaying accuracy.
    loss_ax.set_title(f'{PRETTY_NAMES[feature_type]} training curve') # Set the title.

    lines = loss_ax.plot(train_losses, label='training loss')
    lines += loss_ax.plot(val_losses, linestyle='--', label='validation loss')
    lines += acc_ax.plot(val_accs, linestyle='--', c=COLORS[1], label='validation accuracy')
    lines += acc_ax.plot(train_accs, label='training accuracy')

    loss_ax.set_ylabel('MSE loss')
    loss_ax.set_xlabel('epoch') # Will be the same for both axes.
    acc_ax.set_ylabel('balanced accuracy')
    acc_ax.set_ylim(top=1, bottom=0)

    acc_ax.legend(lines, ['training loss', 'validation loss', 'validation accuracy', 'training accuracy'])

    plt.tight_layout()


def plot_percent_above_random_axis(ax:plt.Axes, binary:bool=False):
    '''Add a second axis to a plot indicating the percent above random performance.'''
    random_baseline = 0.5 if binary else 0.33 # Expected performance for random classifier on task. 
    # Add a second set of y-ticks on the right to indicate percentage performance increase over random.
    new_ax = ax.twinx()
    new_ax.set_ylim(0, 1)
    yticks = np.round(100 * (np.arange(0, 1.1, 0.1) - random_baseline) / random_baseline, 0)
    ytick_labels = [f'{v:.0f}%' for v in yticks]
    new_ax.set_yticks(yticks, ytick_labels)
    new_ax.set_ylabel('percent above random', rotation=270)
    # Add horizontal line marking model performance with random classification. 
    # ax.axhline(random_baseline, color='grey', linestyle='--', linewidth=2, zorder=-10)


def plot_balanced_accuracy_axis(ax:plt.Axes, random_baseline:float=0.33):
    '''

    :param ax: The axis to format. 
    :param random_baseline: Expected performance for random classifier on task. 
    '''

    # Set up left y-axis with the balanced accuracy information. 
    ax.set_ylabel('balanced accuracy')
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1)) # xmax is the number to divide by for the percentage.

    if random_baseline is not None:
        # Add horizontal line marking model performance with random classification. 
        ax.axhline(random_baseline, color='grey', linestyle='--', linewidth=2, zorder=-10)


def plot_model_accuracy_barplot(results:Dict[str, Dict], ax:plt.Axes=None, colors=['tab:blue', 'tab:green'], feature_type_order:List[str]=None) -> NoReturn:
    '''Plot a barplot which displays the training and validation accuracy for a model type trained on different 
    feature types. 

    :param results: A dictionary mapping each feature type to the training results generated by train.py.
    :param ax: A matplotlib Axes object on which to plot the figure. 
    :param colors: A list of two colors to distinguish between annotation-based and annotation-free feature types.
    :param feature_type_order: The order of the feature types along the x-axis. 
    '''
    # Make sure annotation-based and annotation free are iterated over in clumps.
    # Also want to make sure the bars are plotted in order of best validation accuracy, within their clumps.  
    feature_types = feature_type_order if feature_type_order is not None else plot_order_feature_types(list(results.keys())) 
    
    # Extract the final balanced accuracies on training and validation sets from the results dictionaries. 
    train_accs  = [results[feature_type]['training_acc'] for feature_type in feature_types]
    val_accs  = [results[feature_type]['validation_acc'] for feature_type in feature_types]

    # Map annotation-free or annotation-based features to different colors. 
    colors = [plot_color_palette()['blue'] if f in ANNOTATION_BASED_FEATURE_TYPES else plot_color_palette()['green'] for f in feature_types] 
    ax.bar(np.arange(0, len(feature_types), 1) - 0.2, train_accs, width=0.4, label='training', color=colors, edgecolor='k', linewidth=0.5, hatch='//')
    ax.bar(np.arange(0, len(feature_types), 1) + 0.2, val_accs, width=0.4, label='validation', color=colors, edgecolor='k', linewidth=0.5)

    # Custom legend. Colors indicate annotation-free or annotation-full, and hatching indicates training or validation set. 
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='k', linewidth=0.5, hatch='////')]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='k', linewidth=0.5, hatch=''))
 
    labels = ['training', 'validation']
    plt.legend(handles, labels,  loc='lower right')
    
    # Label bins with the feature name. 
    ax.set_xticks(np.arange(0, len(feature_types), 1), [PRETTY_NAMES[f] for f in feature_types], rotation=45, ha='right')

    plot_balanced_accuracy_axis(ax, random_baseline=0.5 if results[feature_types[0]]['binary'] else 0.33 )



def plot_confusion_matrix(results:Dict[str, Dict], ax:plt.Axes=None) -> NoReturn:
    '''Plots a confusion matrix for a particular model evaluated on data of a particular feature type.'''

    binary = results['binary'] # Assume all have the same value, but might want to add a check.
    classes = results['classes'] # This should also be the same for each feature type. 

    # Extract the confusion matrix, which is a flattened list, and need to be reshaped. 
    dim = 2 if binary else 3 # Dimension of the confusion matrix.
    confusion_matrix = np.array(results['confusion_matrix']).reshape(dim, dim)
    
    confusion_matrix = pd.DataFrame(confusion_matrix, columns=classes, index=classes)
    confusion_matrix = confusion_matrix.apply(lambda x: x/x.sum(), axis=1) # Normalize the matrix.
    ax.set_xlabel('predicted label')
    ax.set_ylabel('true label')
    sns.heatmap(confusion_matrix, ax=ax, cmap=plot_color_map(), annot=True, fmt='.1%', cbar=False, linecolor='k', linewidths=0.2)
    # Rotate the tick labels on the x-axis of each subplot.
    ax.set_xticks(np.arange(len(classes)) + 0.5, classes, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(classes)) + 0.5, classes, rotation=0)


def plot_phylo_cv(results:Dict[str, Dict], ax:plt.Axes=None, legend:bool=False, color:str=None) -> NoReturn:
    '''Plots the results of phlogenetic bias analysis.
    
    :param results: A dictionary containing the results of phylo-cv runs. 
    :param ax: A matplotlib Axes object on which to plot the results. 
    :param legend: Whether or not to plot the legend within the function. 
    :param color: If specified, the color of all lines in the plot. 
    ''' 

    levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'][::-1]
    labels, handles = [], []

    for feature_type, results in results.items():

        # Plot the error bar, as well as scatter points for each level. 
        means = [np.mean(results['scores'][level]) for level in levels] # Extract the mean F1 scores.
        errs = [np.std(results['scores'][level]) / np.sqrt(len(results['scores'][level])) for level in levels] # Extract the standard errors. 
        # Store information for the legend. 
        if color is not None:
            handles.append(ax.errorbar(np.arange(1, len(levels) + 1), means, yerr=errs, capsize=3, c=color)[0])
        else: # If no color is specified, just use whatever color is in the lineup for the color cycler. 
            handles.append(ax.errorbar(np.arange(1, len(levels) + 1), means, yerr=errs, capsize=3)[0])
        labels.append(PRETTY_NAMES.get(feature_type, feature_type))

    plot_balanced_accuracy_axis(ax, random_baseline=None)
    ax.set_xticks(np.arange(1, len(levels) + 1), labels=levels, rotation=45)
    ax.set_xlabel('holdout level')

    if legend: # If specified, go ahead and plot the legend. 
        ax.legend(handles, labels)

    return handles, labels # Return information to use for plotting the legend. 



# def plot_fit_logistic_curve(x:np.ndarray, y:np.ndarray, min_x_val:int=1000, max_x_val:int=40000):
#     '''Fit a logistic curve to the input data.'''

#     def logistic(x:np.ndarray, k:float, x0:float, L:float):
#         '''Logistic curve equation for fitting the data.'''
#         return L / (1 + np.exp(-k * (x - x0)))

#     def residuals(params:np.ndarray, y:np.ndarray, x:np.ndarray):
#         '''Compute residuals between least squares approximation and the true y-values.'''
#         k, x0, L = params
#         err = y - logistic(x, k, x0, L)
#         return err

#     params = [0.0001, 0, 1] # Initial guesses for the parameters.
#     lsq = scipy.optimize.least_squares(residuals, params, args=(y, x)) 
#     k, x0, L = lsq.x # Get the fitted params from the least-squares result. 
#     # print('k =', k)
#     # print('x0 =', x0)
#     # print('L =', L)
#     # Compute x and y values for the fitted function. 
#     x = np.linspace(min_x_val, max_x_val, 100)
#     y = logistic(x, k, x0, L)
#     return x, y
