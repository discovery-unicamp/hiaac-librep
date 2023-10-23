import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
import pandas as pd

def resize_data(data,input_shape):        
        if(len(data.shape)==2): 
            submatrix_rows = input_shape[0]
            submatrix_cols = input_shape[1]
            result_matrices = []
    
            for row in data:
                row_matrices = np.split(row, len(row) // submatrix_cols)
                result_matrices.append(row_matrices)
            final_array = np.array(result_matrices)
            return final_array
        else:
            return data
    
def get_resize_data(X,X_val,y,y_val,input_shape):
    if X_val is None:
          X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X=resize_data(X,input_shape)
    X_val=resize_data(X_val,input_shape)
    #y=pd.get_dummies(y).to_numpy()
    #y_val=pd.get_dummies(y_val).to_numpy()
    return X,X_val,y,y_val

def transform_combinations(start=1,end=4):
    from itertools import combinations
    transform_funcs_names = ['noise_vectorized',
                             'scaling_vectorized', 
                             'rotation_vectorized', 
                             'negate_vectorized', 
                             'time_flip_vectorized', 
                             'channel_shuffle_vectorized', 
                             #'time_segment_permutation',  
                             'time_shift',
                             'amplify_attenuate',
                             'add_random_noise',
                             'random_phase_shift', 
                             'spectral_distortion',
                             'phase_modulation']
    short_transform_names = [
    'noise',
    'scaling',
    'rotation',
    'negate',
    'time_flip',
    'channel_shuffle',
    'segment_permutation',
    'time_shift',
    'amplify',
    'add_noise',
    'phase_shift',
    'distortion',
    'modulation'
]
    transform_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    transformations=[]
    for n_transf in range(start,end):
            transformations_ = [list(comb) for comb in combinations(transform_indices, n_transf)]
            transformations.extend(transformations_) 
    return transformations

def evaluate_model_simple(pred, truth, is_one_hot=True, return_dict=True):
        if is_one_hot:
            truth_argmax = np.argmax(truth, axis=1)
            pred_argmax = np.argmax(pred, axis=1)
        else:
            truth_argmax = truth
            pred_argmax = pred

        test_cm = sklearn.metrics.confusion_matrix(truth_argmax, pred_argmax)
        test_f1 = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='macro')
        test_precision = sklearn.metrics.precision_score(truth_argmax, pred_argmax, average='macro')
        test_recall = sklearn.metrics.recall_score(truth_argmax, pred_argmax, average='macro')
        test_kappa = sklearn.metrics.cohen_kappa_score(truth_argmax, pred_argmax)
        test_f1_micro = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='micro')
        test_f1_weighted = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='weighted')

        if return_dict:
            return {
                #'Confusion Matrix': test_cm, 
                'F1 Macro': test_f1, 
                'F1 Micro': test_f1_micro, 
                'F1 Weighted': test_f1_weighted, 
                'Precision': test_precision, 
                'Recall': test_recall, 
                'Kappa': test_kappa
            },test_cm
        else:
            return (test_cm, test_f1, test_f1_micro, test_f1_weighted, test_precision, test_recall, test_kappa)


def evaluate_model(linear_eval_best_model,X_test,y_test,device):
        import torch
        import numpy as np
        import sklearn.metrics
        num_classes = len(y_test[1])
        linear_eval_best_model.eval()

        # Convert your numpy test data to PyTorch tensors
        test_inputs = torch.tensor(X_test, dtype=torch.float32)
        test_labels = torch.tensor(y_test, dtype=torch.float32)

        # Move the data to the same device as the model
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)

        # Evaluate the model with the best weights
        with torch.no_grad():
            test_outputs = linear_eval_best_model(test_inputs)
            predicted_classes = torch.argmax(test_outputs, dim=1)

        predicted_classes_np = predicted_classes.cpu().numpy()

        true_classes = np.argmax(y_test, axis=1)

        evaluation_results_best =  evaluate_model_simple(predicted_classes_np, true_classes, is_one_hot=False, return_dict=True)


        return evaluation_results_best



def plot_tsne(base_model_of_trained_simclr):
    intermediate_model = copy.deepcopy(base_model_of_trained_simclr)
    
    perplexity = 30.0
    test_data = torch.tensor(X_test, dtype=torch.float32).to(device)

    embeddings = intermediate_model(test_data).cpu().detach().numpy()
    tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=1, random_state=42)
    tsne_projections = tsne_model.fit_transform(embeddings)
    label_list_full_name = [    "climbingdown",
    "climbingup",
    "jumping",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking"]
    custom_colors = ['yellow', 'brown', 'red', 'purple', 'blue', 'green', 'cyan']
    tsne_df = pd.DataFrame(data=tsne_projections, columns=['Component 1', 'Component 2'])
    labels_argmax = np.argmax(y_test, axis=1)
    tsne_df['Class'] = labels_argmax
    tsne_df['Class_Name'] = tsne_df['Class'].map(lambda label: label_list_full_name[label])
    fig = px.scatter(tsne_df, x='Component 1', y='Component 2', color='Class_Name',
                     color_discrete_sequence=custom_colors, title='t-SNE Scatter Plot')

    iplot(fig)
    fig.write_html("scatter_plot.html")
    # Registrar el archivo HTML en MLflow
    mlflow.log_artifact("scatter_plot.html")

def save_matriz_confusao(matriz_conf):
    #matriz_conf = confusion_matrix(y_test.values.ravel(), y_predict)
    fig = plt.figure()
    ax = plt.subplot()
    #sns.heatmap(matriz_conf, annot=True, cmap='Blues', ax=ax);
    sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues', ax=ax)  # Usar 'fmt' para mostrar valores como enteros
    ax.set_xlabel('Valor Predito');
    ax.set_ylabel('Valor Real'); 
    ax.set_title('Matriz de Confusão'); 
    #ax.xaxis.set_ticklabels(['Classe 1', 'Classe 2', 'Classe 3']);
    #ax.yaxis.set_ticklabels(['Classe 1', 'Classe 2', 'Classe 3']);
    plt.close()
    temp_name = "confusion-matrix.png"
    fig.savefig(temp_name)
####################################################################################################################
########################################### Model Tracking #########################################################
    mlflow.log_artifact(temp_name, "confusion-matrix-plots")
    return fig