import sklearn.metrics

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

