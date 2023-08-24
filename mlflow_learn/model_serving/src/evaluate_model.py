from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss, confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def predict_on_test_data(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred


def predict_prob_on_test_data(model,X_test):
    y_pred = model.predict_proba(X_test)
    return y_pred


def get_metrics(y_true, y_pred, y_pred_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    entropy = log_loss(y_true, y_pred_prob)
    return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)}


def create_roc_auc_plot(clf, X_test, y_test):
    y_pred_prob = clf.predict_proba(X_test)[:, 1]  # Get predicted probabilities of the positive class
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    # plt.savefig('roc_auc_curve.png')
    plt.show()



def create_confusion_matrix_plot(clf, X_test, y_test):
    # Get predicted labels
    y_pred = clf.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
      
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    # plt.savefig('confusion_matrix.png')
    plt.show()
