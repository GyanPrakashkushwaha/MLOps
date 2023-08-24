from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix



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


def create_roc_auc_plot(clf, X_data, y_data):
    metrics.roc_curve(X_data, y_data) 
    plt.savefig('roc_auc_curve.png')


def create_confusion_matrix_plot(clf, X_test, y_test):    
    plot_confusion_matrix(clf, X_test, y_test)
    plt.savefig('confusion_matrix.png')