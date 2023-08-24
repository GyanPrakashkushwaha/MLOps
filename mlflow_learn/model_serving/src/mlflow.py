import mlflow

class ModelRegistry:
    def __init__(self,experiment_name ,run_name) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        mlflow.set_experiment(experiment_name)

    def create_experiment(self,run_name, run_metrics,model, confusion_matrix_path = None, 
                        roc_auc_plot_path = None, run_params=None):
        
        
        with mlflow.start_run():
            experiment_name = self.experiment_name
            
            if not run_params == None:
                for param in run_params:
                    mlflow.log_param(param, run_params[param])
                
            for metric in run_metrics:
                mlflow.log_metric(metric, run_metrics[metric])
            
            mlflow.sklearn.log_model(model, "model")
            
            if not confusion_matrix_path == None:
                mlflow.log_artifact(confusion_matrix_path, 'confusion_materix')
                
            if not roc_auc_plot_path == None:
                mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")
            
            mlflow.set_tag("tag1", "Random Forest")
            mlflow.set_tags({"tag2":"Randomized Search CV", "tag3":"Production"})
                
        # print('Run - %s is logged to Experiment - %s' %(run_name, 'experiment_name'))


        




        








