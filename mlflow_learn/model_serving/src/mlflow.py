import mlflow

class ModelRegistry:
    def __init__(self,experiment_name ,run_name) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        mlflow.set_experiment(experiment_name)

    def create_exp_and_register_model(self,experiment_name,run_name,run_metrics,model,confusion_matrix_path = None, 
                        roc_auc_plot_path = None, run_params=None):
        
        mlflow.set_tracking_uri("http://localhost:5000") 
        #use above line if you want to use any database like sqlite as backend storage for model else comment this line
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            if not run_params == None:
                for param in run_params:
                    mlflow.log_param(param, run_params[param])
                
            for metric in run_metrics:
                mlflow.log_metric(metric, run_metrics[metric])
            
            if not confusion_matrix_path == None:
                mlflow.log_artifact(confusion_matrix_path, 'confusion_materix')
                
            if not roc_auc_plot_path == None:
                mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")
            
            mlflow.set_tag("tag1", "Random Forest")
            mlflow.set_tags({"tag2":"Randomized Search CV", "tag3":"Production"})
            mlflow.sklearn.log_model(model, "model",registered_model_name="learning MLOps")
        




        








