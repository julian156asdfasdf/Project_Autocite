# Test hyperparameter optimization using optuna functions

import optuna
import matplotlib.pyplot as plt

# The study must exist in this place (using Optuna Dashboard)
study_name = "Autocite_Hyperparam_Optim_Snowflake"
storage = "sqlite:///Autocite.db"
study = optuna.load_study(study_name=study_name, storage=storage)

param_importance = optuna.importance.get_param_importances(study)
print(param_importance)

ax = optuna.visualization.matplotlib.plot_optimization_history(study, target_name="Accuracy")
plt.title('Accuracy Over Trials')
plt.xlim(-1,20)
plt.show()
ax = optuna.visualization.matplotlib.plot_slice(study, params=["distance_measure"], target_name="Accuracy")
plt.title('Accuracy vs Distance_Measure')
plt.xlabel('Distance Metric')
plt.show()
ax = optuna.visualization.matplotlib.plot_slice(study, params=["alpha"], target_name="Accuracy")
plt.title('Accuracy vs Alpha')
plt.xlabel('Alpha')
plt.show()
ax = optuna.visualization.matplotlib.plot_slice(study, params=["context_size_idx"], target_name="Accuracy")
plt.title('Accuracy vs Context_Size')
plt.xlabel('Context_size_idx')
plt.show()