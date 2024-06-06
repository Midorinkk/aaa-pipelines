## ДЗ ААА "Инфраструктура и пайплайны"
Для прогонов в `mlflow` были использованы следующие команды:
- **baseline**:
  
  ```console
  python main_script.py
  ```
- **baseline_tuning**:

  ```console
  python main_script.py --run_name baseline_tuning --model_name baseline_tuning --optimizer_name Adagrad
  ```

- **ALS_best**:

  ```console
  python main_script.py --run_name ALS_best --model_name ALS_best 
  ```
