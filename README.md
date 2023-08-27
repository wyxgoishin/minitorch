# MiniTorch
## Fundamental
### Task 0.5
Q: No module named 'altair.vegalite.v4'

A: You have installed the wrong version of altair, try reinstall altair with specific version, e.g., 
`pip install altair==4.0`

Q: "ModuleNotFoundError: No module named `xxx`" when run `streamlit run app.py -- 0` in Windows

A: This is due to the lack of python path, try `python -m streamlit run app.py -- 0` instead

DataSet:
![img.png](resources/task0/task0.5_dataset.png)
Model:
![img.png](resources/task0/task0.5_model.png)
HyperParameter(Note that the lr_rate is too large and can cause divergence but just save time):
![img.png](resources/task0/task0.5_hyperparameters.png)
Loss:
![img.png](resources/task0/task0.5_loss.png)

### Task 1.5
![img.png](resources/task1/task1.5_Dataset.png)
![img.png](resources/task1/task1.5_Model.png)
![img.png](resources/task1/task1.5_Hyperparameter.png)
![img.png](resources/task1/task1.5_Lossgraph.png)