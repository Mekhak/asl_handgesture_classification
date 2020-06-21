import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn

def report(label, y_pred):
    target_names  = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                     'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    print(accuracy_score(label, y_pred))

    conf_matrix = confusion_matrix(label, y_pred)
    df_cm = pd.DataFrame(conf_matrix, index=target_names, columns=target_names)

    plt.figure(figsize=(25, 16))
    plt.title("Confusion matrix")
    sn.heatmap(df_cm, annot=True)

    print('Classification report')
    print(classification_report(label, y_pred, target_names=target_names))