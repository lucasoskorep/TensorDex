import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


def get_metrics(gen, model, save_predictions_file=None):
    model_output = model.predict(gen, verbose=True, workers=12)
    prediction_indices = np.argmax(model_output, axis=1)
    label_index = {v: k for k, v in gen.class_indices.items()}
    predictions = [label_index[p] for p in prediction_indices]
    reals = [label_index[p] for p in gen.classes]
    acc = accuracy_score(reals, predictions)
    ll = log_loss(gen.classes, model_output, labels=[l for l in label_index.keys()])
    conf_mat = confusion_matrix(reals, predictions, labels=[l for l in label_index.values()])
    # print(classification_report(reals, predictions, labels=[l for l in label_index.values()]))
    print("Testing accuracy score is ", acc)
    print("Confusion Matrix", conf_mat)
    if save_predictions_file:
        df = pd.DataFrame(columns=['fname', 'prediction', 'true_val'])
        df['fname'] = [x for x in gen.filenames]
        df['prediction'] = predictions
        df["true_val"] = reals
        df.to_csv(save_predictions_file, index=False)
    return acc, ll
