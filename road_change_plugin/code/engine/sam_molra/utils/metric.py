import numpy as np


class PixelMetric:
    def __init__(self, num_classes, logdir=None, class_names=None):
        self.num_classes = num_classes
        self.logdir = logdir
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def forward(self, target, pred):
        target = np.asarray(target, dtype=np.int64).reshape(-1)
        pred = np.asarray(pred, dtype=np.int64).reshape(-1)
        valid_mask = (target >= 0) & (target < self.num_classes)
        encoded = self.num_classes * target[valid_mask] + pred[valid_mask]
        bincount = np.bincount(encoded, minlength=self.num_classes ** 2)
        self.confusion_matrix += bincount.reshape(self.num_classes, self.num_classes)

    def summary_all(self):
        cm = self.confusion_matrix.astype(np.float64)
        tp = np.diag(cm)
        actual = cm.sum(axis=1)
        predicted = cm.sum(axis=0)
        union = actual + predicted - tp

        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
        recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(tp),
            where=(precision + recall) > 0,
        )
        iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
        overall_acc = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0

        per_class = {}
        for idx, class_name in enumerate(self.class_names):
            per_class[class_name] = {
                'precision': round(float(precision[idx]), 6),
                'recall': round(float(recall[idx]), 6),
                'f1': round(float(f1[idx]), 6),
                'iou': round(float(iou[idx]), 6),
            }

        return {
            'overall_acc': round(overall_acc, 6),
            'mean_iou': round(float(iou.mean()), 6),
            'per_class': per_class,
        }
