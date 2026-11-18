import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold


def _splits(labels, groups, count, seed):
    for label in np.unique(labels):
        if len(np.unique(groups[labels == label])) < count:
            raise ValueError(f'Class {label} has fewer than {count} independent groups')

    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    result = []
    classes = set(labels)

    for train, test in splitter.split(np.zeros(len(labels)), labels, groups):
        if set(groups[train]) & set(groups[test]):
            raise ValueError('Group overlap across split')

        if set(labels[train]) != classes or set(labels[test]) != classes:
            raise ValueError('A fold is missing a class; revise the split configuration')

        result.append((train, test))

    return result


def nested_group_folds(labels, groups, outer_folds=4, inner_folds=3, seed=42):
    labels, groups = np.asarray(labels), np.asarray(groups)

    if len(labels) != len(groups) or len(labels) == 0:
        raise ValueError('Expected nonempty aligned labels and groups')

    if any(group is None or str(group).strip() == '' for group in groups):
        raise ValueError('Missing patient/group IDs must be resolved before splitting')

    for group in np.unique(groups):
        if len(np.unique(labels[groups == group])) != 1:
            raise ValueError('A patient/group has inconsistent class labels')

    result = []

    for train, test in _splits(labels, groups, outer_folds, seed):
        inner = [
            (train[inner_train], train[inner_valid])
            for inner_train, inner_valid in _splits(
                labels[train], groups[train], inner_folds, seed,
            )
        ]

        result.append({'train': train, 'test': test, 'inner': inner})

    return result


def oof_metrics(truth, predicted, labels):
    truth, predicted = np.asarray(truth), np.asarray(predicted)

    if len(truth) != len(predicted) or not len(truth):
        raise ValueError('Expected nonempty aligned predictions')

    return {
        'accuracy': float(accuracy_score(truth, predicted)),
        'macro_f1': float(
            f1_score(truth, predicted, labels=labels, average='macro', zero_division=0)
        ),
        'confusion_matrix': confusion_matrix(truth, predicted, labels=labels).tolist(),
        'per_class': classification_report(
            truth,
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def patient_bootstrap_interval(truth, predicted, patients, labels, samples=2000, seed=42):
    truth, predicted, patients = (
        np.asarray(truth), np.asarray(predicted), np.asarray(patients),
    )

    if len(truth) != len(predicted) or len(truth) != len(patients) or samples < 1:
        raise ValueError('Expected aligned predictions/patients and a positive bootstrap budget')

    if any(patient is None or str(patient).strip() == '' for patient in patients):
        raise ValueError('Patient IDs are required for a clustered bootstrap')

    grouped = {}

    for patient in np.unique(patients):
        indices = np.flatnonzero(patients == patient)
        classes = np.unique(truth[indices])

        if len(classes) != 1:
            raise ValueError('Patient has inconsistent class labels')

        grouped.setdefault(classes[0], []).append(indices)

    if set(grouped) != set(labels):
        raise ValueError('Every reported class must have OOF support')

    rng = np.random.default_rng(seed)
    scores = []

    for _ in range(samples):
        indices = np.concatenate([
            class_patients[patient_index]
            for class_patients in grouped.values()
            for patient_index in rng.integers(0, len(class_patients), size=len(class_patients))
        ])

        scores.append(
            f1_score(
                truth[indices],
                predicted[indices],
                labels=labels,
                average='macro',
                zero_division=0,
            )
        )

    lower, upper = np.quantile(scores, [.025, .975])

    return {
        'lower': float(lower),
        'upper': float(upper),
        'samples': samples,
        'seed': seed,
        'method': 'class-stratified patient-cluster bootstrap of fixed OOF predictions',
        'limitation': (
            'Conditional approximation; excludes training variability '
            'and external-domain uncertainty'
        ),
    }
