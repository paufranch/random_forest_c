#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include "random_forest.h"
#include "predict.h"
#include "metrics.h"

Metrics EvaluateModel(RandomForest* rf, Dataset* data)
{
    Metrics m = {0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0};

    for(int i = 0; i < data->n_samples; i++)
    {
        int prediction = predict_forest(rf, data->vars[i]);
        int real = data->target_var[i];

        if(prediction == 1 && real == 1) m.true_positives++;
        else if(prediction == 1 && real == 0) m.false_positives++;
        else if(prediction == 0 && real == 0) m.true_negatives++;
        else if(prediction == 0 && real == 1) m.false_negatives++;
    }

    // Accuracy
    if(data->n_samples > 0)
    {
        m.accuracy = (double)(m.true_positives + m.true_negatives) / (double)data->n_samples; 
    }

    // Precision
    if(m.true_positives + m.false_positives > 0)
    {
        m.precision = (double)m.true_positives / (m.true_positives + m.false_positives);
    }

    // Recall
    if(m.true_positives + m.false_negatives > 0)
    {
        m.recall = (double)m.true_positives / (m.true_positives + m.false_negatives);
    }

    // F1
    if(m.precision + m.recall > 0)
    {
        m.f1score = 2.0 * (m.precision * m.recall) / (m.precision + m.recall);
    }

    return m;
}

void PrintMetrics(Metrics m)
{
    printf("==== Forest Metrics ====\n");
    printf("Accuracy: %.4f (%.2f%%)\n", m.accuracy, m.accuracy * 100);
    printf("Precision: %.4f\n", m.precision);
    printf("Recall: %.4f\n", m.recall);
    printf("F1 Score: %.4f\n", m.f1score);
    printf("==== Confusion Matrix ====\n");
    printf("True Positives: %d\n", m.true_positives);
    printf("False Positives: %d\n", m.false_positives);
    printf("True Negatives: %d\n", m.true_negatives);
    printf("False Negatives: %d\n", m.false_negatives);
}