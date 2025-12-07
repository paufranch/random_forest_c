#ifndef METRICS_H
#define METRICS_H

#include "random_forest.h"

typedef struct
{
    double accuracy;
    double precision;
    double recall;
    double f1score;
    int true_positives;
    int true_negatives;
    int false_positives;
    int false_negatives;
} Metrics;

Metrics EvaluateModel(RandomForest* rf, Dataset* data);

void PrintMetrics(Metrics m);

#endif