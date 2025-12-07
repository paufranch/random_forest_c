#ifndef PREDICT_H
#define PREDICT_H

double predict_tree(Node* node, double* sample);
int predict_forest(RandomForest* rf, double* sample);

#endif