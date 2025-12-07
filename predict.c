#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include "random_forest.h"
#include "predict.h"

double predict_tree(Node* node, double* sample)
{
    if(node->leaf) return node->value;

    if(sample[node->featureId] <= node->threshold)
        return predict_tree(node->left, sample);
    else
        return predict_tree(node->right, sample);
}

int predict_forest(RandomForest* rf, double* sample)
{
    int votes0 = 0;
    int votes1 = 0;

    for(int i = 0; i < rf->numTrees; i++)
    {
        double value = predict_tree(rf->trees[i].root, sample);
        if(value == 0.0) votes0++;
        else votes1++;
    }

    return (votes0 > votes1) ? 0 : 1;
}