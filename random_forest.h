#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#define MAX_DEPTH 10
#define MIN_SAMPLES 2

typedef struct
{
    int leaf;
    int featureId;
    double threshold;
    double value;
    struct Node* left;
    struct Node* right;
} Node;

typedef struct
{
    Node* root;
} DecisionTree;

typedef struct
{
    double** vars;
    int* target_var;
    int n_samples;
    int n_features;
} Dataset;

typedef struct {
    DecisionTree* trees;
    int numTrees;
    int maxDepth;
    int numFeatures;
} RandomForest;

#endif RANDOM_FOREST_H