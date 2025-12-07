#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include "random_forest.h"
#include "metrics.h"

double gini_impurity(int* labels, int n)
{
    if (n == 0) return 0.0;

    int count0 = 0, count1 = 0;
    for(int i = 0; i < n; i++)
    {
        if(labels[i] == 0) count0++;
        else count1++;
    }

    double p0 = (double)count0 / n;
    double p1 = (double)count1 / n;
    return 1.0 - (p0 * p0 + p1 * p1);
}

void find_best_split(Dataset* data, int* sampleIndices, int n_samples, int* best_feature,
double* best_threshold, double* best_gain)
{
    *best_gain = 0.0;
    *best_feature = 0.0;

    double root_impurity = gini_impurity(data->target_var, n_samples);

    // Random subset of features, in this case sqrt(featuresamount)
    // Could be changed into a different proportion maybe??
    int n_features_test = (int)sqrt(data->n_features);

    for(int f = 0; f < n_features_test; f++)
    {
        int featureTest = rand() % data->n_features; // Pick random feature

        for(int i = 0; i < n_samples; i++) // For every sample
        {
            // Get test feature value for current sample
            double threshold = data->vars[sampleIndices[i]][featureTest];

            // Split the dataset into two groups:
            // Left group is the samples with a value of random feature lowerequ than threshold
            // Right group is the samples with a value of random feature higher than threshold
            // This calculates the amount of rows in each group
            int n_left = 0, n_right = 0;
            for(int j = 0; j < n_samples; j++)
            {
                if(data->vars[sampleIndices[j]][featureTest] <= threshold) n_left++;
                else n_right++;
            }

            // If one of the two groups is empty, then the other contains the whole dataset, thus
            // it is not a good split, we continue searching for a better gain.
            if(n_left == 0 || n_right == 0) continue;

            // Calculate impurity for current split

            // Gather target variable for left and right groups (could be done in the first loop, but it wouldn't make the code much faster)
            int* left_indices = malloc(n_left * sizeof(int));
            int* right_indices = malloc(n_right * sizeof(int));
            int l_idx = 0, r_idx = 0;

            for(int j = 0; j < n_samples; j++)
            {
                if(data->vars[sampleIndices[j]][featureTest] <= threshold)
                    left_indices[l_idx++] = data->target_var[sampleIndices[j]];
                else
                    right_indices[r_idx++] = data->target_var[sampleIndices[j]];
            }

            double left_impurity = gini_impurity(left_indices, n_left);
            double right_impurity = gini_impurity(right_indices, n_right);

            double weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples;
            double gain = root_impurity - weighted_impurity;

            if(gain > *best_gain)
            {
                *best_gain = gain;
                *best_feature = featureTest;
                *best_threshold = threshold;
            }

            free(left_indices);
            free(right_indices);
        }
    }
}

Node* BuildTree(Dataset *data, int *sampleIndices, int n_samples, int depth, int max_depth)
{
    int count_0 = 0;
    for (int i = 0; i < n_samples; i++) {
        if (data->target_var[sampleIndices[i]] == 0) count_0++;
    }

    printf("Depth %d: %d samples (0:%d, 1:%d)\n", depth, n_samples, count_0, n_samples-count_0);

    Node* node = malloc(sizeof(Node));

    // base case: deeper tree than allowed or not enough samples to split, make a leaf
    if(depth >= max_depth || n_samples < MIN_SAMPLES)
    {
        printf("-> Leaf (max depth reached)\n");
        node->leaf = 1;

        // Get majority targetvariable to choose 0 or 1 for binary classification
        int count0 = 0;
        for(int i = 0; i < n_samples; i++)
        {
            if(data->target_var[sampleIndices[i]] == 0) count0++;
        }
        node->value = (count0 > n_samples - count0) ? 0.0 : 1.0;
        node->left = NULL;
        node->right = NULL;
        return node;
    }

    int best_feature = -1;
    double best_threshold = 0.0, best_gain = 0.0;
    find_best_split(data, sampleIndices, n_samples, &best_feature, &best_threshold, &best_gain);

    printf("Best split: Feature %d, Threshold %.2f, Gain %.4f\n", 
       best_feature, best_threshold, best_gain);

    // Didn't find a good split, make a leaf too
    if(best_feature == -1 || best_gain == 0.0)
    {
        printf("-> Leaf (no good split)\n");
        node->leaf = 1;

        // Get majority targetvariable to choose 0 or 1 for binary classification
        int count0 = 0;
        for(int i = 0; i < n_samples; i++)
        {
            if(data->target_var[sampleIndices[i]] == 0) count0++;
        }
        node->value = (count0 > n_samples - count0) ? 0.0 : 1.0;
        node->left = NULL;
        node->right = NULL;
        return node;
    }

    // ELSE create non leaf
    node->leaf = 0;
    node->featureId = best_feature;
    node->threshold = best_threshold;

    // Gather target variable for left and right groups
    int* left_indices = malloc(n_samples * sizeof(int));
    int* right_indices = malloc(n_samples * sizeof(int));
    int n_left = 0, n_right = 0;

    for(int j = 0; j < n_samples; j++)
    {
        if(data->vars[sampleIndices[j]][best_feature] <= best_threshold)
            left_indices[n_left++] = data->target_var[sampleIndices[j]];
        else
            right_indices[n_right++] = data->target_var[sampleIndices[j]];
    }

    node->left = BuildTree(data, left_indices, n_left, depth+1, max_depth);
    node->right = BuildTree(data, right_indices, n_right, depth+1, max_depth);

    free(left_indices);
    free(right_indices);

    return node;
}

void TrainTree(DecisionTree* tree, Dataset* data, int max_depth)
{
    int* featureIndices = malloc(data->n_samples * sizeof(int));
    for(int i = 0; i < data->n_samples; i++)
    {
        featureIndices[i] = rand() % data->n_samples;
    }

    tree->root = BuildTree(data, featureIndices, data->n_samples, 0, max_depth);
    free(featureIndices);
}

void TrainForest(RandomForest* rf, Dataset* data)
{
    for(int i = 0; i < rf->numTrees; i++)
    {
        char trainmsg[64];
        sprintf(trainmsg, "Training tree num %d...\n", i+1);
        write(0, trainmsg, strlen(trainmsg));
        TrainTree(&rf->trees[i], data, rf->maxDepth);
    }

    char finishmsg[64];
    sprintf(finishmsg, "Finished training model!\n");
    write(0, finishmsg, strlen(finishmsg));
}

RandomForest* CreateRandomForest(int numTrees, int maxDepth, int numFeatures)
{
    RandomForest* rf = malloc(sizeof(RandomForest));
    rf->numTrees = numTrees;
    rf->maxDepth = maxDepth;
    rf->numFeatures = numFeatures;
    rf->trees = malloc(numTrees * sizeof(DecisionTree));

    char sucsmsg[256];
    sprintf(sucsmsg, "Forest created successfully with %d trees and %d features.\n", numTrees, numFeatures);
    write(0, sucsmsg, strlen(sucsmsg));

    return rf;
}

// POSTORDER Deletes the tree from memory
void FreeTree(Node* node)
{
    if(node == NULL) return;
    FreeTree(node->left);
    FreeTree(node->right);
    free(node);
}

void FreeRandomForest(RandomForest* rf)
{
    for(int i = 0; i < rf->numTrees; i++)
    {
        FreeTree(rf->trees[i].root);
    }

    free(rf->trees);
    free(rf);
}

void ReadCSV(Dataset* data, int fd)
{
    char c;
    int ret = read(fd, &c, sizeof(c));
    int features = 0;
    int bytestodata = 1;

    // READ LINE WITH FEATURE LABELS
    while(c != '\n' && ret > 0)
    {
        if(c == ',') features++;
        ret = read(fd, &c, sizeof(c));
        bytestodata++;
    }

    data->n_features = features;

    ret = read(fd, &c, sizeof(c));
    int rows = 0;

    // READ ALL ROWS
    // !!!!! MAKE SURE THAT THE DATASET HAS AN \n ON THE LAST ROW, ELSE THE CODE WILL CRASH !!!!!!
    while(ret > 0)
    {
        if(c == '\n') rows++;
        ret = read(fd, &c, sizeof(c));
    }

    data->n_samples = rows;
    data->vars = malloc(rows * sizeof(double*));
    data->target_var = malloc(rows * sizeof(int));

    // Use lseek to reposition fd pointer to start of data
    lseek(fd, bytestodata, SEEK_SET);
    ret = read(fd, &c, sizeof(c));

    char buff[32];
    sprintf(buff, "%d", rows);
   //write(0, c, sizeof(c));

    // READ DATA
    int row = 0, feature = 0;
    while(ret > 0)
    {
        data->vars[row] = malloc(features * sizeof(double));
        char val[256];
        double featureval;
        int valindx = 0;
        feature = 0;
        while(c != '\n')
        {
            // READ FEATURES AS DOUBLES
            if(c != ',' && feature != features)
            {
                val[valindx] = c;
                valindx++;
            }
            else if(c == ',' && feature != features)
            {
                double finalval = strtof(val, &val[valindx-1]);
                data->vars[row][feature] = finalval;
                feature++;
                valindx = 0;
            }
            else if(c != ',' && feature == features)
            {
                int target_val = atoi(&c);
                data->target_var[row] = target_val;
            }

            //write(0, &c, sizeof(c));
            ret = read(fd, &c, sizeof(c));
        }

        ret = read(fd, &c, sizeof(c));
        row++;
    }
}

void FreeCSVDataset(Dataset* data)
{
    for(int i = 0; i < data->n_samples; i++)
    {
        free(data->vars[i]);
    }

    free(data->vars);
    free(data->target_var);
}

void SafeWrite(int fd, const void *buf, size_t n_bytes)
{
    if(write(fd, buf, n_bytes) < 0)
    {
        perror("Error saving tree, aborting to avoid corrupted data.\n");
        close(fd);
        unlink("model.rfc");
        exit(1);
    }
}

void SaveTree(Node* node, int fd)
{   
    if(!node) return;

    int featureid = (node->leaf) ? -1 : node->featureId;
    double threshold = (node->leaf) ? node->value : node->threshold;
    SafeWrite(fd, &featureid, sizeof(featureid));
    SafeWrite(fd, &threshold, sizeof(threshold));

    if(!node->leaf)
    {
        SaveTree(node->left, fd);
        SaveTree(node->right, fd);
    }
}

void SaveForest(RandomForest* rf, int fd)
{
    SafeWrite(fd, &rf->numTrees, sizeof(rf->numTrees));
    SafeWrite(fd, &rf->numFeatures, sizeof(rf->numFeatures));
    SafeWrite(fd, &rf->maxDepth, sizeof(rf->maxDepth));

    for(int i = 0; i < rf->numTrees; i++)
    {
        SaveTree(rf->trees[i].root, fd);
    }

    char buff[64];
    sprintf(buff, "Forest saved successfully!\n");
    write(0, buff, strlen(buff));
}

int main()
{
    int fd = open("bankdataset.csv", O_RDONLY);
    if(fd == -1)
    {
        perror("Error opening dataset file.");
        exit(1);
    }

    Dataset* data = malloc(sizeof(Dataset));
    ReadCSV(data, fd);

    int numTrees = 10;
    int numFeatures = data->n_features;
    RandomForest* rf = CreateRandomForest(numTrees, MAX_DEPTH, numFeatures);

    // int train_size = (int)(data->n_samples * 0.8);
    // int test_size = data->n_samples - train_size;
    TrainForest(rf, data);
    Metrics m = EvaluateModel(rf, data);
    PrintMetrics(m);

    // Store tree into file
    int writefile = open("model.rfc", O_RDWR | O_CREAT | O_TRUNC, S_IRUSR);
    SaveForest(rf, writefile);

    FreeCSVDataset(data);
    FreeRandomForest(rf);
    free(data);
}