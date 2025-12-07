#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include "random_forest.h"
#include "load.h"

void SafeRead(int fd, void *buff, size_t n_bytes)
{
    if(read(fd, buff, n_bytes) < 0)
    {
        perror("Error reading file: ");
        exit(1);
    }
}

void LoadTree(Node* node, int fd)
{
    int featureId = 0;
    double threshold = 0.0;
    SafeRead(fd, &featureId, sizeof(featureId));
    SafeRead(fd, &threshold, sizeof(threshold));

    node = malloc(sizeof(Node));
    node->leaf = (featureId == -1) ? 1 : 0;

    if(node->leaf)
    {
        node->value = threshold;
        node->left = node->right = NULL;
    }
    else
    {
        node->featureId = featureId;
        node->threshold = threshold;
        LoadTree(node->left, fd);
        LoadTree(node->right, fd);
    }
}

void LoadForest(RandomForest* rf, const char* filename)
{
    int fd = open(filename, O_RDONLY);
    if(fd == -1)
    {
        perror("Error loading file.\n");
        exit(1);
    }

    SafeRead(fd, &rf->numTrees, sizeof(rf->numTrees));
    SafeRead(fd, &rf->numFeatures, sizeof(rf->numFeatures));
    SafeRead(fd, &rf->maxDepth, sizeof(rf->maxDepth));

    rf->trees = malloc(rf->numTrees * sizeof(DecisionTree));

    for(int i = 0; i < rf->numTrees; i++)
    {
        LoadTree(rf->trees[i].root, fd);
    }
}