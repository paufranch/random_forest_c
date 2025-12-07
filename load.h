#ifndef LOAD_H
#define LOAD_H

void SafeRead(int fd, void *buff, size_t n_bytes);
void LoadTree(Node* node, int fd);
void LoadForest(RandomForest* rf, const char* filename);

#endif