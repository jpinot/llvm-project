int a_global = 3;
double b_global = 3.2;
void *c_global = &a_global;
int d_global[2] = {2,2};
double e_global[4][4];
int f_global[2][2][2];
struct g_global_type {
    int g_1;
    double g_2;
    void *g_3;
    int g_4[2];
};

struct g_global_type g_global = {.g_1 = a_global, .g_2 = b_global, .g_3 = c_global, .g_4 = {2,2}};

#define NUM_ITER 2
#define REPETITIONS 10

