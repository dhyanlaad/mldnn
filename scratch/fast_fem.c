
#include <math.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    int thread_id;
    int start_path;
    int end_path;
    int n;
    double alpha;
    double mu;
    double sigma;
    double y0;
    const double* dB;
    const double* wdet;
    const double* ksto;
    double* y_out;
    int n_eval;
    const double* t_eval;
} WorkerArgs;

void* worker_fem(void* ptr) {
    WorkerArgs* args = (WorkerArgs*)ptr;
    int n = args->n;
    double mu = args->mu;
    double sigma = args->sigma;
    double y0 = args->y0;
    int n_eval = args->n_eval;
    
    double* y_hist = (double*)malloc((n + 1) * sizeof(double));
    double* Bh = (double*)malloc(n * sizeof(double));
    double* Sh = (double*)malloc(n * sizeof(double));
    
    for (int p = args->start_path; p < args->end_path; p++) {
        const double* path_dB = args->dB + (size_t)p * n;
        double* path_out = args->y_out + (size_t)p * n_eval;
        
        y_hist[0] = y0;
        
        for (int k = 0; k < n; k++) {
            double y_k = y_hist[k];
            Bh[k] = mu * y_k;
            Sh[k] = sigma * y_k * path_dB[k];
            
            // Vectorized dot product for past memory
            double sum = 0.0;
            const double* wd = args->wdet + k;
            const double* ks = args->ksto + k;
            
            #pragma clang loop vectorize(enable) interleave(enable)
            for (int j = 0; j <= k; j++) {
                sum += Bh[j] * wd[-j] + Sh[j] * ks[-j];
            }
            y_hist[k + 1] = y0 + sum;
        }
        
        // Linear interpolation to t_eval
        double dt = 1.0 / n;
        for (int e = 0; e < n_eval; e++) {
            double te = args->t_eval[e];
            int idx = (int)(te / dt);
            if (idx >= n) idx = n - 1;
            double rem = (te - idx * dt) / dt;
            path_out[e] = y_hist[idx] * (1.0 - rem) + y_hist[idx + 1] * rem;
        }
    }
    
    free(y_hist);
    free(Bh);
    free(Sh);
    return NULL;
}

void solve_fem_c(int n_paths, int n, double alpha, double mu, double sigma, double y0,
                 const double* dB, const double* wdet, const double* ksto,
                 double* y_out, int n_eval, const double* t_eval, int num_threads) {
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    WorkerArgs* args = (WorkerArgs*)malloc(num_threads * sizeof(WorkerArgs));
    
    int paths_per_thread = (n_paths + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        args[t].thread_id = t;
        args[t].start_path = t * paths_per_thread;
        args[t].end_path = (t + 1) * paths_per_thread;
        if (args[t].end_path > n_paths) args[t].end_path = n_paths;
        args[t].n = n;
        args[t].alpha = alpha;
        args[t].mu = mu;
        args[t].sigma = sigma;
        args[t].y0 = y0;
        args[t].dB = dB;
        args[t].wdet = wdet;
        args[t].ksto = ksto;
        args[t].y_out = y_out;
        args[t].n_eval = n_eval;
        args[t].t_eval = t_eval;
        
        pthread_create(&threads[t], NULL, worker_fem, &args[t]);
    }
    
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    
    free(threads);
    free(args);
}
