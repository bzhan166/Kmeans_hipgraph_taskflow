#include <hip/hip_runtime.h>

#include<iostream>
#include <iomanip>
#include <cfloat>
#include <climits>
#include <vector>

using namespace std;

#define L2(x1, y1, x2, y2)((x1 - x2)*(x1 -x2) + (y1 - y2)*(y1 - y2))

// ----------------------------------------------------------------------------
// CPU (sequential) implementation
// ----------------------------------------------------------------------------
// run k-means on cpu
std::pair<std::vector<float>, std::vector<float>> cpu_seq(
    const int N, const int K, const int M,
    const std::vector<float>& px,
    const std::vector<float>& py)
{
    // clare some vectors
    std::vector<int> c(K);  // vector c: K elements (K groups)
    std::vector<float> sx(K), sy(K), mx(K), my(K);

    // initial centroids for each cluster/group
    for (int i = 0; i < K; i++){
        mx[i] = px[i];
        my[i] = py[i];
    }

    // loop for all iterations
    for (int iter = 0; iter < M; iter++){

        // clear the statistics
        for (int k = 0; k < K; k++){
            sx[k] = 0.0f;
            sy[k] = 0.0f;
            c[k]  = 0;
        }

        // find the best cluster-id for each points
        // loop: check all points, calculate the distance
        for (int i = 0; i < N; i++){
            float x = px[i];
            float y = py[i];
            float best_distance = std::numeric_limits<float>::max();    // just to assign a big value
            int best_k = 0;

            for (int k = 0; k < K; k++){
                const float d = L2(x, y, mx[k], my[k]);
                if (d < best_distance){
                    best_distance = d;
                    best_k = k;
                }
            }
        
            // gather all points belong to a cluster
            sx[best_k] += x;
            sy[best_k] += y;
            c[best_k] += 1;
        }

        // update the centroids
        for (int k = 0; k < K; k++){
            const int count = max(1, c[k]);
            mx[k] = sx[k] / count;
            my[k] = sy[k] / count;
        }
    }

    return {mx, my};
}


/* Each point (thread) computes its distance to each centroid 
and adds its x and y values to the sum of its closest
centroid, as well as incrementing that centroid's count of assigned points. */
__global__ void assign_clusters(
    const float *px, const float *py,
    int N, const float *mx, const float *my,
    float *sx, float *sy, int k, int *c)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N){
        return;
    }

    // make global loads
    const float x = px[index];
    const float y = py[index];

    float best_distance = FLT_MAX;
    int best_cluster = 0;
    for (int c = 0; c < k; ++c){
        const float distance = L2(x, y, mx[c], my[c]);
        if (distance < best_distance){
            best_distance = distance;
            best_cluster = c;
        }
    }

    // update clusters/assign clusters to compute the centroids
    atomicAdd(&sx[best_cluster], x);
    atomicAdd(&sy[best_cluster], y);
    atomicAdd(&c[best_cluster], 1);
}

/* Each thread is one cluster, which just recomputes its coordinates as the mean
 of all points assigned to it. */
/* Each thread is one cluster, which just recomputes its coordinates as the mean
 of all points assigned to it. */
__global__ void compute_new_means(
    float *mx, float *my,
    const float *sx, const float *sy, const int *c)
{
    const int cluster = threadIdx.x;
    const int count = max(1, c[cluster]);
    mx[cluster] = sx[cluster] / count;
    my[cluster] = sy[cluster] / count;
}

std::pair<std::vector<float>, std::vector<float>> gpu_normal_kernal(
    int N, int K, int M,
    const std::vector<float> &h_px,
    const std::vector<float> &h_py)
{
    std::vector<float> h_mx, h_my;  // contains the returned centroids
    // mx, my for keeping centroids/cluster per iter
    float *d_px, *d_py, *d_mx, *d_my, *d_sx, *d_sy;
    int *d_c;   
    // copy values of all points to host_mx, _my
    for (int i = 0; i < K; i++){
        h_mx.push_back(h_px[i]);
        h_my.push_back(h_py[i]);
    }

    //allocate mem
    hipMalloc((void**)&d_px, N*sizeof(float));
    hipMalloc((void**)&d_py, N*sizeof(float));
    hipMalloc((void**)&d_mx, K*sizeof(float));
    hipMalloc((void**)&d_my, K*sizeof(float));
    hipMalloc((void**)&d_sx, K*sizeof(float));
    hipMalloc((void**)&d_sy, K*sizeof(float));
    hipMalloc((void**)&d_c, K*sizeof(float));

    //copy data from host to device
    hipMemcpy(d_px, h_px.data(), N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_py, h_py.data(), N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_mx, h_mx.data(), K*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_my, h_my.data(), K*sizeof(float), hipMemcpyHostToDevice);

    //init data on GPU
    hipMemset(d_c, 0,  K*sizeof(float));
    hipMemset(d_sx, 0,  K*sizeof(float));
    hipMemset(d_sy, 0,  K*sizeof(float));
    
    
    assign_clusters<<<dim3((N+1024-1)/1024, 1, 1), dim3(1024, 1, 1)>>> (d_px, d_py, N, d_mx, d_my, d_sx, d_sy, K, d_c);

    compute_new_means<<<dim3(1, 1, 1), dim3(K, 1, 1)>>> (d_mx, d_my, d_sx, d_sy, d_c);
    
    /*
    //For Graph
    hipStream_t stream;
    hipGraph_t graph;
    hipGraphNode_t graph_cluster, graph_means;
    hipKernelNodeParams cluster_Params = { 0 };
    hipKernelNodeParams compute_means_Params = { 0 };

    hipGraphCreate(&graph, 0);
    hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);  
    
    //add cluster node parameters
    hipMemset(&cluster_Params, 0, sizeof(cluster_Params));
    cluster_Params.func = (void*)assign_clusters;
    cluster_Params.gridDim = dim3((N+1024-1)/1024, 1, 1);
    cluster_Params.blockDim = dim3(1024, 1, 1);
    cluster_Params.sharedMemBytes = 0;
    void* cluster_Args[9] = { (void*)&d_px, (void*)&d_py, &N, (void*)&d_mx, 
    (void*)&d_my, (void*)&d_sx, (void*)&d_sy, &K, (void *)&d_c };
    cluster_Params.kernelParams = cluster_Args;
    cluster_Params.extra = NULL;
    
    //compute new means parameters
    hipMemset(&compute_means_Params, 0, sizeof(compute_means_Params));
    compute_means_Params.func = (void*)compute_new_means;
    compute_means_Params.gridDim = dim3(1, 1, 1);
    compute_means_Params.blockDim = dim3(K, 1, 1);
    compute_means_Params.sharedMemBytes = 0;
    void* compute_Args[5] = { (void*)&d_mx, (void*)&d_my, (void*)&d_sx, (void*)&d_sy,
    (void*)&d_c};
    compute_means_Params.kernelParams = compute_Args;
    compute_means_Params.extra = NULL;

    hipGraphAddKernelNode(&graph_cluster, graph, NULL, 0, &cluster_Params);
    hipGraphAddKernelNode(&graph_means, graph, NULL, 0, &compute_means_Params);
  
    hipGraphAddDependencies(graph, &graph_cluster, &graph_means, 1);

    hipGraphExec_t graphExec;
    hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
  
    hipGraphLaunch(graphExec, stream);
    hipStreamSynchronize(stream);

    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);
    */

    hipFree(d_px);
    hipFree(d_py);
    hipFree(d_mx);
    hipFree(d_my);
    hipFree(d_sx);
    hipFree(d_sy);
    hipFree(d_c);

    return {h_mx, h_my};
}

std::pair<std::vector<float>, std::vector<float>> gpu_cond_tasks(
    int N, int K, int M,
    const std::vector<float> &h_px,
    const std::vector<float> &h_py)
{
    std::vector<float> h_mx, h_my;  // contains the returned centroids
    // mx, my for keeping centroids/cluster per iter
    float *d_px, *d_py, *d_mx, *d_my, *d_sx, *d_sy;
    int *d_c;   
    // copy values of all points to host_mx, _my
    for (int i = 0; i < K; i++){
        h_mx.push_back(h_px[i]);
        h_my.push_back(h_py[i]);
    }

    //allocate mem
    hipMalloc((void**)&d_px, N*sizeof(float));
    hipMalloc((void**)&d_py, N*sizeof(float));
    hipMalloc((void**)&d_mx, K*sizeof(float));
    hipMalloc((void**)&d_my, K*sizeof(float));
    hipMalloc((void**)&d_sx, K*sizeof(float));
    hipMalloc((void**)&d_sy, K*sizeof(float));
    hipMalloc((void**)&d_c, K*sizeof(float));

    //copy data from host to device
    hipMemcpy(d_px, h_px.data(), N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_py, h_py.data(), N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_mx, h_mx.data(), K*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_my, h_my.data(), K*sizeof(float), hipMemcpyHostToDevice);

    //init data on GPU
    hipMemset(d_c, 0,  K*sizeof(float));
    hipMemset(d_sx, 0,  K*sizeof(float));
    hipMemset(d_sy, 0,  K*sizeof(float));
    
    /*
    assign_clusters<<<dim3((N+1024-1)/1024, 1, 1), dim3(1024, 1, 1)>>> (d_px, d_py, N, d_mx, d_my, d_sx, d_sy, K, d_c);

    compute_new_means<<<dim3(1, 1, 1), dim3(K, 1, 1)>>> (d_mx, d_my, d_sx, d_sy, d_c);
    */
    
    //For Graph
    hipStream_t stream;
    hipGraph_t graph;
    hipGraphNode_t graph_cluster, graph_means;
    hipKernelNodeParams cluster_Params = { 0 };
    hipKernelNodeParams compute_means_Params = { 0 };

    hipGraphCreate(&graph, 0);
    hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);  
    
    //add cluster node parameters
    hipMemset(&cluster_Params, 0, sizeof(cluster_Params));
    cluster_Params.func = (void*)assign_clusters;
    cluster_Params.gridDim = dim3((N+1024-1)/1024, 1, 1);
    cluster_Params.blockDim = dim3(1024, 1, 1);
    cluster_Params.sharedMemBytes = 0;
    void* cluster_Args[9] = { (void*)&d_px, (void*)&d_py, &N, (void*)&d_mx, 
    (void*)&d_my, (void*)&d_sx, (void*)&d_sy, &K, (void *)&d_c };
    cluster_Params.kernelParams = cluster_Args;
    cluster_Params.extra = NULL;
    
    //compute new means parameters
    hipMemset(&compute_means_Params, 0, sizeof(compute_means_Params));
    compute_means_Params.func = (void*)compute_new_means;
    compute_means_Params.gridDim = dim3(1, 1, 1);
    compute_means_Params.blockDim = dim3(K, 1, 1);
    compute_means_Params.sharedMemBytes = 0;
    void* compute_Args[5] = { (void*)&d_mx, (void*)&d_my, (void*)&d_sx, (void*)&d_sy,
    (void*)&d_c};
    compute_means_Params.kernelParams = compute_Args;
    compute_means_Params.extra = NULL;

    hipGraphAddKernelNode(&graph_cluster, graph, NULL, 0, &cluster_Params);
    hipGraphAddKernelNode(&graph_means, graph, NULL, 0, &compute_means_Params);
  
    hipGraphAddDependencies(graph, &graph_cluster, &graph_means, 1);

    hipGraphExec_t graphExec;
    hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
  
    hipGraphLaunch(graphExec, stream);
    hipStreamSynchronize(stream);

    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);
    

    hipFree(d_px);
    hipFree(d_py);
    hipFree(d_mx);
    hipFree(d_my);
    hipFree(d_sx);
    hipFree(d_sy);
    hipFree(d_c);

    return {h_mx, h_my};
}


int main(int argc, const char *argv[])
{
    // check num of arguments
    if(argc != 4) {
        std::cout << "usage: ./kmean_cudaflow num_points k num_iterations\n";
        std::exit(EXIT_FAILURE);
    }

    // get args
    const int N = std::atoi(argv[1]);
    const int K = std::atoi(argv[2]);
    const int M = std::atoi(argv[3]);

    // conditions for each arguments
    if(N < 1) {
        throw std::runtime_error("num_points must be at least one");
    }
    
    if(K >= N) {
        throw std::runtime_error("k must be smaller than the number of points");
    }
    
    if(M < 1) {
        throw std::runtime_error("num_iterations must be larger than 0");
    }


    // declare arrays
    std::vector<float> h_px, h_py, mx, my;

    // Randomly generate N points
    std::cout << "generating " << N << " random points ...\n";
    for(int i=0; i<N; ++i) {
        h_px.push_back(rand()%1000 - 500);
        h_py.push_back(rand()%1000 - 500);
    }


    // ----------------- k-means on cpu_seq
    std::cout << "running k-means on cpu (sequential) ... ";
    // start_time
    auto sbeg = std::chrono::steady_clock::now();
    // call cpu_kmean_kernel: std::tie is to create a tuple of values
    std::tie(mx, my) = cpu_seq(N, K, M, h_px, h_py);
    // end_time
    auto send = std::chrono::steady_clock::now();
    // show results
    std::cout << "completed with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(send-sbeg).count()
            << " ms\n";
    std::cout << "k centroids found by cpu (sequential)\n";
    for(int k = 0; k < K; ++k) {
        std::cout << "centroid " << k << ": " << std::setw(10) << mx[k] << ' ' 
                                            << std::setw(10) << my[k] << '\n';
    }

        // ----------------- k-means on gpu with normal kernals
    std::cout << "running k-means on GPU (with normal kernal) ...";
    auto pgpu_kernal_beg_time = std::chrono::steady_clock::now();
    std::tie(mx, my) = gpu_normal_kernal(N, K, M, h_px, h_py);
    auto pgpu_kernal_end_time = std::chrono::steady_clock::now();
    std::cout << "completed with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(pgpu_kernal_end_time-pgpu_kernal_beg_time).count()
            << " ms\n";
	std::cout << "k centroids found by gpu (normal kernal)\n";
    for(int k = 0; k < K; ++k) {
        std::cout << "centroid " << k << ": " << std::setw(10) << mx[k] << ' ' 
                                            << std::setw(10) << my[k] << '\n';
    }


    // ----------------- k-means on gpu with graph
    std::cout << "running k-means on GPU (with graph) ...";
    auto pgpu_con_beg_time = std::chrono::steady_clock::now();
    std::tie(mx, my) = gpu_cond_tasks(N, K, M, h_px, h_py);
    auto pgpu_con_end_time = std::chrono::steady_clock::now();
    std::cout << "completed with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(pgpu_con_end_time-pgpu_con_beg_time).count()
            << " ms\n";
	std::cout << "k centroids found by gpu (graph)\n";
    for(int k = 0; k < K; ++k) {
        std::cout << "centroid " << k << ": " << std::setw(10) << mx[k] << ' ' 
                                            << std::setw(10) << my[k] << '\n';
    }
	

    h_px.erase (h_px.begin(), h_px.end());
    h_py.erase (h_py.begin(), h_py.end()); 

    return 0;
}
