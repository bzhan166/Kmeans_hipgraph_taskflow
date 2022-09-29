#include <hip/hip_runtime.h>

#include <iomanip>
#include <cfloat>
#include <climits>
#include <vector>

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

std::pair<std::vector<float>, std::vector<float>> gpu_cond_tasks(
    const int N, const int K, const int M,
    const std::vector<float> &h_px,
    const std::vector<float> &h_py)
{
    std::vector<float> h_mx, h_my;  // contains the returned centroids
    float *d_px, *d_py, *d_mx, *d_my, *d_sx, *d_sy, *d_c;   // mx, my for keeping centroids/cluster per iter

    // copy values of all points to host_mx, _my
    for (int i = 0; i < K; i++){
        h_mx.push_back(h_px[i]);
        h_my.push_back(h_py[i]);
    }

    //allocate mem
    HIP_CHECK(hipMalloc(&d_px, N*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_py, N*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_mx, K*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_my, K*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_sx, K*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_sy, K*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_c, K*sizeof(float)));

    //copy data from host to device
    HIP_CHECK(hipmemcpy(d_px, h_px.data(), N*sizeof(float), hipMemcpyHostToDevice));
}


int main(int argc, const char *argv[])
{
    // check num of arguments
    if(argc != 4) {
        std::cerr << "usage: ./kmean_cudaflow num_points k num_iterations\n";
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

    // ----------------- k-means on gpu with conditional tasking
    std::cout << "running k-means on GPU (with conditional tasking) ...";
    auto pgpu_con_beg_time = std::chrono::steady_clock::now();
    std::tie(mx, my) = gpu_cond_tasks(N, K, M, h_px, h_py);
    auto pgpu_con_end_time = std::chrono::steady_clock::now();
    std::cout << "completed with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(pgpu_con_end_time-pgpu_con_beg_time).count()
            << " ms\n";
	std::cout << "k centroids found by cpu (sequential)\n";
    for(int k = 0; k < K; ++k) {
        std::cout << "centroid " << k << ": " << std::setw(10) << mx[k] << ' ' 
                                            << std::setw(10) << my[k] << '\n';
    }
	

    return 0;
}
