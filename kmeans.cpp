#include <hip/hip_runtime.h>

#include <iomanip>
#include <cfloat>
#include <climits>
#include <vector>

#define L2(x1, y1, x2, y2)((x1 - x2)*(x1 -x2) + (y1 - y2)*(y1 - y2))

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
	// contains the returned centroids
	std::vector<float> h_mx, h_my;  

	// mx, my for keeping centroids/cluster per iter
    float *d_px, *d_py, *d_mx, *d_my, *d_sx, *d_sy, *d_c;

	//allocating mem on GPU devices
	hipMalloc(&d_px, N*sizeof(float));
	hipMalloc(&d_py, N*sizeof(float));
	hipMalloc(&d_mx, N*sizeof(float));
	hipMalloc(&d_my, N*sizeof(float));
	hipMalloc(&d_sx, N*sizeof(float));
	hipMalloc(&d_sy, N*sizeof(float));
	hipMalloc(&d_c, N*sizeof(float));
	
	hipMemset(d_sx, 0, N*sizeof(float));
	hipMemset(d_sy, 0, N*sizeof(float));
	hipMemset(d_c, 0, N*sizeof(float));



    // copy values of all points to host_mx, _my
    for (int i = 0; i < K; i++){
        h_mx.push_back(h_px[i]);
        h_my.push_back(h_py[i]);
    }

	hipFree(d_px);
	hipFree(d_py);
	hipFree(d_mx);
	hipFree(d_my);
	hipFree(d_sx);
	hipFree(d_sy);
	hipFree(d_c);
	
	return 0;

}

// for try

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
