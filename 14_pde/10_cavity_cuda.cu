#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>
using namespace std;
typedef vector<vector<float>> matrix;
#define BLOCK_SIZE 16

__global__ void calc_b(
                const int nx,
                const int ny,
                float* b,
                float* u,
                float* v,
                double rho,
                double dx,
                double dy,
                double dt
                ){
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < 1 || j >= ny - 1 || i < 1 || i >= nx - 1) return;
    b[j*nx+i] = rho * (1 / dt *
    ((u[j*nx+(i+1)] - u[j*nx+(i-1)]) / (2 * dx) + (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy)) -
    ((u[j*nx+(i+1)] - u[j*nx+(i-1)]) / (2 * dx))*((u[j*nx+(i+1)] - u[j*nx+(i-1)]) / (2 * dx))
        - 2 * ((u[(j+1)*nx+i] - u[(j-1)*nx+i]) / (2 * dy) *
        (v[j*nx+(i+1)] - v[j*nx+(i-1)]) / (2 * dx)) -
    ((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy))*((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy)));
}

__global__ void calc_p(
                const int nx,
                const int ny,
                float* b,
                float* p,
                float* pn,
                double dx,
                double dy
                ){
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < 1 || j >= ny - 1 || i < 1 || i >= nx - 1) return;
    p[j*nx+i] = (dy*dy * (pn[j*nx+(i+1)] + pn[j*nx+(i-1)]) +
                dx*dx * (pn[(j+1)*nx+i] + pn[(j-1)*nx+i]) -
                b[j*nx+i] * dx*dx * dy*dy) / (2 * (dx*dx + dy*dy));

}

__global__ void calc_uv(
                const int nx,
                const int ny,
                float* p,
                float* u,
                float* un,
                float* v,
                float* vn,
                double rho,
                double dx,
                double dy,
                double dt,
                double nu
                ){
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < 1 || j >= ny - 1 || i < 1 || i >= nx - 1) return;
    u[j*nx+i] = un[j*nx+i] - un[j*nx+i] * dt / dx * (un[j*nx+i] - un[j*nx+(i-1)])
                        - un[j*nx+i] * dt / dy * (un[j*nx+i] - un[(j-1)*nx+i])
                        - dt / (2 * rho * dx) * (p[j*nx+(i+1)] - p[j*nx+(i-1)])
                        + nu * dt / dx*dx * (un[j*nx+(i+1)] - 2 * un[j*nx+i] + un[j*nx+(i-1)])
                        + nu * dt / dy*dy * (un[(j+1)*nx+i] - 2 * un[j*nx+i] + un[(j-1)*nx+i]);
                        
    v[j*nx+i] = vn[j*nx+i] - vn[j*nx+i] * dt / dx * (vn[j*nx+i] - vn[j*nx+(i-1)])
                        - vn[j*nx+i] * dt / dy * (vn[j*nx+i] - vn[(j-1)*nx+i])
                        - dt / (2 * rho * dx) * (p[(j+1)*nx+i] - p[(j-1)*nx+i])
                        + nu * dt / dx*dx * (vn[j*nx+(i+1)] - 2 * vn[j*nx+i] + vn[j*nx+(i-1)])
                        + nu * dt / dy*dy * (vn[(j+1)*nx+i] - 2 * vn[j*nx+i] + vn[(j-1)*nx+i]);

}
__global__ void set_y(const int nx, const int ny, float* u, float* v){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < 1 || j >= ny - 1) return;
    u[j*nx]      = 0;
    u[j*nx+nx-1] = 0;
    v[j*nx]      = 0;
    v[j*nx+nx-1] = 0;
}
__global__ void set_x(const int nx, const int ny, float* u, float* v){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1 || i >= nx - 1) return;
    u[i]           = 0;
    u[(ny-1)*nx+i] = 1;
    v[i]           = 0;
    v[(ny-1)*nx+i] = 0;
}
int main(){
    const int nx = 161;
    const int ny = 161;
    int nt = 10;
    int nit = 50;
    double dx = 2. / (nx - 1);
    double dy = 2. / (ny - 1);
    double dt = 0.001;
    double rho = 1;
    double nu = 0.02;
    size_t xysize = ny*nx*sizeof(float);
    float *u, *v, *p, *b, *pn, *un, *vn;
    const dim3 grid_size((ny + BLOCK_SIZE - 1) / BLOCK_SIZE, ((nx + BLOCK_SIZE - 1) / BLOCK_SIZE));
	const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    cudaMallocManaged(&u, xysize);
    cudaMallocManaged(&v, xysize);
    cudaMallocManaged(&b, xysize);
    cudaMallocManaged(&p, xysize);
    cudaMallocManaged(&un, xysize);
    cudaMallocManaged(&vn, xysize);
    cudaMallocManaged(&pn, xysize);
    cudaMemset(u, 0, xysize);
    cudaMemset(v, 0, xysize);
    cudaMemset(p, 0, xysize);
    cudaMemset(b, 0, xysize);
    for (int n=0; n<nt; n++){
        calc_b<<<grid_size, block_size>>>(nx, ny, b, u, v, rho, dx, dy, dt);
        cudaDeviceSynchronize();

        for (int it=0; it<nit; it++){
            memcpy(pn, p, xysize);
            calc_p<<<grid_size, block_size>>>(nx, ny, b, p, pn, dx, dy);
            cudaDeviceSynchronize();
            for(int j=1; j<ny-1; j++){
                p[j*nx+nx-1] = p[j*nx+nx-2];
                p[j*nx]      = p[j*nx+1];
            }
            for(int i=1; i<nx-1; i++){
                p[i]           = p[nx+i];
                p[(ny-1)*nx+i] = 0;
            }
        }
        memcpy(un, u, xysize);
        memcpy(vn, v, xysize);
        calc_uv<<<grid_size, block_size>>>(nx, ny, p, u, un, v, vn, rho, dx, dy, dt, nu);
        cudaDeviceSynchronize();
        set_y<<<(ny + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(nx, ny, u, v);
        cudaDeviceSynchronize();
        set_x<<<(nx + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(nx, ny, u, v);
        cudaDeviceSynchronize();
        // if(n == 9)
        //     for (int j=0; j<ny; j++){
        //         for (int i=0; i<nx; i++){
        //             if(b[j*nx+i]!=0)
        //             printf("b[%d][%d] = %f %f %f\n", j, i, b[j*nx+i], p[j*nx+i], u[j*nx+i]);
        //         }
        //     }
    }
    
    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);
    cudaFree(pn);
    cudaFree(un);
    cudaFree(vn);
}