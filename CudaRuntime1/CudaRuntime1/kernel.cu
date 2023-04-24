#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <ctime>
#include <cstdio>
#include <random>
#define TTMATH_NOASM
#include "ttmath.h"
using namespace std;
typedef ttmath::Big<TTMATH_BITS(64), TTMATH_BITS(256)> lf;
typedef double ld;
typedef long long ll;
#define rnd(mt) ((mt=mt*1487267231ull+1499940803ull)%4999)
lf sqrt(lf x) {
	lf low = 0, high = 1; if (x > 1)high = x;
	for (int i = 0; i < 256; i++) {
		lf mid = (low + high) / 2;
		if (mid * mid <= x)
			low = mid;
		else
			high = mid;
	}
	return low;
}
#define AP 1887602657ull
#define P 1989944597ull
cudaError_t addWithCuda(int *crr,int num);
__global__ void addKernel(int *crr,int num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	crr+= tid*10; crr[0] = 0;
	unsigned long long mt = AP+num*P+tid;
	for (ll iiii= 0;iiii<500000; iiii++) {
		#define r rnd(mt)
		ll Ax = 1000000-r, Ay= r-1000000, Az= r-r;
		ll Bx = r-r,By=1000000-r,Bz=r-1000000;
		ll Hx=Ay*Bz-By*Az,Hy=Az*Bx-Bz*Ax,Hz=Ax*By-Bx*Ay;
		#undef r
		
		ll Gx=Hx-Hy,Gy=Hy-Hz,Gz=Hz-Hx;
		ll Xa=1,Xb=-1,Xc=0; //<- little
		ll Ya=0,Yb=1,Yc=-1; //<- little
		ll Za=-1,Zb=0,Zc=1; //<- about 1e6
		if(abs(Gz)>1e3)
			continue;
		if(!Gx||!Gz)
			continue;
		ll Iz=970000;ll J=Iz*Gz;
		ll Ix=-J/Gx;J+=Ix*Gx;
		ll Iy=0;
		Iz-=J/Gz;J=Iz*Gz+Ix*Gx;
		if(J<0)J=-J;
		//if(J==0||J>100)
		//	continue;
		if(crr[0]&&crr[0]<=J)
			continue;
		crr[0] = J;
		crr[1] = Ax; crr[2] = Ay; crr[3] = Az;
		crr[4] = Bx; crr[5] = By; crr[6] = Bz;
		crr[7] = Ix-Iz; crr[8] = Iy-Ix; crr[9] = Iz-Iy;
	}
}

int main() 
{
	ld maxk = 0;
	ld mink = 1;
	int mtmp[10] = { 0 };
	int kkn = 0;
	static int crr[16384*10];
	for(int iiii=0;;iiii++){
		if (iiii % 30 == 0) {
			system("cls");
			cout << iiii << ':' << maxk << '/' << kkn << '=' << maxk / kkn << ';'<<mink << endl;
			for (int i = 0; i < 10; i++)
				cout << mtmp[i] << ' ';
			cout << endl;
			ld ex = iiii * maxk / kkn / 1.5e-19;
			cout << "remnant expectation: " << ex <<endl;
			cout << "property: " << pow(0.5, iiii / ex) << endl;
		}
		cudaError_t cudaStatus = addWithCuda(crr,iiii);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
		cout<<"---";
		for(int iii=0;iii<16384;iii++)
			if(crr[iii*10]){
				ld k; {
					ll a=crr[iii * 10+1],b=crr[iii * 10+2],c=crr[iii * 10+3];
					ll d=crr[iii * 10+4],e=crr[iii * 10+5],f=crr[iii * 10+6];
					ll g=crr[iii * 10+7],h=crr[iii * 10+8],i=crr[iii * 10+9];
					lf AA = a * a + b * b + c * c; AA = sqrt(AA);
					lf BB = d * d + e * e + f * f; BB = sqrt(BB);
					lf CC = g * g + h * h + i * i; CC = sqrt(CC);
					lf ax = lf(a) / AA, ay = lf(b) / AA, az = lf(c) / AA;
					lf bx = lf(d) / BB, by = lf(e) / BB, bz = lf(f) / BB;
					lf cx = lf(g) / CC, cy = lf(h) / CC, cz = lf(i) / CC;
					ax -= cx; ay -= cy; az -= cz;
					bx -= cx; by -= cy; bz -= cz;
					lf dx = ay * bz - az * by;
					lf dy = az * bx - ax * bz;
					lf dz = ax * by - ay * bx;
					lf DD = sqrt(dx * dx + dy * dy + dz * dz);
					dx /= DD; dy /= DD; dz /= DD;
					k = abs((dx * cx + dy * cy + dz * cz).ToDouble());
				}
				cout << "!" << k; 
				if (k > 1e-30) {
					maxk = max(k, maxk);
					if (k < mink) {
						mink = k;
						memcpy(mtmp, crr + iii * 10, sizeof mtmp);
					}
					mink = min(k, mink);
					kkn++;
					if (k < 1.5e-19) {
						cout << endl;
						for (int j = 0; j < 10; j++)
							cout << crr[iii * 10 + j] << ' ';
						cout << endl;
						return 0;
					}
				}
			}
		cout << endl;
	}
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, int num)
{
    int *dev_c = 0;
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_c, 16384*10*sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    addKernel<<<128,128>>>(dev_c,num);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    cudaStatus = cudaMemcpy(c, dev_c, 16384 * 10 * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
Error:
    cudaFree(dev_c);
    return cudaStatus;
}
