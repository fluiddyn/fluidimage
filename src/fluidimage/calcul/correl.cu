__global__ void cucorrelate(float *g_im0data, float *g_im1data, float *g_odata,
                            unsigned int nx0, unsigned int nx1,
                            unsigned int ny0,
                            unsigned int dispx, unsigned int dispy,
                            unsigned int nymax, unsigned int ny1dep,
                            unsigned int ny0dep, unsigned int nx0dep,
                            unsigned int nxmax, unsigned int nx1dep
) 
{

unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int iy = blockIdx.y*blockDim.y+ threadIdx.y;
unsigned int index = ix+nx0*iy;

if ((ix<nxmax)&&(iy<nymax)){
g_odata[index] = g_im1data[(iy+ny1dep)*nx1+(ix+nx1dep)]*g_im0data[(ny0dep+iy)*nx0+nx0dep+ix];
//g_odata[ix+nxmax*iy] = g_im1data[iy+ny1dep, ix+nx1dep]*g_im0data[ny0dep+iy, nx0dep+ix];
}
else if ((ix<nx0)&&(iy<ny0))
g_odata[index] = 0;
}