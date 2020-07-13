#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <xmmintrin.h>
using std::ofstream;
using std::cout;
using std::endl;

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES    10
#endif

#define SSE_WIDTH           4

const int ARRAY_SIZE = 8000000;
int numT = 2;
int elementsPerCore = ARRAY_SIZE / numT;
const float MIN = 1.0;
const float MAX = 100.0;

float Ranf( float, float );
int Ranf( int, int );
void TimeOfDaySeed( );
float SimdMulSum(float*, float*, int );
float NonSimdMulSum(float*, float*, int);

int main( int argc, char *argv[ ] ) {
    #ifndef _OPENMP
        fprintf( stderr, "No OpenMP support!\n" );
        return 1;
    #endif
    
    ofstream myFile;
    ofstream coreFile;
    myFile.open("results2.txt", std::ios::app);
    coreFile.open("core.txt", std::ios::app);
    
    float *a = new float[ARRAY_SIZE];
    float *b = new float[ARRAY_SIZE];
    TimeOfDaySeed();
    
    for(int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = Ranf(MIN, MAX);
        b[i] = Ranf(MIN, MAX);
    }
    
    float maxPerformanceSimd = 0;
    float maxPerformanceNonSimd = 0.0;
    float maxPerformanceTwoCore = 0.0;
    float maxPerformanceFourCore = 0.0;
    float result;
    double time0, time1;
    
    //SIMD
    for(int i = 0; i < NUMTRIES; i++) {
        time0 = omp_get_wtime( );
        result = SimdMulSum(a, b, ARRAY_SIZE);
        time1 = omp_get_wtime( );
        double MegaMultsPerSecond = (double)(ARRAY_SIZE) / ( time1 - time0 ) / 1000000.;
        if( MegaMultsPerSecond > maxPerformanceSimd )
                maxPerformanceSimd = MegaMultsPerSecond;
    }
    printf("Simd Result: %f\n", result);
    printf("Simd Max Performance: %f\n\n", maxPerformanceSimd);
    
    //NON-SIMD
    for(int i = 0; i < NUMTRIES; i++) {
        time0 = omp_get_wtime( );
         result = NonSimdMulSum(a, b, ARRAY_SIZE);
        time1 = omp_get_wtime( );
        double MegaMultsPerSecond = (double)(ARRAY_SIZE) / ( time1 - time0 ) / 1000000.;
        if( MegaMultsPerSecond > maxPerformanceNonSimd )
                maxPerformanceNonSimd = MegaMultsPerSecond;
    }
    printf("NonSimd Result: %f\n", result);
    printf("NonSimd Max Performance: %f\n", maxPerformanceNonSimd);
    float speedupSimd = maxPerformanceSimd / maxPerformanceNonSimd;
    printf("Speedup SIMD: %f\n\n", speedupSimd);
    
    //SIMD 2-CORE
    omp_set_num_threads(numT);
    for(int i = 0; i < NUMTRIES; i++) {
        result = 0.0;
        time0 = omp_get_wtime();
        #pragma omp parallel reduction(+: result)
        {
            int first = omp_get_thread_num() * elementsPerCore;
            float sum = SimdMulSum(&a[first], &b[first], elementsPerCore);
            result = result + sum;
        }
        time1 = omp_get_wtime();
        
        double MegaMultsPerSecond = (double)(ARRAY_SIZE) / ( time1 - time0 ) / 1000000.;
        if( MegaMultsPerSecond > maxPerformanceTwoCore)
                maxPerformanceTwoCore = MegaMultsPerSecond;
    }
    printf("SIMD 2-Core Result: %f\n", result);
    printf("SIMD 2-Core Max Performance: %f\n", maxPerformanceTwoCore);
    float speedupTwoCore = maxPerformanceTwoCore / maxPerformanceNonSimd;
    printf("Speedup 2-Core SIMD: %f\n\n", speedupTwoCore);
    
    
    //SIMD 4-CORE
    numT = 4;
    elementsPerCore = ARRAY_SIZE / numT;
    omp_set_num_threads(numT);
    for(int i = 0; i < NUMTRIES; i++) {
        result = 0.0;
        time0 = omp_get_wtime();
        #pragma omp parallel reduction(+: result)
        {
            int first = omp_get_thread_num() * elementsPerCore;
            float sum = SimdMulSum(&a[first], &b[first], elementsPerCore);
            result = result + sum;
        }
        time1 = omp_get_wtime();
        
        double MegaMultsPerSecond = (double)(ARRAY_SIZE) / ( time1 - time0 ) / 1000000.;
        if( MegaMultsPerSecond > maxPerformanceFourCore )
                maxPerformanceFourCore = MegaMultsPerSecond;
    }
    printf("SIMD 4-Core Result: %f\n", result);
    printf("SIMD 4-Core Max Performance: %f\n", maxPerformanceFourCore);
    float speedupFourCore = maxPerformanceFourCore / maxPerformanceNonSimd;
    printf("Speedup 4-Core SIMD: %f\n", speedupFourCore);
    
    
    //coreFile << ARRAY_SIZE << "\t" << speedupSimd << "\t" << speedupTwoCore << "\t" << speedupFourCore << "\n";
    //myFile << ARRAY_SIZE << "\t" << maxPerformanceSimd << "\t" << maxPerformanceNonSimd << "\t" << speedupSimd << "\n";
    
    myFile.close();
    coreFile.close();
    
    delete[] a;
    delete[] b;
    return 0;
}

float SimdMulSum( float *a, float *b, int len )
{
    float sum[4] = { 0., 0., 0., 0. };
    int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
    register float *pa = a;
    register float *pb = b;

    __m128 ss = _mm_loadu_ps( &sum[0] );
    for( int i = 0; i < limit; i += SSE_WIDTH )
    {
        ss = _mm_add_ps( ss, _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
        pa += SSE_WIDTH;
        pb += SSE_WIDTH;
    }
    _mm_storeu_ps( &sum[0], ss );

    for( int i = limit; i < len; i++ )
    {
        sum[0] += a[i] * b[i];
    }

    return sum[0] + sum[1] + sum[2] + sum[3];
}

float NonSimdMulSum(float *a, float *b, int len) {
    float sum = 0.0;
    for(int i = 0; i < len; i++) {
        sum = sum + (a[i] * b[i]);
    }
    return sum;
}

float Ranf( float low, float high )
{
        float r = (float) rand();               // 0 - RAND_MAX
        float t = r  /  (float) RAND_MAX;       // 0. - 1.

        return   low  +  t * ( high - low );
}

int Ranf( int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = ceil( (float)ihigh );

        return (int) Ranf(low,high);
}

void TimeOfDaySeed( )
{
    struct tm y2k = { 0 };
    y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
    y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

    time_t  timer;
    time( &timer );
    double seconds = difftime( timer, mktime(&y2k) );
    unsigned int seed = (unsigned int)( 1000.*seconds );    // milliseconds
    srand( seed );
}

