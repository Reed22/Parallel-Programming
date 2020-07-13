#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
using std::ofstream;
using std::cout;
using std::endl;

// setting the number of threads:
#ifndef NUMT
#define NUMT        4
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES    10
#endif


const float XMIN =    -1.0;
const float XMAX =     1.0;
const float YMIN =     0.0;
const float YMAX =     2.0;
const float N =        2.0;
const int NUMNODES =    1000;

float Height(int,int);

int main( int argc, char *argv[ ] ) {
    #ifndef _OPENMP
        fprintf( stderr, "No OpenMP support!\n" );
        return 1;
    #endif
    
    ofstream myFile;
    myFile.open("results.txt", std::ios::app);
    
    omp_set_num_threads( NUMT );    // set the number of threads to use in the for-loop:`
    
    float maxPerformance = 0.;
    float fullTileArea = (((XMAX - XMIN)/(float)(NUMNODES-1))  * ((YMAX - YMIN)/(float)(NUMNODES-1)));
    float volume = 0;

    for(int j = 0; j < NUMTRIES; j++) {
        volume = 0;
        double time0 = omp_get_wtime( );
        #pragma omp parallel for default(none) shared(fullTileArea) reduction(+:volume)
        for(int i = 0; i < NUMNODES*NUMNODES; i++) {
            int iu = i % NUMNODES;
            int iv = i / NUMNODES;
            float z = Height(iu,iv);
            
            float tileArea = fullTileArea;
            
            //if these conditions are met, then the node is AT LEAST an edge (maybe a corner)
            if(iv == 0 || iv == 3 || iu == 0 || iu == 3) {
                //If any of these pairs of conditions are met, then it is an edge.
                if((iv == 0 && iu == 0) || (iv == 3 && iu == 3) || (iv == 0 && iu == 3) || (iv == 3 && iu == 0))
                    tileArea = tileArea / 4.0;
                else
                    tileArea = tileArea / 2.0;
            }
            
            volume = volume + (tileArea * z);
        }
        double time1 = omp_get_wtime( );
        double megaHeightsPerSecond = (double)(NUMNODES*NUMNODES) / ( time1 - time0 ) / 1000000.;
        if( megaHeightsPerSecond > maxPerformance )
            maxPerformance = megaHeightsPerSecond;
        printf("MH Try %d: %f", (j+1), megaHeightsPerSecond);
        printf("Volume Try %d: %f \n",(j+1), volume);

    }
    printf("Max Performance: %f\n", maxPerformance);
    printf("Volume: %f \n", volume);
    myFile << NUMT << "\t" << NUMNODES << "\t" << volume << "\t" << maxPerformance << "\n";

    
    myFile.close();
    return 0;
}

float Height(int iu, int iv) {
    float x = -1.0 + 2.0 * (float)iu / (float)(NUMNODES - 1);
    float y = -1.0 + 2.0 * float(iv) / (float)(NUMNODES - 1);
    
    float xn = pow(fabs(x), (double)N);
    float yn = pow(fabs(y), (double)N);
    float r = 1.0 - xn - yn;
    
    if(r < 0)
        return 0.0;
    
    float height = pow(1.0 - xn - yn, 1.0/(float)N);
    
    //multiply by 2 to account for bottom half.
    return 2 * height;
}
