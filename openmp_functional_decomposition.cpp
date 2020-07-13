#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdio.h>
#include <fstream>

const float GRAIN_GROWS_PER_MONTH = 9.0;
const float ONE_DEER_EATS_PER_MONTH = 1.0;

const float AVG_PRECIP_PER_MONTH = 7.0;
const float AMP_PRECIP_PER_MONTH = 6.0;
const float RANDOM_PRECIP = 2.0;

const float AVG_TEMP = 60.0;
const float AMP_TEMP = 20.0;
const float RANDOM_TEMP = 10.0;

const float MIDTEMP = 40.0;
const float MIDPRECIP = 10.0;

int NowYear = 2020;
int NowMonth = 0;
int NowNumDeer = 1;
int NowNumWolves = 1;

float NowPrecip;
float NowTemp;
float NowHeight = 1.0;

unsigned int seed = 0;

std::ofstream myFile;

void Graindeer();
void Grain();
void Wolves();
void Watcher();
float SQR(float);
float Ranf(unsigned int*, float, float);


int main( int argc, char *argv[ ] ) {
    #ifndef _OPENMP
        fprintf( stderr, "No OpenMP support!\n" );
        return 1;
    #endif
    
    myFile.open("results.txt");
    
    //Initialize NowTemp and NowPrecip
    float ang = (30.0 * (float)NowMonth + 15.0) * (M_PI / 180.0);
    
    float temp = AVG_TEMP - AMP_TEMP * cos(ang);
    NowTemp = temp + Ranf(&seed, -RANDOM_TEMP, RANDOM_TEMP);
    
    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
    NowPrecip = precip + Ranf(&seed, -RANDOM_PRECIP, RANDOM_PRECIP);
    if(NowPrecip < 0.0)
        NowPrecip = 0.0;
    
    omp_set_num_threads( 4 );
    #pragma omp parallel sections shared(NowYear, NowMonth, NowNumDeer, NowPrecip, NowTemp, NowHeight)
    {
        #pragma omp section
        {
            Graindeer();
        }
        
        #pragma omp section
        {
            Grain();
        }
        
        #pragma omp section
        {
            Wolves();
        }
        
        #pragma omp section
        {
            Watcher();
        }

    }
    
    myFile.close();
    return 0;
}

void Graindeer() {

    while(NowYear < 2026) {
        
        //Compute temp next value(of deer)based on current state of sim
        int tempNumDeer = NowNumDeer;
        if((float)NowNumDeer > NowHeight)
            tempNumDeer -= 1;
        else if((float)NowNumDeer < NowHeight)
            tempNumDeer += 1;
        
        if((float)tempNumDeer / (float)NowNumWolves < 2)
            tempNumDeer--;
        
        if(tempNumDeer < 0)
            tempNumDeer = 0;
            
        //DoneComputing barrier:
        #pragma omp barrier
        
        //Copy Next variables into now variables
        NowNumDeer = tempNumDeer;
        
        //DoneAssigning barrier:
        #pragma omp barrier
        
        //DonePrinting barrier:
        #pragma omp barrier
    }
}

void Grain() {

    while(NowYear < 2026) {
        //Compute temp next value(of based on current state of sim
        float tempHeight = NowHeight;
        
        float tempFactor = exp( -SQR( (NowTemp - MIDTEMP) / 10.0) );
        float precipFactor = exp( -SQR( (NowPrecip - MIDPRECIP) / 10.0) );
        
        tempHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        tempHeight -= float(NowNumDeer * ONE_DEER_EATS_PER_MONTH);
        if(tempHeight < 0.0)
            tempHeight = 0.0;

        //DoneComputing barrier:
        #pragma omp barrier
        
        //Copy Next variables into now variables
        NowHeight = tempHeight;
        
        //DoneAssigning barrier:
        #pragma omp barrier
        
        //DonePrinting barrier:
        #pragma omp barrier
    }
}

void Wolves() {

    while(NowYear < 2026) {
        //Compute temp next value(of based on current state of sim
        int tempNumWolves = NowNumWolves;
        if((float)NowNumDeer / (float)tempNumWolves < 2.0)
            tempNumWolves--;
        else if((float)NowNumDeer / (float)tempNumWolves > 2.0)
            tempNumWolves++;
        
        if(tempNumWolves < 0)
            tempNumWolves = 0;
        //DoneComputing barrier:
        #pragma omp barrier
        
        //Copy Next variables into now variables
        NowNumWolves = tempNumWolves;
        
        //DoneAssigning barrier:
        #pragma omp barrier
        
        //DonePrinting barrier:
        #pragma omp barrier
    }
}

void Watcher() {
 
     while(NowYear < 2026) {
        //DoneComputing barrier:
        #pragma omp barrier
        
        
        //DoneAssigning barrier:
        #pragma omp barrier
        
        //Print Current set of global state variables.
        printf("Results for %d - %d\n", NowMonth + 1, NowYear);
        printf("Current Temp: %f\n", NowTemp);
        printf("Current Precipitation: %f\n", NowPrecip);
        printf("Current Grain Height %f\n", NowHeight);
        printf("Current Number of Deer: %d\n", NowNumDeer);
        printf("Current Number of Wolves: %d\n\n", NowNumWolves);
        
        float tempC = (5.0/9.0) * (NowTemp - 32.0);
        
        myFile << (NowMonth + 1) << "/" << NowYear << "\t" << tempC << "\t" << NowPrecip << "\t" << NowHeight << "\t" << NowNumDeer << "\t" << NowNumWolves << "\n";
        
        //Increment month
        NowMonth++;
        if(NowMonth == 12) {
            NowMonth = 0;
            NowYear++;
        }
        
        //Use new month to compute the new temp and precip
        float ang = (30.0 * (float)NowMonth + 15.0) * (M_PI / 180.0);
        
        float temp = AVG_TEMP - AMP_TEMP * cos(ang);
        float randomFloat = Ranf(&seed, -RANDOM_TEMP, RANDOM_TEMP);
        NowTemp = temp + randomFloat;
        
        float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
        NowPrecip = precip + Ranf(&seed, -RANDOM_PRECIP, RANDOM_PRECIP);
        if(NowPrecip < 0.0)
            NowPrecip = 0.0;
        
        //DonePrinting barrier:
        #pragma omp barrier
    }
}

float SQR(float x) {

    return x*x;
}

float Ranf( unsigned int *seedp,  float low, float high )
{
        float r = (float) rand_r( seedp );              // 0 - RAND_MAX

        return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}
