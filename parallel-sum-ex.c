#define _GNU_SOURCE
#include <sched.h>    // for processor affinity
#include <unistd.h>   // unix standard apis
#include <pthread.h>  // pthread api

#include <stdlib.h>
#include <stdio.h>

int vector_len;       // length of vector we will sum
double* vector;       // the vector we will sum

int num_threads;            // number of threads we will use
int num_processors;         // number of processors we will use
pthread_t* tid;             // array of thread ids
double* partial_sum;        // partial sums, one for each processor


// The following code reads the intel/amd cycle counter
// This is the number of cycles since the machine was started
// Note that with billions of cycles per second, this is the
// most accurate timer on the machine
unsigned long long int rdtsc(void) {
  unsigned int a;
  unsigned int d;

  // This is inline assembler - rdtsc is the specific instruction
  // we are assembling here
  __asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));

  return ((unsigned long long)a) | (((unsigned long long)d) << 32);
}

// Function run by each thread
void* worker(void* arg) {
  long myid     = (long)arg;
  int blocksize = vector_len / num_threads; // note: floor

  // put ourselves on the desired processor
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(myid % num_processors, &set);
  if (sched_setaffinity(0, sizeof(set), &set) < 0) { // do it
    perror("Can't setaffinity"); // hopefully doesn't fail
    exit(-1);
  }

  // This figures out the chunk of the vector I should
  // work on based on my id
  int mystart = myid * blocksize;
  int myend   = 0;
  if (myid == (num_threads - 1)) { // last thread
    // the last thread will take care of the leftover
    // elements of the vector, in case num_threads doesn't
    // divide vector_len
    // WARNING: this is a suboptimal solution. It means that the last thread
    // might do much more work than the other threads (up to almost double)
    // which will slow down the entire job. A better solution would split up
    // remainder work equally between threads...
    myend = vector_len;
  } else {
    myend = (myid + 1) * blocksize;
  }

  // Now I sum that chunk and put the result in partial_sum
  partial_sum[myid] = 0.0;
  for (int i = mystart; i < myend; i++) {
    partial_sum[myid] += vector[i];
  }

  // Done.  The master thread will sum up the partial sums
  pthread_exit(NULL);           // finish - no return value

}


int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage: parallel-sum-ex number-of-threads number-of-procs length-of-vector\n");
    exit(-1);
  }

  num_threads = atoi(argv[1]); // number of threads
  num_processors = atoi(argv[2]); // numer of processors to use
  vector_len  = atoi(argv[3]);    // length of vector to sum

  vector      = (double*)malloc(sizeof(double) * vector_len);
  tid         = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
  partial_sum = (double*)malloc(sizeof(double) * num_threads);

  if (!vector || !tid || !partial_sum) {
    fprintf(stderr, "cannot allocate memory\n");
    exit(-1);
  }

  // populate vector
  for (int i = 0; i < vector_len; i++) {
    vector[i] = (double)i;
  }

  // sequential sum
  unsigned long long start = rdtsc();
  double sequential_sum    = 0.0;
  for (int i = 0; i < vector_len; i++) {
    sequential_sum += vector[i];
  }
  unsigned long long sequential_time = rdtsc() - start;

  start = rdtsc();

  // launch threads
  // thread i will compute partial_sum[i], which will sum
  // vector[i*ceiling(vector_size/num_threads)
  //          through
  //        (i+1)*floor(vector_size/num_threads) ]
  // the last thread will also handle the additional elements
  for (long i = 0; i < num_threads; i++) {
    int returncode = pthread_create(&(tid[i]),  // thread id gets put here
                                    NULL, // use default attributes
                                    worker, // thread will begin in this function
                                    (void*)i // we'll give it i as the argument
                                    );
    if (returncode != 0) {
      perror("Failed to start thread");
      exit(-1);
    }
  }

  // now we will join all the threads
  for (int i = 0; i < num_threads; i++) {
    int returncode = pthread_join(tid[i], NULL);
    if (returncode != 0) {
      perror("join failed");
      exit(-1);
    }
  }

  // Now we know that the partial_sums are done, so we'll add them up
  double parallel_sum = 0.0;
  for (int i = 0; i < num_threads; i++) {
    parallel_sum += partial_sum[i];
  }

  unsigned long long parallel_time = rdtsc() - start;

  printf("Sequential sum:   %lf (%llu cycles)\n"
         "Parallel sum:     %lf (%llu cycles)\n",
         sequential_sum, sequential_time, parallel_sum, parallel_time);

  return 0;
}

