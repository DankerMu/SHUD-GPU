#include <time.h>
#include <unistd.h>
int main(){
struct timespec spec;
clock_gettime(CLOCK_MONOTONIC_RAW, &spec);
clock_getres(CLOCK_MONOTONIC_RAW, &spec);
return(0);
}
