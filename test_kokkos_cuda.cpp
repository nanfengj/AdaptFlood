#include <Kokkos_Core.hpp>
#include <cstdio>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        printf("Kokkos execution space: %s\n", typeid(Kokkos::DefaultExecutionSpace).name());
        
        int N = 1000;
        Kokkos::View<double*> a("a", N);
        
        // 并行初始化
        Kokkos::parallel_for("init", N, KOKKOS_LAMBDA(int i) {
            a(i) = i * 2.0;
        });
        
        // 并行规约求和
        double sum = 0;
        Kokkos::parallel_reduce("sum", N, KOKKOS_LAMBDA(int i, double& lsum) {
            lsum += a(i);
        }, sum);
        
        printf("Sum = %f (expected: %f)\n", sum, (double)(N-1)*N);
        
        if(abs(sum - (double)(N-1)*N) < 0.001) {
            printf("Kokkos test PASSED!\n");
        } else {
            printf("Kokkos test FAILED!\n");
        }
    }
    Kokkos::finalize();
    return 0;
}
