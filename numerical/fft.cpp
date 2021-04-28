#include <complex>
#include <tuple>
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>

const float pi = 3.141592;

std::vector<float> freq_domain(int n, int d) {
    std::vector<float> fd;
    for(int i = 0; i < floor((n-1)/2)+1; i++) fd.push_back(i/(d*n));
    for(int i = -floor(n/2)+1; i < 0; i++) fd.push_back(i/(d*n));
    return fd;
}

int main () {
    std::ifstream infile("points.csv");
    std::vector<std::tuple<float, float, float>> point_list;

    float t, theta, theta_t;
    char c;

    while((infile >> t >> c >> theta >> c >> theta_t) && (c == ',')) {
        std::cout << 
        point_list.push_back(std::make_tuple(t, theta, theta_t));
    };

    std::ofstream fout("fft.csv");
    std::size_t n = point_list.size();
    std::vector<float> fd = freq_domain(n, 0.01);
    for(std::size_t i = 0; i < n; i++) {
        float sum = 0;
        for(std::size_t j = 0; j < n; j++) {
            std::tie(t, theta, theta_t) = point_list[j];
            sum += theta * cos( 2.0 * pi * float(i * j) / float(n) );
        }
        fout << fd[i] << "," << sum << "\n";
    }
    fout.close();
    return 0;
}