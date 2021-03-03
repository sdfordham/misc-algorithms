#include <complex>
#include <tuple>
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>

const float pi = 3.141592;

int main () {
    std::ifstream infile("points.csv");
    std::vector<std::tuple<float, float, float>> point_list;

    float t, theta, theta_t;
    char c;

    while((infile >> t >> c >> theta >> c >> theta_t) && (c == ',')) {
        point_list.push_back(std::make_tuple(t, theta, theta_t));
    };

    std::vector<float> fft;
    std::size_t n = point_list.size();
    for(std::size_t i = 0; i < 5; i++) {
        float sum = 0;
        for(std::size_t j = 0; j < 2; j++) {
            std::tie(t, theta, theta_t) = point_list[j];
            std::cout << i << " " << j << " " << n << "\n";
            std::cout << float(i * j) / float(n) << "\n";
            std::cout << theta << "\n\n";
            sum += theta * cos( 2.0 * pi * float(i * j) / float(n) );
        }
        fft.push_back(sum); 
    } 
    
    std::ofstream fout("fft.csv");
    for(std::size_t i = 0; i < 5; i++) {
        std::cout << i << "," << fft[i] << "\n";
    }
    fout.close();
    return 0;
}