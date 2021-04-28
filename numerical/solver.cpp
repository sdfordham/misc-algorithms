#include <iostream>
#include <fstream>
#include <forward_list>
#include <math.h> 

const float pi = 3.141592;

typedef struct {
    float time;
    float theta;
    float theta_t;
} Point;
 
class ODE {
    public:
        virtual float at(float t, float theta, float theta_t) = 0;
};

class DampedDrivenPendulum : public ODE {
    public:
        float period;
        float damping;
        float driving_str;
        float driving_freq;

        DampedDrivenPendulum(float p, float d, float d_s, float d_f) 
            :period{p}, damping{d}, driving_str{d_s}, driving_freq{d_f} {} 

        float at(float t, float theta, float theta_t) {
            float calc = driving_str * cos(driving_freq * t);
            calc -= period * period * sin(theta);
            calc -= damping * theta_t;
            return calc;
        };
};

class RungeKutta4 {
    public:
        float run_time;
        float step_size;
        float omega_0;
        float theta_0;
        const float a[4][4] =     // Butcher tableau
        {
            { 0, 0, 0, 0 },
            { 0.5, 0, 0, 0 },
            { 0, 0.5, 0, 0 },
            { 0, 0, 1, 0 }
        };
        const float b[4] = { 0.1667, 0.3334, 0.3334, 0.1667 };
        const float c[4] = { 0, 0.5, 0.5, 1 };

    RungeKutta4(float r_t, float s_s, float o_0, float t_0)
        :run_time{r_t}, step_size{s_s}, omega_0{o_0}, theta_0{t_0} {}

    void run(ODE& f, std::ofstream& fout) {
        float k[4] = { 0, 0, 0, 0 };
        float l[4] = { 0, 0, 0, 0 };
        float omega = omega_0;
        float theta = theta_0;
        float theta_arg = theta;
        float omega_arg = omega;

        for(float t = 0.0; t < run_time; t += step_size) {
            for(int i = 0; i <= 3; i++) {
                theta_arg = theta;
                omega_arg = omega;
                for(int j = 0; j <= 3; j++) {
                    theta_arg += (k[j] * a[i][j]) * step_size;
                    omega_arg += (l[j] * a[i][j]) * step_size;
                }
                k[i] = omega_arg;
                l[i] = f.at(t + c[i] * step_size, theta_arg, omega_arg);
                theta += (k[i] * b[i]) * step_size;
                omega += (l[i] * b[i]) * step_size;
            }
            while (theta > pi) theta -= 2.0 * pi; 
            while (theta < -pi) theta += 2.0 * pi;
            if(t > 20.0) {
                fout << t << "," << omega << "," << theta << "\n";
            }
        }
    };
};

int main() {
    DampedDrivenPendulum f(1.0, 0.5, 1.0, 0.667);
    RungeKutta4 rk4(100.0, 0.01, 0.0, 1.0);

    // std::forward_list<Point> point_list;
    std::ofstream fout("points.csv");
    rk4.run(f, fout);
    fout.close();

    return 0;
};
