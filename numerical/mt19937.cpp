#include <iostream>

typedef unsigned long ulint_t;

static const size_t N = 624;
static const size_t M = 397;
static const ulint_t LCG_C = 69069UL;
static const ulint_t MATRIX_A = 0x9908b0dfUL;
static const ulint_t UPPER_MASK = 0x80000000UL;
static const ulint_t LOWER_MASK = 0x7fffffffUL;

class MT19937 {
    public:
        ulint_t state[N];
        size_t pos;

        MT19937(ulint_t seed) {
            sgenrand(seed);
        }

        void sgenrand(ulint_t seed) {
            state[0] = seed & 0xffffffffUL;
            for(int i = 1; i <= N; i++){
                state[i] = (LCG_C * state[i - 1]) & 0xffffffffUL;
            };
            pos = N;
        }

        double genrand() {
            ulint_t y;
            static ulint_t mag01[2] = {0x0, MATRIX_A};

            if(pos >= N) {
                for(int i = 0; i < N - M; i++) {
                    y = (state[i] & UPPER_MASK) | (state[i + 1] & LOWER_MASK);
                    state[i] = state[i + M] ^ (y >> 1) ^ mag01[y & 0x1];
                }
                for(int i = 0; i < N - 1; i++) {
                    y = (state[i] & UPPER_MASK) | (state[i + 1] & LOWER_MASK);
                    state[i] = state[i + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1];
                }
                y = (state[N - 1] & UPPER_MASK) | (state[0] & LOWER_MASK);
                state[N - 1] = state[M - 1] ^ (y >> 1) ^ mag01[y & 0x1];
                pos = 0;
            }

            y = state[pos++];
            y ^= (y >> 11);
            y ^= (y << 7) & 0x9d2c5680UL;
            y ^= (y << 15) & 0xefc60000UL;
            y ^= (y >> 18);

            return ( (double)y / (ulint_t)0xffffffff );
        }
};

int main () {
    MT19937 mt = MT19937(123456);
    for (int i = 0; i < 10; i++) {
        std::cout << mt.genrand() << std::endl;
    }
    return 0;
}
