#include <iostream>

typedef unsigned long ulint_t;

static const size_t N = 624;
static const size_t M = 397;
static const ulint_t LCG_C = 69069UL;
static const ulint_t MATRIX_A = 0x9908b0dfUL;
static const ulint_t UPPER_MASK = 0x80000000UL;
static const ulint_t LOWER_MASK = 0x7fffffffUL;
static const ulint_t mag01[2] = {0x0, MATRIX_A};

/*
  Reference Matusmoto--Nishimura 1997
  Appendix C
  doi:10.1145/272991.272995
*/
class MT19937 {
  public:
    ulint_t state[N];
    size_t pos;

    MT19937(ulint_t seed) {
      seedGenerator(seed);
    }

    /*
      Initial: use seed to gen N nums using
      a linear congruential generator
    */
    void seedGenerator(ulint_t seed) {
      state[0] = seed & 0xffffffffUL;
      for(int i = 1; i <= N; i++){
          state[i] = (LCG_C * state[i - 1]) & 0xffffffffUL;
      };
      pos = N;
    }

    /*
      Generate in batches of size N, keep track
      using `pos`
    */
    void genNewBatch() {
      ulint_t y;

      // Reuse old array to store new
      // Move x_k <- x_n+k
      for(int k = 0; k < N - M; k++) {
        y = (state[k] & UPPER_MASK) | (state[k + 1] & LOWER_MASK);
        state[k] = state[k + M] ^ (y >> 1) ^ mag01[y & 0x1];
      }

      // Once k >= N-M then recurrence needs newly
      // generated values: x_(N+k) which is
      // in index k now, so re-index by -N.
      for(int k = N - M; k < N - 1; k++) {
        y = (state[k] & UPPER_MASK) | (state[k + 1] & LOWER_MASK);
        state[k] = state[k + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1];
      }
      // Wrap around N index to 0.
      y = (state[N - 1] & UPPER_MASK) | (state[0] & LOWER_MASK);
      state[N - 1] = state[M - 1] ^ (y >> 1) ^ mag01[y & 0x1];
      pos = 0;
    }

    double gen() {
      ulint_t y;

      if(pos >= N) genNewBatch();

      y = state[pos++];
      y ^= (y >> 11);
      y ^= (y << 7) & 0x9d2c5680UL;
      y ^= (y << 15) & 0xefc60000UL;
      y ^= (y >> 18);

      return ( (double)y / (ulint_t)0xffffffffUL );
    }
};

int main () {
  MT19937 mt = MT19937(123456UL);
  for (int i = 0; i < 1000; i++) {
    std::cout << mt.gen() << std::endl;
  }
  return 0;
}
