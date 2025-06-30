
#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#include "nn_interfaces.h"
#include <vector>

namespace utec {
    namespace neural_network {

        // SGD: actualiza los valores poco a poco usando el gradiente
        template<typename T>
        class SGD final : public IOptimizer<T> {
            T lr;
        public:
            explicit SGD(T learning_rate = 0.01) : lr(learning_rate) {}

            void update(Tensor<T,2>& param, const Tensor<T,2>& grad) override {
                for (size_t i = 0; i < param.size(); ++i)
                    param[i] -= lr * grad[i];
            }
        };

        // Adam: optimizador que ajusta el paso automaticamente
        template<typename T>
        class Adam final : public IOptimizer<T> {
            T lr, beta1, beta2, eps;
            std::vector<T> m, v;
            size_t step_count = 0;
        public:
            Adam(T lr_=0.001, T b1=0.9, T b2=0.999, T e=1e-8)
              : lr(lr_), beta1(b1), beta2(b2), eps(e) {}

            void update(Tensor<T,2>& param, const Tensor<T,2>& grad) override {
                if (m.size() != param.size()) {
                    m.assign(param.size(), 0);
                    v.assign(param.size(), 0);
                    step_count = 0;
                }
                ++step_count;
                for (size_t i = 0; i < param.size(); ++i) {
                    m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
                    v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
                    T m_hat = m[i] / (1 - std::pow(beta1, step_count));
                    T v_hat = v[i] / (1 - std::pow(beta2, step_count));
                    param[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                }
            }
            void step() override {}
        };

    }
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H