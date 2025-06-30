
#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "nn_interfaces.h"
#include <cmath>

namespace utec {
    namespace neural_network {

        // ReLU: activa valores positivos, cero los negativos
        template<typename T>
        class ReLU final : public ILayer<T> {
            Tensor<T,2> input_record;
        public:
            Tensor<T,2> forward(const Tensor<T,2>& z) override {
                input_record = z;
                auto output = z;
                for (auto& val : output)
                    val = std::max(T(0), val);  // aplica ReLU
                return output;
            }

            Tensor<T,2> backward(const Tensor<T,2>& grad) override {
                Tensor<T,2> result = grad;
                for (size_t i = 0; i < grad.size(); ++i)
                    result[i] = input_record[i] > 0 ? grad[i] : 0;  // derivada de ReLU
                return result;
            }
        };

        // Sigmoid: convierte valores a un rango entre 0 y 1
        template<typename T>
        class Sigmoid final : public ILayer<T> {
            Tensor<T,2> activ;
        public:
            Tensor<T,2> forward(const Tensor<T,2>& z) override {
                activ = z;
                for (auto& v : activ)
                    v = T(1) / (T(1) + std::exp(-v));  // aplica sigmoid
                return activ;
            }

            Tensor<T,2> backward(const Tensor<T,2>& grad) override {
                auto out = grad;
                for (size_t i = 0; i < out.size(); ++i)
                    out[i] = grad[i] * activ[i] * (1 - activ[i]);  // derivada de sigmoid
                return out;
            }
        };

    }
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H