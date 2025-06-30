
#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include "nn_interfaces.h"
#include <cmath>

namespace utec {
    namespace neural_network {

        // MSE: mide que tan lejos están las predicciones de los valores reales
        template<typename T>
        class MSELoss final : public ILoss<T, 2> {
            Tensor<T,2> pred, real;
        public:
            MSELoss(const Tensor<T,2>& y_pred, const Tensor<T,2>& y_true)
              : pred(y_pred), real(y_true) {}

            T loss() const override {
                T total = 0;
                for (size_t i = 0; i < pred.size(); ++i)
                    total += (pred[i] - real[i]) * (pred[i] - real[i]);
                return total / pred.size();
            }

            Tensor<T,2> loss_gradient() const override {
                Tensor<T,2> grad = pred;
                for (size_t i = 0; i < grad.size(); ++i)
                    grad[i] = 2 * (pred[i] - real[i]) / pred.size();
                return grad;
            }
        };

        // BCE: se usa para problemas de clasificación con salida 0 o 1
        template<typename T>
        class BCELoss final : public ILoss<T, 2> {
            Tensor<T,2> yhat, y;
            static constexpr T epsilon = 1e-7;
        public:
            BCELoss(const Tensor<T,2>& y_pred, const Tensor<T,2>& y_true)
              : yhat(y_pred), y(y_true) {}

            T loss() const override {
                T sum = 0;
                for (size_t i = 0; i < yhat.size(); ++i) {
                    T p = std::clamp(yhat[i], epsilon, T(1) - epsilon);  // evitar log(0)
                    sum += -y[i]*std::log(p) - (1 - y[i])*std::log(1 - p);
                }
                return sum / yhat.size();
            }

            Tensor<T,2> loss_gradient() const override {
                Tensor<T,2> grad = yhat;
                for (size_t i = 0; i < grad.size(); ++i) {
                    T p = std::clamp(yhat[i], epsilon, T(1) - epsilon);
                    grad[i] = (p - y[i]) / (p * (1 - p)) / yhat.size();  // derivada de BCE
                }
                return grad;
            }
        };

    }
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
