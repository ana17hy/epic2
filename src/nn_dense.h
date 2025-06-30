
#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "nn_interfaces.h"

namespace utec {
    namespace neural_network {

        // Capa Dense
        template<typename T>
        class Dense final : public ILayer<T> {
            Tensor<T,2> W, b, input_cache, dW, db;
        public:
            // Inicializa pesos y sesgos
            template<typename InitW, typename InitB>
            Dense(size_t in, size_t out, InitW iw, InitB ib)
              : W(in, out), b(1, out) {
                iw(W);
                ib(b);
            }

            // Propagación hacia adelante
            Tensor<T,2> forward(const Tensor<T,2>& x) override {
                input_cache = x;
                auto out = matrix_product(x, W);
                for (size_t i = 0; i < out.shape()[0]; ++i)
                    for (size_t j = 0; j < out.shape()[1]; ++j)
                        out(i, j) += b(0, j);
                return out;
            }

            // Retropropagación
            Tensor<T,2> backward(const Tensor<T,2>& grad_out) override {
                dW = matrix_product(transpose_2d(input_cache), grad_out);
                db = sum_rows(grad_out);
                return matrix_product(grad_out, transpose_2d(W));
            }

            // Actualización de parámetros
            void update_params(IOptimizer<T>& opt) override {
                opt.update(W, dW);
                opt.update(b, db);
            }

        private:
            // Suma por filas para calcular gradiente del sesgo
            Tensor<T,2> sum_rows(const Tensor<T,2>& t) {
                Tensor<T,2> result(1, t.shape()[1]);
                for (size_t j = 0; j < t.shape()[1]; ++j) {
                    T acum = 0;
                    for (size_t i = 0; i < t.shape()[0]; ++i)
                        acum += t(i, j);
                    result(0, j) = acum;
                }
                return result;
            }
        };

    }
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H