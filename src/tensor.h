
#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <numeric>
#include <iostream>

namespace utec {
    namespace algebra {

    template <typename T, size_t tam>
    class Tensor {

    private:
        std::array<size_t, tam> forma;
        std::vector<T> datos;

        template <typename... Idxs>
        size_t calculate_index(Idxs... idxs) const {
            std::array<size_t, tam> indices{static_cast<size_t>(idxs)...};
            for (size_t i = 0; i < tam; ++i) {
                if (indices[i] >= forma[i]) {
                    throw std::out_of_range("Index out of range");}}
            size_t indice = 0;
            size_t paso = 1;
            for (int i = tam - 1; i >= 0; --i) {
                indice += indices[i] * paso;
                paso *= forma[i];}
            return indice;}

        void multiply_with_broadcasting(Tensor& resultado, const Tensor& a, const Tensor& b,
                                        std::array<size_t, tam>& indices, size_t dim = 0) const {
            if (dim == tam) {
                std::array<size_t, tam> a_indices, b_indices;
                for (size_t i = 0; i < tam; ++i) {
                    a_indices[i] = (a.forma[i] == 1) ? 0 : indices[i];
                    b_indices[i] = (b.forma[i] == 1) ? 0 : indices[i];}
                resultado(indices) = a(a_indices) * b(b_indices);
                return;}
            for (size_t i = 0; i < resultado.shape()[dim]; ++i) {
                indices[dim] = i;
                multiply_with_broadcasting(resultado, a, b, indices, dim + 1);}}

        template <typename... Dims>
        static std::array<size_t, tam> make_dims_array(Dims... dims) {
            if (sizeof...(Dims) != tam) {
                throw std::invalid_argument("Number of dimensions must match tensor rank");}
            return std::array<size_t, tam>{static_cast<size_t>(dims)...};}

    public:
        size_t size() const { return datos.size(); }

        T& operator[](size_t idx) { return datos[idx]; }
        const T& operator[](size_t idx) const { return datos[idx]; }

        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;

        iterator begin() noexcept { return datos.begin(); }
        iterator end() noexcept { return datos.end(); }
        const_iterator begin() const noexcept { return datos.begin(); }
        const_iterator end() const noexcept { return datos.end(); }
        const_iterator cbegin() const noexcept { return datos.cbegin(); }
        const_iterator cend() const noexcept { return datos.cend(); }

        Tensor() : forma{}, datos{} {}

        Tensor(const std::array<size_t, tam>& forma) : forma(forma) {
            size_t tam_total = 1;
            for (size_t dim : forma) tam_total *= dim;
            datos.resize(tam_total, T{});}

        template <typename... Dims>
        Tensor(Dims... dims) {
            if (sizeof...(Dims) != tam) {
                throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(tam));
            }
            std::array<size_t, tam> temp;
            size_t i = 0;
            ((temp[i++] = static_cast<size_t>(dims)), ...);
            forma = temp;
            size_t total = 1;
            for (auto d : forma) total *= d;
            datos.resize(total, T{});}

        Tensor& operator=(std::initializer_list<T> lista) {
            if (lista.size() != datos.size()) {
                throw std::invalid_argument("Data size does not match tensor size");}
            std::copy(lista.begin(), lista.end(), datos.begin());
            return *this;}

        template <typename... Idxs>
        T& operator()(Idxs... idxs) {
            return datos[calculate_index(idxs...)];}

        template <typename... Idxs>
        const T& operator()(Idxs... idxs) const {
            return datos[calculate_index(idxs...)];}

        T& operator()(const std::array<size_t, tam>& idxs) {
            size_t indice = 0;
            size_t paso = 1;
            for (int i = tam - 1; i >= 0; --i) {
                if (idxs[i] >= forma[i]) {
                    throw std::out_of_range("Index out of range");}
                indice += idxs[i] * paso;
                paso *= forma[i];}
            return datos[indice];}

        const T& operator()(const std::array<size_t, tam>& idxs) const {
            size_t indice = 0;
            size_t paso = 1;
            for (int i = tam - 1; i >= 0; --i) {
                if (idxs[i] >= forma[i]) {
                    throw std::out_of_range("Index out of range");}
                indice += idxs[i] * paso;
                paso *= forma[i];}
            return datos[indice];}

        const std::array<size_t, tam>& shape() const noexcept {
            return forma;}

        void reshape(const std::array<size_t, tam>& nueva_forma) {
            size_t total_nuevo = std::accumulate(nueva_forma.begin(), nueva_forma.end(), 1ul, std::multiplies<>());
            size_t total_actual = datos.size();
            if (total_nuevo != total_actual) {
                if (total_nuevo > total_actual) {
                    datos.resize(total_nuevo, T{});}
                forma = nueva_forma;}
            else {
                forma = nueva_forma;}}

        template <typename... Dims>
        void reshape(Dims... dims) {
            if (sizeof...(Dims) != tam) {
                throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(tam));}
            std::array<size_t, tam> temp;
            size_t i = 0;
            ((temp[i++] = static_cast<size_t>(dims)), ...);
            reshape(temp);}

        void fill(const T& valor) {
            std::fill(datos.begin(), datos.end(), valor);}

        Tensor operator+(const Tensor& otro) const {
            std::array<size_t, tam> array_resultado;
            for (size_t i = 0; i < tam; ++i) {
                if (forma[i] != otro.forma[i] && forma[i] != 1 && otro.forma[i] != 1) {
                    throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");}
                array_resultado[i] = std::max(forma[i], otro.forma[i]);}
            Tensor resultado(array_resultado);
            std::array<size_t, tam> indices{};
            auto rec = [&](auto&& self, size_t dim = 0) -> void {
                if (dim == tam) {
                    std::array<size_t, tam> a_idx, b_idx;
                    for (size_t i = 0; i < tam; ++i) {
                        a_idx[i] = (forma[i] == 1) ? 0 : indices[i];
                        b_idx[i] = (otro.forma[i] == 1) ? 0 : indices[i];}
                    resultado(indices) = (*this)(a_idx) + otro(b_idx);
                    return;}
                for (size_t i = 0; i < array_resultado[dim]; ++i) {
                    indices[dim] = i;
                    self(self, dim + 1);}};
            rec(rec);
            return resultado;}

        Tensor operator-(const Tensor& otro) const {
            std::array<size_t, tam> array_resultado;
            for (size_t i = 0; i < tam; ++i) {
                if (forma[i] != otro.forma[i] && forma[i] != 1 && otro.forma[i] != 1) {
                    throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");}
                array_resultado[i] = std::max(forma[i], otro.forma[i]);}
            Tensor resultado(array_resultado);
            std::array<size_t, tam> indices{};
            auto rec = [&](auto&& self, size_t dim = 0) -> void {
                if (dim == tam) {
                    std::array<size_t, tam> a_idx, b_idx;
                    for (size_t i = 0; i < tam; ++i) {
                        a_idx[i] = (forma[i] == 1) ? 0 : indices[i];
                        b_idx[i] = (otro.forma[i] == 1) ? 0 : indices[i];}
                    resultado(indices) = (*this)(a_idx) - otro(b_idx);
                    return;}
                for (size_t i = 0; i < array_resultado[dim]; ++i) {
                    indices[dim] = i;
                    self(self, dim + 1);}};
            rec(rec);
            return resultado;}

        Tensor operator*(const Tensor& otro) const {
            std::array<size_t, tam> forma_res;
            for (size_t i = 0; i < tam; ++i) {
                if (forma[i] != otro.forma[i] && forma[i] != 1 && otro.forma[i] != 1) {
                    throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");}
                forma_res[i] = std::max(forma[i], otro.forma[i]);}

            Tensor resultado(forma_res);
            std::array<size_t, tam> indices{};

            auto rec = [&](auto&& self, size_t dim = 0) -> void {
                if (dim == tam) {
                    std::array<size_t, tam> a_idx, b_idx;
                    for (size_t i = 0; i < tam; ++i) {
                        a_idx[i] = (forma[i] == 1) ? 0 : indices[i];
                        b_idx[i] = (otro.forma[i] == 1) ? 0 : indices[i];}
                    resultado(indices) = (*this)(a_idx) * otro(b_idx);
                    return;}
                for (size_t i = 0; i < forma_res[dim]; ++i) {
                    indices[dim] = i;
                    self(self, dim + 1);}};

            rec(rec);
            return resultado;}

        Tensor operator+(const T& escalar) const {
            Tensor resultado(forma);
            for (size_t i = 0; i < datos.size(); ++i) {
                resultado.datos[i] = datos[i] + escalar;}
            return resultado;}

        Tensor operator-(const T& escalar) const {
            Tensor resultado(forma);
            for (size_t i = 0; i < datos.size(); ++i) {
                resultado.datos[i] = datos[i] - escalar;}
            return resultado;}

        Tensor operator*(const T& escalar) const {
            Tensor resultado(forma);
            for (size_t i = 0; i < datos.size(); ++i) {
                resultado.datos[i] = datos[i] * escalar;}
            return resultado;}

        Tensor operator/(const T& escalar) const {
            Tensor resultado(forma);
            for (size_t i = 0; i < datos.size(); ++i) {
                resultado.datos[i] = datos[i] / escalar;}
            return resultado;}

        friend Tensor operator+(const T& esc, const Tensor& t) {
            return t + esc;}

        friend Tensor operator-(const T& esc, const Tensor& t) {
            Tensor resultado(t.forma);
            for (size_t i = 0; i < t.datos.size(); ++i) {
                resultado.datos[i] = esc - t.datos[i];}
            return resultado;}

        friend Tensor operator*(const T& esc, const Tensor& t) {
            return t * esc;}

        friend Tensor operator/(const T& esc, const Tensor& t) {
            Tensor resultado(t.forma);
            for (size_t i = 0; i < t.datos.size(); ++i) {
                resultado.datos[i] = esc / t.datos[i];}
            return resultado;}

        Tensor transpose_2d() const {
            if constexpr (tam < 2) {
                throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");}
            std::array<size_t, tam> nueva_forma = forma;
            std::swap(nueva_forma[tam - 1], nueva_forma[tam - 2]);
            Tensor resultado(nueva_forma);

            std::array<size_t, tam> idx;
            auto rec = [&](auto&& mismo, size_t dim) -> void {
                if (dim == tam) {
                    std::array<size_t, tam> new_idx = idx;
                    std::swap(new_idx[tam - 1], new_idx[tam - 2]);
                    resultado(new_idx) = (*this)(idx);
                    return;}
                for (size_t i = 0; i < forma[dim]; ++i) {
                    idx[dim] = i;
                    mismo(mismo, dim + 1);}};
            rec(rec, 0);
            return resultado;}

        friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
            const auto& shape = t.shape();

            if constexpr (tam == 4) {
                os << "{" << std::endl;
                for (size_t i = 0; i < shape[0]; ++i) {
                    os << "{" << std::endl;
                    for (size_t j = 0; j < shape[1]; ++j) {
                        os << "{" << std::endl;
                        for (size_t k = 0; k < shape[2]; ++k) {
                            for (size_t l = 0; l < shape[3]; ++l) {
                                if (l > 0) os << " ";
                                os << t(i, j, k, l);}
                            os << std::endl;}
                        os << "}" << std::endl;}
                    os << "}" << std::endl;}
                os << "}" << std::endl;
                return os;}

            else if constexpr (tam >= 2) {
                size_t out = 1;
                for (size_t i = 0; i < tam - 2; ++i) out *= shape[i];
                size_t fils = shape[tam - 2];
                size_t cols = shape[tam - 1];
                os << "{" << std::endl;
                for (size_t i = 0; i < out; ++i) {
                    if (out > 1) os << "{" << std::endl;

                    for (size_t r = 0; r < fils; ++r) {
                        for (size_t c = 0; c < cols; ++c) {
                            std::array<size_t, tam> idx;
                            size_t tmp = i;
                            for (int d = static_cast<int>(tam) - 3; d >= 0; --d) {
                                size_t dim = shape[d];
                                idx[d] = tmp % dim;
                                tmp /= dim;}

                            idx[tam - 2] = r;
                            idx[tam - 1] = c;
                            size_t idx_now = 0, paso = 1;
                            for (int j = tam - 1; j >= 0; --j) {
                                idx_now += idx[j] * paso;
                                paso *= shape[j];}

                            if (c > 0) os << " ";
                            if (idx_now < t.datos.size())
                                os << t(idx);
                            else
                                os << 0;}
                        os << std::endl;}

                    if (out > 1) os << "}" << std::endl;}
                os << "}" << std::endl;}

            else {
                for (size_t i = 0; i < shape[0]; ++i) {
                    if (i < t.datos.size())
                        os << t.datos[i];
                    else
                        os << 0;
                    if (i + 1 < shape[0]) os << " ";}}
            return os;}
    };

    template <typename T, size_t tam>
    Tensor<T, tam> transpose_2d(const Tensor<T, tam>& t) {
        if constexpr (tam < 2) {
            throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");}
        return t.transpose_2d();}

    template <typename T, size_t tam>
    Tensor<T, tam> matrix_product(const Tensor<T, tam>& A, const Tensor<T, tam>& B) {
        if constexpr (tam < 2) throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
        if (A.shape()[tam - 1] != B.shape()[tam - 2])
            throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
        for (size_t i = 0; i < tam - 2; ++i) {
            if (A.shape()[i] != B.shape()[i])
                throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");}
        std::array<size_t, tam> forma_resultante = A.shape();
        forma_resultante[tam - 1] = B.shape()[tam - 1];
        Tensor<T, tam> resultado(forma_resultante);
        std::array<size_t, tam> idx;
        auto rec = [&](auto&& self, size_t d) -> void {
            if (d == tam - 2) {
                for (size_t i = 0; i < A.shape()[tam - 2]; ++i) {
                    for (size_t j = 0; j < B.shape()[tam - 1]; ++j) {
                        T sum = T{};
                        for (size_t k = 0; k < A.shape()[tam - 1]; ++k) {
                            auto a_idx = idx;
                            a_idx[tam - 2] = i;
                            a_idx[tam - 1] = k;
                            auto b_idx = idx;
                            b_idx[tam - 2] = k;
                            b_idx[tam - 1] = j;
                            sum += A(a_idx) * B(b_idx);}
                        auto r_idx = idx;
                        r_idx[tam - 2] = i;
                        r_idx[tam - 1] = j;
                        resultado(r_idx) = sum;}}
                return;}
            for (size_t i = 0; i < A.shape()[d]; ++i) {
                idx[d] = i;
                self(self, d + 1);}};
        rec(rec, 0);
        return resultado;}

    template <typename T, size_t tam, typename F>
    Tensor<T, tam> apply(const Tensor<T, tam>& t, F&& func) {
        Tensor<T, tam> resultado(t.shape());
        for (size_t i = 0; i < t.size(); ++i) {
            resultado[i] = func(t[i]);
        }
        return resultado;
    }

}}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H