#pragma once
#include <vector>
#include <mkl.h>


template<typename T>
class Allocator
{
public:
    Allocator() = default;
    virtual ~Allocator() = default;

    virtual T* allocate(size_t size) {
        return new T[size];
    }
    virtual void deallocate(T* p) {
        if (p)
            delete p;
    }

};

template<class T>
class MklAllocator : public Allocator<T>
{
public:
    MklAllocator() : Allocator<T>()
    {
    }
    virtual ~MklAllocator() = default;
    virtual T* allocate(size_t size) override 
    {
        return (T*)mkl_malloc(size * sizeof(T), 64);
    }
    virtual void deallocate(T* p) override
    {
        if (p)
            mkl_free(p);
    }

};

template<typename T>
class Matrix_
{
public:
    Matrix_(size_t rows, size_t columns, const std::shared_ptr<Allocator<T>>& allocator) :
        rows_(rows),
        columns_(columns),
        allocator_(allocator), 
        data_(allocator_->allocate(size_), [this](T* p) {this->allocator_->deallocate(p); })
    {
        //data_.reset(allocator_->allocate(rows_*columns_));
    }
    Matrix_(size_t rows, size_t columns):
        rows_(rows),
        columns_(columns) {
        allocator_.reset(new MklAllocator<T>());
        data_.reset(allocator_->allocate(rows_*columns_), [this](T* p) {this->allocator_->deallocate(p);});
    }
    Matrix_(){
        allocator_.reset(new MklAllocator<T>());
        data_ = nullptr;
        rows_ = 0;
        columns_ = 0;
    }
    //Matrix_(const Matrix_&) = delete;

    void resize(size_t rows, size_t columns) {
        rows_ = rows;
        columns_ = columns;
        data_.reset(allocator_->allocate(rows_*columns_), [this](T* p) {this->allocator_->deallocate(p); });
    }
    //cp data to entry storage
    void fill(const T* data) {
        T* p = data_.get();
        for (size_t i = 0; i < rows_ * columns_; ++i)
        {
            p[i] = data[i];
        }
    }

    ~Matrix_() = default;

//protected:
    std::shared_ptr<Allocator<T>> allocator_;
    std::shared_ptr<T> data_;
    size_t rows_;
    size_t columns_;
};


template<typename T>
class Vector_
{
public:
    Vector_(size_t size, const std::shared_ptr<Allocator<T>>& allocator) :
        size_(size), 
        allocator_(allocator),
        data_(allocator_->allocate(size_), [this](T* p) {this->allocator_->deallocate(p);})
        
    {
    }

    explicit Vector_(size_t size) :
        size_(size), 
        allocator_(new MklAllocator<T>()), 
        data_(allocator_->allocate(size_), [this](T* p) {this->allocator_->deallocate(p); })
    {
    }
    //Vector_(const Vector_&) = delete;
    Vector_()
    {
        size_ = 0;
        data_.reset();
        allocator_.reset(new MklAllocator<T>());
    }

    Vector_(const Vector_& v):
        size_(v.size_),
        allocator_(new MklAllocator<T>()),
        data_(allocator_->allocate(size_), [this](T* p) { this->allocator_->deallocate(p); })
    {
        fill(v.data_.get());
    }
    Vector_(size_t size, const T* data): 
        size_(size), 
        allocator_(new MklAllocator<T>()), 
        data_(allocator_->allocate(size_), [this](T* p) { this->allocator_->deallocate(p); })
    {
        fill(data);
    }

    void resize(size_t size) {
        if (data_.get())
            allocator_->deallocate(data_.get());
        size_ = size;
        data_.reset(allocator_->allocate(size_), [this](T* p) {this->allocator_->deallocate(p); });
    }

    void fill(const T* data) {
        T* p = data_.get();
        for (size_t i = 0; i < size_; ++i)
        {
            p[i] = data[i];
        }
    }
    void fill(T const_) {
        T* p = data_.get();
        for (size_t i = 0; i < size_; ++i)
        {
            p[i] = const_;
        }
    }
    ~Vector_() = default;

//protected:
    size_t size_;
    std::shared_ptr<Allocator<T>> allocator_;
    std::shared_ptr<T> data_;
};

template<typename T>
using Mat = Matrix_<T>;
template<typename T>
using Vec = Vector_<T>;

using Vecf = Vector_<double>;
using Matf = Matrix_<double>;


//template<typename T>
//Vec<T> operator*(const Mat<T>& m, const Vec<T>& v) {
//    return Vecf();
//}

template<typename T>
void MULT(const Matrix_<T>& m, const Vector_<T>& v, Vector_<T>& o_v) {
    static const CBLAS_LAYOUT layout_ = CBLAS_LAYOUT::CblasRowMajor;
    static const CBLAS_TRANSPOSE transpose_ = CBLAS_TRANSPOSE::CblasNoTrans;
    static const double alpha_ = 1.;
    static const int incX_ = 1;
    static const double beta_ = 0;
    static const int incY_ = 1;

    if (m.columns_ != v.size_)
        throw std::logic_error("m.columns_ != v.size_");
    cblas_dgemv(layout_, transpose_, m.rows_, m.columns_, alpha_, m.data_.get(), m.columns_, v.data_.get(), incX_, beta_, o_v.data_.get(), incY_);
}

template<typename T>
void MULT(const Vector_<T>& v, const Matrix_<T>& m, Vector_<T>& o_v) {
    static const CBLAS_LAYOUT layout_ = CBLAS_LAYOUT::CblasRowMajor;
    static const CBLAS_TRANSPOSE transpose_ = CBLAS_TRANSPOSE::CblasTrans;
    static const double alpha_ = 1.;
    static const int incX_ = 1;
    static const double beta_ = 0;
    static const int incY_ = 1;

    if (m.rows_ != v.size_)
        throw std::logic_error("m.columns_ != v.size_");
    cblas_dgemv(layout_, transpose_, m.rows_, m.columns_, alpha_, m.data_.get(), m.columns_, v.data_.get(), incX_, beta_, o_v.data_.get(), incY_);
}

template <class T>
Vector_<T> GET_TMP_VEC(const Matrix_<T>& m){
    Vector_<T> v(m.rows_);
    v.data_ = m.data_;
    return v;
}

template <class T>
Matrix_<T> GET_TMP_MAT(const Vector_<T>& v, bool Transpose=false) {
    int r = v.size_;
    int c = 1;
    if (Transpose) {
        r = 1;
        c = v.size_;
    }
    Matrix_<T> m(r, c);
    m.data_ = v.data_;
    return m;
}

template<typename T>
void MULT(const Vector_<T>& v1, const Vector_<T>& v2, Matrix_<T>& o_m) {
    //FIX
    Matrix_<T> m1 = GET_TMP_MAT(v1);
    Matrix_<T> m2 = GET_TMP_MAT(v2, true);
    
    MULT(m1, m2, o_m);
}

template<typename T>
void MULT(const Matrix_<T>& m1, const Matrix_<T>& m2, Matrix_<T>& o_m) {
    static const CBLAS_LAYOUT layout_ = CBLAS_LAYOUT::CblasRowMajor;
    static const CBLAS_TRANSPOSE transpose_ = CBLAS_TRANSPOSE::CblasNoTrans;
    static const double alpha_ = 1.f;
    static const double beta_ = 0.f;

    if (m1.columns_ != m2.rows_)
        throw std::logic_error("m.columns_ != m.rows");
    cblas_dgemm(layout_, transpose_, 
        transpose_, m1.rows_, m2.columns_, 
        m1.columns_, alpha_, m1.data_.get(), 
        m1.columns_, m2.data_.get(), m2.columns_, 
        beta_, o_m.data_.get(), o_m.columns_);
}

template<typename T>
void ADD(const Matrix_<T>& m1, Matrix_<T>& m2, double alpha_ = 1.f) {
    //static const double alpha_ = 1;
    static const double beta_ = 1;
    static const int incX_ = 1;
    static const int incY_ = 1;
    if ((m1.rows_ != m2.rows_) || (m1.columns_ != m2.columns_))
        throw std::logic_error("(m1.rows_ != m2.rows_) || (m1.columns_ != m2.colums_)");
    int size = m1.columns_ * m1.rows_;
 
    cblas_daxpby(size, alpha_, m1.data_.get(),
        incX_, beta_, m2.data_.get(), incY_);
}

template<typename T>
void ADD(const Vector_<T>& v1, Vector_<T>& v2, double alpha_ = 1.f) {
    //static const double alpha_ = 1;
    static const double beta_ = 1;
    static const int incX_ = 1;
    static const int incY_ = 1;
    if(v1.size_ != v2.size_)
        throw std::logic_error("v1.size_ != v2.size_");
    cblas_daxpby(v1.size_, alpha_, v1.data_.get(), 
        incX_, beta_,v2.data_.get(), incY_);
}

template<typename T>
void FUNC_APPLY(const Vector_<T>& src, const std::function<T(T)>& func, Vector_<T>& dst) {
    if (src.size_ != dst.size_)
        throw std::logic_error("src.size != dst.size");
    T* s_p = src.data_.get();
    T* d_p = dst.data_.get();

    for (size_t i = 0; i < src.size_; i++)
    {
        d_p[i] = func(s_p[i]);
    }
}



#include <ostream>
template<class T>
std::ostream& operator<<(std::ostream& os, const Matrix_<T>& m)
{
    T* p = m.data_.get();
    os << std::endl;
    for (size_t i = 0; i < m.rows_; i++)
    {
        for (size_t j = 0; j < m.columns_; j++)
        {
            os << p[i*m.columns_ + j]<<" ";
        }
        os << std::endl;

    }
    return os;
}
template<class T>
std::ostream& operator<<(std::ostream& os, const Vector_<T>& v)
{
    T* p = v.data_.get();
  
        for (size_t j = 0; j < v.size_; j++)
        {
            os << p[j] << " ";
        }
        os << std::endl;

   
    return os;
}

template<class T>
void TENS_MULT(const Matrix_<T>& m1, const Matrix_<T>& m2, Matrix_<T>& o_m) {
    if (m1.rows_ != m2.rows_)
        throw std::logic_error("m1.rows_ != m2.rows_");
    if (m1.columns_ != 1 || m2.columns_ != 1)
        throw std::logic_error("m1.columns_ !=1 || m2.columns_ != 1");

    T* p1 = m1.data_.get();
    T* p2 = m2.data_.get();
    if (m1.rows_ != o_m.rows_ || o_m.columns_ != 1)
        o_m.resize(m1.rows_, 1);

    T* r = o_m.data_.get();

    for (size_t i = 0; i < o_m.rows_; i++)
    {
        r[i] = p1[i] * p2[i];
    }
}

template<class T>
void TENS_MULT(const Vector_<T>& v1, const Vector_<T>& v2, Vector_<T>& o_v) {
    if (v1.size_ != v2.size_)
        throw std::logic_error("v1.size_ != v2.size_");
    if (o_v.size_ != v1.size_)
        o_v.resize(v1.size_);
    Matrix_<T> m1 = GET_TMP_MAT(v1);
    Matrix_<T> m2 = GET_TMP_MAT(v2);
    Matrix_<T> o_m = GET_TMP_MAT(o_v);
    TENS_MULT(m1, m2, o_m);
}

Vecf operator*(double f, const Vecf& v) {
    Vecf r(v.size_);
    for (size_t i = 0; i < v.size_; i++)
    {
        r.data_.get()[i] = v.data_.get()[i] * f;
    }
    return r;
}

Vecf operator*(const Vecf& v, double f) {
    Vecf r(v.size_);
    for (size_t i = 0; i < v.size_; i++)
    {
        r.data_.get()[i] = v.data_.get()[i] * f;
    }
    return r;
}


Vecf L2(const Vecf& x, const Vecf& y) {
    if (x.size_ != y.size_)
        throw std::logic_error("x.size_ != y.size_");
    Vecf r(x.size_);
    for (size_t i = 0; i < x.size_; i++)
    {
        r.data_.get()[i] = 2 * (x.data_.get()[i] - y.data_.get()[i]);
    }
    return r;
}


/*
class DataMat
{
public:
    class Iterator
    {
    public:
        Iterator(double* p, size_t step = 0) :p_(p), step_(step) {

        }
        ~Iterator() = default;
        Iterator operator++() {
            return p_ += step_;
        }
        bool operator!=(const Iterator& i) {
            return p_ != i.p_;
        }

        bool operator*() const {
        }

        double *p_;
        size_t step_;
    };

    DataMat(size_t count, size_t size, double* data) {
               
    }
    Iterator begin() const {
        return Iterator(data_.get(), columns_);
    };
    Iterator end() const {
        return Iterator(data_.get() + columns_ * rows_);
    };
    virtual ~DataMat() = default;

private:
    
   
   
};
*/