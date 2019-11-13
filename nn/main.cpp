#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <algorithm>
#include <thread>
#include <numeric>
#include "data.h"
#include "network.h"

#define ASSERT(expr) if(!(expr)) throw std::exception(#expr);

class DataReader {
public:
    DataReader(const std::string& path):path_(path){
       
    }
    void read(size_t count, bool have_header=true) {
        data_mat_.reserve(count);
        labels_mat_.reserve(count);
        std::string line;
        std::ifstream myfile(path_);
        size_t readed_count = 0;
        if (myfile.is_open())
        {
            if (have_header)
                std::getline(myfile, line);
            while (std::getline(myfile, line) && readed_count != count)
            {
                parse_line(line);
                ++readed_count;
            }
            myfile.close();
        }
    }
    std::vector<Vecf> data_mat_;
    std::vector<Vecf> labels_mat_;
private:
    std::string path_;
    void parse_line(std::string& s) {
        Vecf label(10);
        label.fill(0.f);
        size_t last_pos = 0;
        size_t pos = s.find(',', last_pos);
        size_t index = std::stoi(s.substr(last_pos, pos-last_pos));
        label.data_.get()[index] = 1;
        labels_mat_.emplace_back(label);
        //std::cout << "label vec:" << label;

        Vecf data(784);
        double * p = data.data_.get();
        size_t i = 0;
        do
        {
            last_pos = pos + 1;
            pos = s.find(',', last_pos);
            p[i++] = std::stof(s.substr(last_pos, pos-last_pos));
        } while (pos != std::string::npos);
        double mean = std::accumulate(p, p + data.size_, 0.) / double(data.size_);
        ;
        double max_mean = *std::max_element(p, p + data.size_) -mean;
        std::for_each(p, p + data.size_, [mean, max_mean](double& x) {x -= mean; x /= max_mean; });
        //std::cout << "data vec:" << data;
        data_mat_.emplace_back(data);
    }
};

class MyWeightInitializer : public WeightInitializer
{
public:
    MyWeightInitializer() : WeightInitializer() {

    }
    virtual void initialize(const Matf& weights) {
        std::for_each(weights.data_.get(),
            weights.data_.get() + weights.rows_ * weights.columns_,
            [this](double& x) {a += 0.1f; x = a; });
    };
    static WeightInitializer::Ptr create() {
        return WeightInitializer::Ptr(new MyWeightInitializer);
    }
    virtual ~MyWeightInitializer() = default;
    double a = 1.f;

};

bool test_matrix_multiplication2() {
    Matf m1(2, 1);
    Matf m2(1, 3);
    Matf m(2, 3);
    double m1_d[] = { 1, 2 };
    double m2_d[] = { 1, 2, 3 };
    m1.fill(m1_d);
    m2.fill(m2_d);

    MULT(m1, m2, m);
    std::cout << m;
    //std::cout << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << std::endl;


    return true;
}

bool test_matrix_multiplication() {
    Matf m0(2, 1);
    double m0_d[] = { 1, 2 };
    m0.fill(m0_d);
    Matf m1(2, 1);
    double m1_d[] = { 3, 4 };
    m1.fill(m1_d);
    Matf m2(2, 1);
    TENS_MULT(m0, m1, m2);

    Matf m3(1, 3);
    double m3_d[] = { 5, 6, 7 };
    m3.fill(m3_d);

    Matf m(2, 3);
    MULT(m2, m3, m);
    std::cout << m;


    //std::cout << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << std::endl;


    return true;
}

bool test_matrix_multiplication1() {
    Matf m(4, 5);
    Vecf v(5);
    double m_d[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    double v_d[] = { 1, 2, 3, 4, 5 };
    m.fill(m_d);
    v.fill(v_d);
    Vecf res(4);
    double res_d[] = { 0,0,0,0 };
    res.fill(res_d);
    MULT(m, v, res);
    double * p = res.data_.get();
    //std::cout << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << std::endl;
    ASSERT(10 == 9);

    return (p[0] == 55 && p[1] == 130 && p[2] == 205 && p[3] == 280);
}

bool test_add() {

    double v_d1[] = { 1, 2, 3, 4, 5 };
    double v_d2[] = { 2, 3, 4, 5, 6 };
    Vecf v1(5, v_d1);
    Vecf v2(5, v_d2);
    ADD(v1, v2);
    double *p = v2.data_.get();
    ASSERT(p[0] == 3 && p[1] == 5 && p[2] == 7 && p[3] == 9 && p[4] == 11);
    std::cout << v2;

    return true;
}

/*bool infer_test() {
    Layer::Ptr l1 = std::shared_ptr<FC_Layer>(new FC_Layer(3, 3,
        [](double x) {return (x*x); },
        [](const Vecf& x) { return 2 * x; },
        MyWeightInitializer::create()));

    Layer::Ptr l2 = std::shared_ptr<FC_Layer>(new FC_Layer(3, 2, [](double x) {return (x*x); }, [](const Vecf& x) { return 2 * x; }));

    Network n([](const Vecf& x, const Vecf& y) { return L2(x, y); });
    n.addLayer(l1);
    n.addLayer(l2);

    double data[] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
    Vecf inf_r;
    n.infer(Vecf(3, data), inf_r);
    double* p = inf_r.data_.get();
    ASSERT(p[0] == 11664.f && p[1] == 11664.f);
    return false;
}*/

/*bool network_test1() {

    Layer::Ptr l1 = std::shared_ptr<FC_Layer>(
        new FC_Layer(3, 3,
            [](double x) {return (x*x); },
            [](const Vecf& x) { return 2 * x; },
            MyWeightInitializer::create()));

    Layer::Ptr l2 = std::shared_ptr<FC_Layer>(
        new FC_Layer(3, 2,
            [](double x) {return 1.f / (1.f + exp(-x)); },
            [](const Vecf& x) { return 2 * x; },
            MyWeightInitializer::create()));

    Network n([](const Vecf& x, const Vecf& y) { return L2(x, y); });
    n.addLayer(l1);
    n.addLayer(l2);

    double data[] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
    double labels[] = { 1, 0, 1, 0 , 0, 1 };
    std::vector<Vecf> data_mat, labels_mat;

    for (double *d = data, *l = labels; d != data + 3 * 3; d += 3, l += 2)
    {
        data_mat.push_back(Vecf(3, d));
        labels_mat.push_back(Vecf(2, l));
    }

    n.train(data_mat, labels_mat);

    return true;
}*/


bool network_test() {
    auto sigmoid = [](const Vecf& x, Vecf& y) {
        double* px = x.data_.get();
        double* py = y.data_.get();

        for (size_t i = 0; i < x.size_; i++)
            py[i] = 1. / (1. + exp(-px[i]));

    };
    auto sigmoid_derivative = [](const Vecf& x, Vecf& y) {
        double* py = y.data_.get();
        double* px = x.data_.get();

        for (size_t i = 0; i < x.size_; i++)
        {
            double s = 1. / (1. + exp(-px[i]));
            py[i] = s * (1 - s);
        }
    };

    auto softmax = [](const Vecf& x, Vecf& y) {
        double* px = x.data_.get();
        double* py = y.data_.get();

        // TODO rewrite with mkl sum
        double denom = 0;
        for (size_t i = 0; i < x.size_; i++)
            denom += exp(px[i]);

        for (size_t i = 0; i < x.size_; i++)
            py[i] = exp(px[i]) / denom; 
    };
    auto softmax_derivative = [](const Vecf& x, Vecf& y) 
    {
        double* px = x.data_.get();
        double* py = y.data_.get();

        // TODO rewrite with mkl sum
        double denom = 0;
        for (size_t i = 0; i < x.size_; i++)
            denom += exp(px[i]);

        for (size_t i = 0; i < x.size_; i++) {
            double s = (exp(px[i]) / denom);
            py[i] = s * (1 - s);
        }
    };

    auto cross_entropy_derivative = [](const Vecf& x, const Vecf& y) {

        Vecf res(x.size_);
        double* px = x.data_.get();
        double* py = y.data_.get();
        double* pr = res.data_.get();

        for (size_t i = 0; i < x.size_; i++) {
            pr[i] = py[i] - px[i];
        }
        return res;
    };

    Layer::Ptr l1 = std::shared_ptr<FC_Layer>(
        new FC_Layer(784, 300,
            sigmoid,
            sigmoid_derivative,
            RandomWeightInitializer::create()));

    Layer::Ptr l2 = std::shared_ptr<FC_Layer>(
        new FC_Layer(300, 10,
            softmax,
            softmax_derivative,
            RandomWeightInitializer::create()));

    Network n(cross_entropy_derivative, 0.1);
    n.addLayer(l1);
    n.addLayer(l2);

    DataReader train_data_reader("C:\\Users\\kkoryun\\repository\\dl\\mnist_train.csv");
    train_data_reader.read(50000);
    n.train(train_data_reader.data_mat_, train_data_reader.labels_mat_, 50000);

    DataReader test_data_reader("C:\\Users\\kkoryun\\repository\\dl\\mnist_test.csv");
    test_data_reader.read(10000);
    n.test(test_data_reader.data_mat_, test_data_reader.labels_mat_);


    return true;
}

bool test1() {
    MklAllocator<double> f;
    f.allocate(10);
    return true;
}

typedef std::function<bool()> TestFunc;
void check_test(std::string desc, const TestFunc& test_func) {
    bool success = false;
    try
    {
        success = test_func();
    }
    catch (const std::exception& e)
    {
        std::cout << "Failed test: " << desc.c_str() << std::endl << e.what() << std::endl;
        return;
    }
    if (success) {
        std::cout << "Success test: " << desc.c_str() << std::endl;
    }
    else {
        std::cout << "Failed test: " << desc.c_str() << std::endl;
    }
}

int main() {
    //check_test("check test", test1);
    //check_test("test_matrix_multiplication", test_matrix_multiplication);
    check_test("network_test", network_test);
    //check_test("test add", test_add);


    return 0;
}