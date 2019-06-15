#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <string>
#include <exception>

#define protected public
#include <torch/script.h>
#include <torch/csrc/autograd/engine.h>
#undef protected
#include "pytorchinternals.h"

#include <ceres/ceres.h>
#include <glog/logging.h>
#include "angle_local_parameterization.h"

// #define DISABLE_EASY_PROFILER
#include <easy/profiler.h>

#include "json.hpp"
using json = nlohmann::json;

using mod_ptr = std::shared_ptr<torch::jit::script::Module>;

#include "shared_pool.hpp"

struct lock_policy {
    using mutex_type = std::mutex;
    using lock_type = std::lock_guard<mutex_type>;
};

using mod_pool = recycle::shared_pool<torch::jit::script::Module, lock_policy>;

class ModuleCost : public ceres::CostFunction {
public:
    ModuleCost(std::string name,
               mod_pool& mod_pool,
               std::unordered_map<std::string, torch::jit::IValue> constant_params,
               std::vector<int> pbs,
               int num_residuals)
        : _mod_pool(mod_pool), _constant_params(constant_params), _name(name) {
        mutable_parameter_block_sizes()->assign(pbs.begin(), pbs.end());
        set_num_residuals(num_residuals);
    }

    virtual bool
    Evaluate(double const* const* parameters,
             double* residuals,
             double** jacobians) const {
        EASY_FUNCTION(profiler::colors::Magenta);
        bool requires_grad = jacobians != NULL;

        EASY_BLOCK("creating params");
        std::vector<torch::jit::IValue> inputs;
        for (unsigned int i = 0; i < parameter_block_sizes().size(); i++) {
            int param_size = parameter_block_sizes()[i];
            torch::ArrayRef<double> ref(parameters[i], param_size);
            auto inp = torch::tensor(ref, torch::dtype(torch::kFloat64).requires_grad(false));
            if (requires_grad) {
                inp = inp.repeat({num_residuals(), 1});
                inp.set_requires_grad(true);
            }
            inputs.push_back(inp);
        }
        EASY_END_BLOCK;

        torch::Tensor residual;
        {
            auto mod = _mod_pool.allocate();

            EASY_BLOCK("creating constant params");
            for (auto& it : _constant_params) {
                mod->parameter_slot(it.first).setValue(it.second);
            }
            EASY_END_BLOCK;

            if (!requires_grad) {
                // easy case first
                EASY_BLOCK("forward only", profiler::colors::Blue);
                residual = mod->forward(inputs).toTensor();
                for (int i = 0; i < num_residuals(); i++) {
                    residuals[i] = residual[i].item<double>();
                }
                EASY_END_BLOCK;
                return true;
            } else {
                EASY_BLOCK("forward grad", profiler::colors::Blue);
                std::vector<torch::Tensor> each_residual;
                for (int i = 0; i < num_residuals(); i++) {
                    std::vector<torch::jit::IValue> these_inputs;
                    for (auto& inp : inputs) {
                        these_inputs.push_back(inp.toTensor()[i]);
                    }
                    each_residual.push_back(mod->forward(these_inputs).toTensor());
                }
                for (int i = 0; i < num_residuals(); i++) {
                    residuals[i] = each_residual[0][i].item<double>();
                }
                residual = torch::stack(each_residual);
                EASY_END_BLOCK;

                std::string block_name = "backward ";
                block_name.append(_name);
                EASY_BLOCK(block_name, profiler::colors::Red);
                auto eye = torch::eye(num_residuals(), torch::dtype(torch::kFloat64));
                residual.backward(eye, false, false);
                EASY_END_BLOCK;

                EASY_BLOCK("setting grad", profiler::colors::Green);
                for (unsigned int j = 0; j < parameter_block_sizes().size(); j++) {
                    int param_size = parameter_block_sizes()[j];
                    auto& grad = inputs[j].toTensor().grad();
                    if (grad.defined()) {
                        for (int k = 0; k < param_size; k++) {
                            for (int i = 0; i < num_residuals(); i++) {
                                jacobians[j][i * param_size + k] = grad[i][k].item<double>();
                            }
                        }
                    }
                }
                EASY_END_BLOCK;
            }
        }

        return true;
    }


private:
    mod_pool& _mod_pool;
    std::unordered_map<std::string, torch::jit::IValue> _constant_params;
    std::string _name;
};

class MyIterationCallback : public ceres::IterationCallback {
public:
    ceres::CallbackReturnType
    operator()(const ceres::IterationSummary& summary) {
        EASY_EVENT("iteration", profiler::colors::Cyan);
        return ceres::SOLVER_CONTINUE;
    }
};

void
sneaky_thread_main(torch::autograd::Engine *eng)
{
    // cpu queue
    auto queue = eng->ready_queues[0];
    while (1) {
        auto task = queue->pop();
        if (task.fn_ && !task.base_->has_error_.load()) {
            torch::autograd::GradMode::set_enabled(task.base_->grad_mode_);
            eng->evaluate_function(task);
        }
        auto base_owner = task.base_->owner_;
        if (base_owner == torch::autograd::NO_DEVICE) {
            if (--task.base_->outstanding_tasks_ == 0) {
                std::lock_guard<std::mutex> lock(task.base_->mutex_);
                task.base_->not_done_.notify_all();
            }
        } else {
            throw std::runtime_error("bad owner");
        }
    }
}

int
main(int argc, char *argv[])
{
    EASY_PROFILER_ENABLE;

    std::string dirname(argv[1]);
    printf("started optorch main\n");

    std::string desc_path = dirname + "/description.json";
    std::ifstream desc_f(desc_path);
    json desc;
    desc_f >> desc;

    ceres::Problem::Options problem_opts;
    problem_opts.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_opts);

    std::vector<int> param_sizes;
    std::vector<double *> params;
    for (auto& param : desc["params"]) {
        int size = param["values"].size();
        double *param_ptr = new double[size];
        for (int i = 0; i < size; i++) {
            param_ptr[i] = param["values"][i].get<double>();
        }
        param_sizes.push_back(size);
        params.push_back(param_ptr);
    }

    std::unordered_map<std::string, mod_pool> modules;
    for (auto& module_name : desc["module_names"]) {
        std::string mod_path = dirname + "/" + module_name.get<std::string>() + ".pt";
        auto mod_factory = [mod_path]() {
            mod_ptr mod = torch::jit::load(mod_path);
            mod->to(torch::kFloat64);
            return mod;
        };
        // modules.emplace(module_name.get<std::string>(), mod_factory);
        modules[module_name] = mod_pool(mod_factory);
    }

    EASY_BLOCK("making cost fns");
    for (auto& cost_fn : desc["cost_fns"]) {
        // std::cout << cost_fn["name"] << std::endl;

        // std::string mod_path = dirname + "/" + cost_fn["name"].get<std::string>() + ".pt";
        // mod_ptr mod = torch::jit::load(mod_path);
        // mod->to(torch::kFloat64);

        std::vector<int> pbs;
        std::vector<double *> cost_params;
        for (int pidx : cost_fn["pidxs"]) {
            int param_size = param_sizes[pidx];
            // std::cout << "pidx: " << pidx << " size: " << param_size << std::endl;
            pbs.push_back(param_size);
            cost_params.push_back(params[pidx]);
        }

        int num_residuals = cost_fn["num_residuals"];

        std::string module_name = cost_fn["module"];
        auto& mod_pool = modules[module_name];

        std::unordered_map<std::string, torch::jit::IValue> constant_params;
        for (auto& cp_item : cost_fn["constant_params"].items()) {
            auto& cp = cp_item.value();
            int cp_size = cp.size();
            torch::Tensor cp_tensor;
            if (cp[0].is_array()) {
                int cp_size_2 = cp[0].size();
                cp_tensor = torch::empty({cp_size, cp_size_2}, torch::dtype(torch::kFloat64));
                for (int i = 0; i < cp_size; i++) {
                    for (int j = 0; j < cp_size_2; j++) {
                        cp_tensor[i][j] = cp[i][j].get<double>();
                    }
                }
            } else {
                cp_tensor = torch::empty(cp_size, torch::dtype(torch::kFloat64));
                for (int i = 0; i < cp_size; i++) {
                    cp_tensor[i] = cp[i].get<double>();
                }
            }
            constant_params[cp_item.key()] = cp_tensor;
        }

        // std::cout << "pbs: " << pbs << " nr: " << num_residuals << " cp: " << cost_params << std::endl;
        ModuleCost *cost = new ModuleCost(cost_fn["name"], mod_pool, constant_params, pbs, num_residuals);
        problem.AddResidualBlock(cost, NULL, cost_params);
    }

    for (auto it : desc["local_params"].items()) {
        int pidx = std::stoi(it.key());
        std::string name = it.value();
        if (name == "quat") {
            auto lp = new ceres::QuaternionParameterization();
            problem.SetParameterization(params[pidx], lp);
        } else if (name == "angle") {
            auto lp = AngleLocalParameterization::Create();
            problem.SetParameterization(params[pidx], lp);
        } else {
            throw std::runtime_error("bad local param");
        }
    }

    EASY_END_BLOCK;

    // do some math to make sure torch creates the ready queues
    auto t1 = torch::empty({2, 3}, torch::requires_grad(true));
    auto t2 = t1 * 3;
    t2.backward();

    torch::autograd::Engine *eng = &torch::autograd::Engine::get_default_engine();
    for (int i = 0; i < desc["options"]["num_threads"].get<int>() - 1; i++) {
        std::thread t(&sneaky_thread_main, eng);
        t.detach();
    }

    ceres::Solver::Options opts;
    std::string linear_solver = desc["options"]["linear_solver"];
    if (linear_solver == "dense_qr") {
        opts.linear_solver_type = ceres::DENSE_QR;
    } else if (linear_solver == "sparse_normal_cholesky") {
        opts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    opts.minimizer_progress_to_stdout = desc["options"]["verbose"].get<bool>();
    opts.num_threads = desc["options"]["num_threads"].get<int>();
    opts.max_num_iterations = desc["options"]["max_iterations"].get<int>();
    opts.function_tolerance = desc["options"]["function_tolerance"].get<double>();
    opts.gradient_tolerance = desc["options"]["gradient_tolerance"].get<double>();
    opts.parameter_tolerance = desc["options"]["parameter_tolerance"].get<double>();
    opts.callbacks.push_back(new MyIterationCallback());
    ceres::Solver::Summary summary;
    std::cout << "solving" << std::endl;
    ceres::Solve(opts, &problem, &summary);
    std::cout << "solved" << std::endl;

    if (desc["options"]["verbose"]) {
        std::cout << "summary:" << std::endl;
        std::cout << summary.FullReport() << std::endl;
    }

    std::string output_path = dirname + "/output.json";
    json output = json::array();

    for (unsigned int i = 0; i < param_sizes.size(); i++) {
        int param_size = param_sizes[i];
        double *param_ptr = params[i];
        json param_array = json::array();
        for (int j = 0; j < param_size; j++) {
            param_array.push_back(param_ptr[j]);
        }
        output.push_back(param_array);
    }

    std::ofstream out_f(output_path);
    out_f << output;

    profiler::dumpBlocksToFile("easy.prof");
    // profiler::stopListen();
}
