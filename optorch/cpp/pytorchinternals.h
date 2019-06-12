#include <queue>
#include <torch/csrc/autograd/function.h>

namespace torch { namespace autograd {

static constexpr int NO_DEVICE = -2;

struct GraphTask;

struct FunctionTask {
  GraphTask* base_;
  std::shared_ptr<Function> fn_;
  // This buffer serves as an implicit "addition" node for all of the
  // gradients flowing here.  Once all the dependencies are finished, we
  // use the contents of this buffer to run the function.
  InputBuffer inputs_;

  FunctionTask(GraphTask* base, std::shared_ptr<Function> fn, InputBuffer inputs)
    : base_(base)
    , fn_(std::move(fn))
    , inputs_(std::move(inputs)) {}
};

// Returns true when t2 should be (weakly) BEFORE t1 in the queue.
// Empty FunctionTask are first.
struct CompareFunctionTaskTime {
  bool operator()(FunctionTask const & t1, FunctionTask const & t2) {
    if (!t1.fn_) {
      return false;
    } else if (!t2.fn_) {
      return true;
    } else {
      return t1.fn_->sequence_nr() < t2.fn_->sequence_nr();
    }
  }
};

struct ReadyQueue {
  std::priority_queue<FunctionTask, std::vector<FunctionTask>, CompareFunctionTaskTime> heap_;
  std::condition_variable not_empty_;
  std::mutex mutex_;

  void push(FunctionTask item);
  FunctionTask pop();
};

struct GraphTask {
  std::exception_ptr exception_;
  // Indicates if an error occurred while executing any task.  When this is
  // true, it signals all threads to stop executing.
  std::atomic_bool has_error_;
  std::atomic<uint64_t> outstanding_tasks_;
  bool keep_graph_;
  bool grad_mode_;

  std::mutex mutex_;
  // Notified when a task finishes executing.  Check outstanding_tasks_ to see
  // if all tasks are done.
  std::condition_variable not_done_;
  std::unordered_map<Function*, InputBuffer> not_ready_;
  std::unordered_map<Function*, int> dependencies_;

  struct ExecInfo {
    struct Capture {
      Capture(int input_idx, int output_idx) : input_idx_(input_idx), output_idx_(output_idx) {}
      int input_idx_; // within Function inputs
      int output_idx_; // within the output vector of a GraphTask
    };

    bool should_execute() const {
      return needed_ || captures_;
    }

    bool needed_ = false;
    std::unique_ptr<std::vector<Capture>> captures_;
  };
  // Exec info has a bit complicated semantics. If it's empty, it means the task is
  // run in a "default" mode, which means that all next_edges we encounter should
  // get executed. If it's not empty, only functions that have an entry and this entry
  // has needed == True should be executed.
  // exec_info_.empty() means it's .backward(), otherwise it's .grad().
  std::unordered_map<Function*, ExecInfo> exec_info_;
  std::vector<Variable> captured_vars_;

  void init_to_execute(Function& graph_root, const edge_list& outputs);

  // The value of worker_device in the thread that created this task.
  // See Note [Reentrant backwards]
  int owner_;

  bool can_checkpoint() {
    return exec_info_.empty();
  }

  GraphTask(bool keep_graph, bool grad_mode)
    : has_error_(false)
    , outstanding_tasks_(0)
    , keep_graph_(keep_graph)
    , grad_mode_(grad_mode)
    , owner_(NO_DEVICE) {}
};

}}
