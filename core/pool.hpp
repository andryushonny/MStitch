// Minimal thread pool: submit jobs, wait until all are done.
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ms {

class Pool {
public:
    explicit Pool(unsigned n = 0) {
        if (n == 0) n = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned i = 0; i < n; ++i) workers_.emplace_back([this] { run(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    unsigned size() const { return unsigned(workers_.size()); }
    void submit(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(m_);
            q_.push_back(std::move(f));
            ++pending_;
        }
        cv_.notify_one();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(m_);
        idle_.wait(lk, [this] { return pending_ == 0; });
    }

private:
    void run() {
        for (;;) {
            std::function<void()> f;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) return;
                f = std::move(q_.front());
                q_.pop_front();
            }
            f();
            {
                std::lock_guard<std::mutex> lk(m_);
                if (--pending_ == 0) idle_.notify_all();
            }
        }
    }
    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> q_;
    std::mutex m_;
    std::condition_variable cv_, idle_;
    size_t pending_ = 0;
    bool stop_ = false;
};

// Run f(i) for i in [0, n) on all cores.
template <class F>
void parallel_for(size_t n, F f, unsigned threads = 0) {
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    threads = unsigned(std::min<size_t>(threads, n));
    if (threads <= 1) {
        for (size_t i = 0; i < n; ++i) f(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> ts;
    for (unsigned k = 0; k < threads; ++k)
        ts.emplace_back([&] {
            for (size_t i; (i = next.fetch_add(1)) < n;) f(i);
        });
    for (auto& t : ts) t.join();
}

}  // namespace ms
