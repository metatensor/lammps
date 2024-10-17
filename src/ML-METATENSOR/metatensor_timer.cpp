#include <mutex>
#include <iostream>

#include "metatensor_timer.h"

using namespace LAMMPS_NS;

// lock to protect the other static from concurrent modification
static std::mutex METATENSOR_TIMER_MUTEX = {};
// depth of the timer, i.e. how many different timers are alive right now
static int64_t METATENSOR_TIMER_DEPTH = -1;
// strictly increasing timer counter, to know if a new one was created inside
// the scope of the current one.
static uint64_t METATENSOR_TIMER_COUNTER = 0;
// Is profiling enabled?
static bool METATENSOR_TIMER_ENABLED = false;


void MetatensorTimer::enable(bool toggle) {
    auto guard_ = std::lock_guard(METATENSOR_TIMER_MUTEX);

    METATENSOR_TIMER_ENABLED = toggle;
}


MetatensorTimer::MetatensorTimer(std::string name):
    enabled_(false),
    name_(std::move(name))
{
    auto guard_ = std::lock_guard(METATENSOR_TIMER_MUTEX);
    if (METATENSOR_TIMER_ENABLED) {
        METATENSOR_TIMER_DEPTH += 1;
        METATENSOR_TIMER_COUNTER += 1;

        this->enabled_ = true;
        this->starting_counter_ = METATENSOR_TIMER_COUNTER;
        this->start_ = std::chrono::high_resolution_clock::now();
        auto indent = std::string(METATENSOR_TIMER_DEPTH * 3, ' ');

        if (METATENSOR_TIMER_DEPTH == 0) {
            std::cerr << "\n";
        }

        std::cerr << "\n" << indent << this->name_ << " ...";
    }
}

MetatensorTimer::~MetatensorTimer() {
    auto guard_ = std::lock_guard(METATENSOR_TIMER_MUTEX);

    if (METATENSOR_TIMER_ENABLED && this->enabled_) {
        auto stop = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start_).count();

        if (METATENSOR_TIMER_COUNTER != starting_counter_) {
            auto indent = std::string(METATENSOR_TIMER_DEPTH * 3, ' ');
            std::cerr << "\n" << indent << this->name_;
        }

        std::cerr << " took " << elapsed / 1e6 << "ms" << std::flush;
        METATENSOR_TIMER_DEPTH -= 1;
    }
}
