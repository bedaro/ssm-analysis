// Based on https://gist.github.com/mcleary/b0bf4fa88830ff7c882d
#ifndef _TIMER_HPP
#define _TIMER_HPP

#include <chrono>

using std::chrono::steady_clock;
using std::chrono::time_point;

class Timer {
  public:
    void start() {
      m_StartTime = steady_clock::now();
      m_bRunning = true;
    }

    void stop() {
      m_EndTime = steady_clock::now();
      m_bRunning = false;
    }

    double elapsedMilliseconds() {
      time_point<steady_clock> endTime;
      
      if(m_bRunning) {
        endTime = std::chrono::steady_clock::now();
      } else {
        endTime = m_EndTime;
      }

      return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count();
    }
    
    double elapsedSeconds() {
      return elapsedMilliseconds() / 1000.0;
    }

  private:
    time_point<steady_clock> m_StartTime;
    time_point<steady_clock> m_EndTime;
    bool                     m_bRunning = false;
};

#endif
