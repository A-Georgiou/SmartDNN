#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <string>
#include <sstream>

class Logger {
public:
    enum class Level {
        INFO,
        WARNING,
        ERROR
    };

    static void log(Level level, const std::string& message) {
#ifdef ENABLE_LOGGING
        std::ostringstream oss;
        oss << "[" << levelToString(level) << "] " << message;
        std::cout << oss.str() << std::endl;
#else
        (void)level;
        (void)message;
#endif
    }

    template<typename T>
    static void log(Level level, const T& message) {
#ifdef ENABLE_LOGGING
        std::ostringstream oss;
        oss << "[" << levelToString(level) << "] " << message;
        std::cout << oss.str() << std::endl;
#else
        (void)level;
        (void)message;
#endif
    }

private:
    static std::string levelToString(Level level) {
        switch (level) {
            case Level::INFO: return "INFO";
            case Level::WARNING: return "WARNING";
            case Level::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
};

#ifdef ENABLE_LOGGING
#define LOG_INFO(message) Logger::log(Logger::Level::INFO, message)
#define LOG_WARNING(message) Logger::log(Logger::Level::WARNING, message)
#define LOG_ERROR(message) Logger::log(Logger::Level::ERROR, message)
#else
#define LOG_INFO(message) ((void)0)
#define LOG_WARNING(message) ((void)0)
#define LOG_ERROR(message) ((void)0)
#endif

#endif // LOGGER_HPP