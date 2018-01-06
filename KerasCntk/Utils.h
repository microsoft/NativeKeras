#pragma once

#include <algorithm>
#include <codecvt>
#include <string>

#include "boost/algorithm/string.hpp"
#include "boost/uuid/uuid.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"

namespace keras
{
    namespace utils
    {
        static inline std::string ToLower(const std::string & str)
        {
            std::string result = str;
            std::transform(result.begin(), result.end(), result.begin(), ::tolower);
            return result;
        }

        static inline std::string ToUpper(const std::string & str)
        {
            std::string result = str;
            std::transform(result.begin(), result.end(), result.begin(), ::toupper);
            return result;
        }

        static inline std::wstring ToWide(const std::string & str)
        {
            std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
            return converter.from_bytes(str);
        }

        static inline std::string ToString(const std::wstring & str)
        {
            std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
            return converter.to_bytes(str);
        }

        static inline std::string GenerateUuid()
        {
            std::string uuid = boost::uuids::to_string(boost::uuids::random_generator()());
            // Remove hyphens
            boost::erase_all(uuid, "-");
            return uuid;
        }

        static inline std::wstring GenerateWideUuid()
        {
            std::wstring uuid = boost::uuids::to_wstring(boost::uuids::random_generator()());
            // Remove hyphens
            boost::erase_all(uuid, "-");
            return uuid;
        }
    }
}
